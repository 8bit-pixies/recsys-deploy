import os
import random
import string
from typing import List

import faiss
import fasttext
import gensim
import numpy as np
import pandas as pd
import pkg_resources
from gensim import corpora, models
from pydantic import BaseModel, validator
from scipy.spatial import distance

from recsys import utils


class Query(BaseModel):
    query: List[str] = []
    limit: int = 5

    @validator("limit")
    def limit_is_positive(cls, v):
        if v <= 0:
            raise ValueError("limit needs to be positive integer")
        return v


class Output(BaseModel):
    tag: str
    score: float


def recommend_lang(query, k=5, df=None, fasttext_model=None, model=None):
    """
    This is the entry point for launching the API
    """
    NON_LATIN_LANGS = ["ja", "ko", "zh"]
    if fasttext_model is None:
        fasttext_model = fasttext.load_model(os.path.join(os.path.dirname(__file__), "metadata_dual/lid.176.bin"))
    if model is None:
        model = {
            "en": (
                corpora.Dictionary.load(os.path.join(os.path.dirname(__file__), "metadata_dual/dictionary_en")),
                models.Word2Vec.load(os.path.join(os.path.dirname(__file__), "metadata_dual/w2v_en")),
                models.TfidfModel.load(os.path.join(os.path.dirname(__file__), "metadata_dual/tfidf_en")),
                models.LsiModel.load(os.path.join(os.path.dirname(__file__), "metadata_dual/lsi_en")),
                faiss.read_index(os.path.join(os.path.dirname(__file__), "metadata_dual/faiss_en.index")),
            ),
            "other": (
                corpora.Dictionary.load(os.path.join(os.path.dirname(__file__), "metadata_dual/dictionary_other")),
                models.Word2Vec.load(os.path.join(os.path.dirname(__file__), "metadata_dual/w2v_other")),
                models.TfidfModel.load(os.path.join(os.path.dirname(__file__), "metadata_dual/tfidf_other")),
                models.LsiModel.load(os.path.join(os.path.dirname(__file__), "metadata_dual/lsi_other")),
                faiss.read_index(os.path.join(os.path.dirname(__file__), "metadata_dual/faiss_other.index")),
            ),
        }
    lang = fasttext_model.predict([" ".join(query)])[0][0][0][-2:]
    if lang in NON_LATIN_LANGS:
        query_clean = query
        lang = "other"
    else:
        query_clean = utils.preprocess_tags(query)
        lang = "en"

    dictionary, w2v, tfidf, lsi, index = model[lang]

    if df is None:
        df_lang = utils.load_reference_data(lang, fasttext_model)
    else:
        df_lang = df[df["lan_split"] == lang]
    return generate_recommendations(
        query, k=k, df=df_lang, lang=lang, dictionary=dictionary, w2v=w2v, tfidf=tfidf, lsi=lsi, index=index
    )


def generate_recommendations(
    query, k=5, df=None, fasttext_model=None, lang=None, dictionary=None, w2v=None, tfidf=None, lsi=None, index=None
):
    if df is None:
        df = utils.load_reference_data(lang, fasttext_model)
    if dictionary is None or w2v is None or tfidf is None or lsi is None or index is None or lang is None:
        raise Exception("you should use the higher order function?")

    query_clean = utils.preprocess_tags(query)
    bow_query = [dictionary.doc2bow(query_clean)]
    output_query = lsi[tfidf[bow_query]]

    d = index.d
    xt = gensim.matutils.corpus2csc(output_query, num_terms=d).A.T.astype(np.float32)
    D, I = index.search(xt, k)
    suggested = df.iloc[I.flatten()].copy()
    suggested["score"] = D.flatten() * 100
    suggested["tag"] = suggested.tags.apply(lambda x: [y for y in x.split(",") if y not in query])

    # now flatten and return top 5
    output = suggested.explode("tag").groupby("tag").first().reset_index()
    if output.shape[0] < k:
        # have to search more...
        D, I = index.search(xt, k * 50)
        suggested = df.iloc[I.flatten()].copy()
        suggested["score"] = D.flatten() * 100
        suggested["tag"] = suggested.tags.apply(lambda x: [y for y in x.split(",") if y not in query])

        # now flatten and return top 5
        output = suggested.explode("tag").groupby("tag").first().reset_index()

    # calculate tag_list average
    mean_query = [w2v.wv[x] for x in query if x in w2v.wv.key_to_index]
    if len(mean_query) == 0:
        return output.head(k)[["tag", "score"]].to_dict(orient="records")
    else:
        mean_query = np.mean(mean_query, axis=0)
        try:
            w2v_tag = np.stack([w2v.wv[x] if x in w2v.wv.key_to_index else mean_query for x in output["tag"]], 0)
        except:
            w2v_tag = mean_query.reshape(1, -1)
        mean_query = mean_query.reshape(1, -1)
        w2v_dist = distance.cdist(mean_query, w2v_tag).flatten()
        output["score"] = output["score"] * (w2v_dist + 1)

        # sort by score and output
        output = output.sort_values(by=["score"])
        return output.head(k)[["tag", "score"]].to_dict(orient="records")
