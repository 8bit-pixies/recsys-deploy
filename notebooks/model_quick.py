"""
This is a small training script to demonstrate how our processes work.
We will try to have the whole "object" placed into a joblib item
so that it can be uploaded to Github as a release(?)

Or at least easily distributed to platforms/email

Once this is done, we will also create a regular size one which reflects the
current MVP2 version.

This should run very quickly - no caching required.
"""

import os
import string
import timeit
from typing import List

import faiss
import fasttext
import gensim
import joblib
import numpy as np
import pandas as pd
import tqdm
from colorama import Fore, Style
from gensim import corpora, models
from gensim.parsing.porter import PorterStemmer
from scipy.spatial import distance

# putting these parameters here for now...
REC_MODEL = "notebooks/model_quick/model.joblib"
FASTTEST_MODEL = "notebooks/model_quick/lid.176.ftz"
INDEX_CORE_MODEL = "notebooks/model_quick/core.index"
INDEX_OTHER_MODEL = "notebooks/model_quick/other.index"
NON_CORE_LANG = ["ja", "ko", "zh"]  # renaming models to be "core" and "other"
MAX_DIM = 1024
BATCH_SIZE = 1024


def preprocess_tags(tag: List[str]):
    # remove punctuation
    p = PorterStemmer()
    stripped_list = [
        gensim.utils.simple_preprocess(p.stem(x.replace("_", " ").translate(str.maketrans("", "", string.punctuation))))
        for x in tag
    ]
    return [val for sublist in stripped_list for val in sublist]


def load_reference_data(lang=None):
    # there are different languages - the naive approach is to train a different
    # model for each language; we'll deal with that detail later -
    fmodel = fasttext.load_model(FASTTEST_MODEL)
    df = pd.read_csv("data/tags_on_posts_sample.csv")[["tags", "lang"]]
    # ignore countries....for now
    df["lan"] = df["lang"].apply(lambda x: x[:2])
    df["predict_lan"] = pd.Series(fmodel.predict(df["tags"].tolist())[0]).apply(lambda x: x[0][-2:])
    df["lan_split"] = df["predict_lan"].apply(lambda x: "core" if x not in NON_CORE_LANG else "other")
    df["clean_tags"] = [preprocess_tags(document.split(",")) for document in df.tags]
    df["split_tags"] = [document.split(",") for document in df.tags]

    if lang is None:
        return df
    else:
        return df[df["lan_split"] == lang]


def train_or_load_model(lang="core"):
    df = load_reference_data()
    df = df[df["lan_split"] == lang]

    bag_of_tags = df.clean_tags
    dictionary = corpora.Dictionary(bag_of_tags)
    dictionary.filter_extremes(no_below=5)
    dictionary.compactify()
    f"..calculated dictionary size (w2v): {Style.BRIGHT + Fore.BLUE}{max(dictionary.cfs.keys())}{Style.RESET_ALL}."
    bow_corpus = [dictionary.doc2bow(text) for text in bag_of_tags]
    tfidf = models.TfidfModel(bow_corpus)
    lsi = models.LsiModel(
        num_topics=MAX_DIM,
        chunksize=BATCH_SIZE * 2,
        power_iters=2,
        id2word=dictionary,
        dtype=np.float32,
    )
    tfidf_corpus = tfidf[bow_corpus]
    # chunk and merge - to avoid weird errors...
    # set this to like 5 to speed up training...
    for idx_set in tqdm.tqdm(range(len(tfidf_corpus) // BATCH_SIZE)[:5]):
        if idx_set == 0:
            lsi_proj = models.lsimodel.Projection(
                m=max(dictionary.cfs.keys()) + 1,
                k=MAX_DIM,
                docs=tfidf_corpus[idx_set * BATCH_SIZE : (idx_set + 1) * (BATCH_SIZE)],
            )
        elif idx_set == (df.shape[0] // BATCH_SIZE):
            try:
                lsi_temp = models.lsimodel.Projection(
                    m=max(dictionary.cfs.keys()) + 1,
                    k=MAX_DIM,
                    docs=tfidf_corpus[idx_set * BATCH_SIZE :],
                )
                lsi_proj.merge(lsi_temp)
            except Exception as e:
                print(e)
        else:
            try:
                lsi_temp = models.lsimodel.Projection(
                    m=max(dictionary.cfs.keys()) + 1,
                    k=MAX_DIM,
                    docs=tfidf_corpus[idx_set * BATCH_SIZE : (idx_set + 1) * (BATCH_SIZE)],
                )
                lsi_proj.merge(lsi_temp)
            except Exception as e:
                print(e)
    lsi.projection = lsi_proj

    w2v = models.Word2Vec(bag_of_tags, vector_size=MAX_DIM)

    # this is for querying - save as the baseline if none of the tags have generated vectors
    index = faiss.IndexFlatL2(MAX_DIM)
    output = lsi[tfidf[bow_corpus]]
    for idx_set in tqdm.tqdm(range(df.shape[0] // BATCH_SIZE)):
        if idx_set == (df.shape[0] // BATCH_SIZE):
            output_array = gensim.matutils.corpus2csc(output[idx_set * BATCH_SIZE :], num_terms=index.d).T.A.astype(
                np.float32
            )
        else:
            output_array = gensim.matutils.corpus2csc(
                output[idx_set * BATCH_SIZE : (idx_set + 1) * (BATCH_SIZE)], num_terms=index.d
            ).T.A.astype(np.float32)
        # print(output_array.shape)
        index.add(output_array)

    if lang == "core":
        faiss.write_index(index, INDEX_CORE_MODEL)
    else:
        faiss.write_index(index, INDEX_OTHER_MODEL)
    model = {"df": df, "dictionary": dictionary, "tfidf": tfidf, "lsi": lsi, "w2v": w2v}  # , "index": index}
    return model


def recsys(query, limit, model):
    lang = model["fasttext"].predict([" ".join(query)])[0][0][0][-2:]
    lang = "core" if lang not in NON_CORE_LANG else "other"
    query_clean = preprocess_tags(query)
    model_sub = model[lang]
    dictionary = model_sub["dictionary"]
    tfidf = model_sub["tfidf"]
    lsi = model_sub["lsi"]
    w2v = model_sub["w2v"]
    df = model_sub["df"]
    index = model_sub["index"]

    bow_query = [dictionary.doc2bow(query_clean)]
    output_query = lsi[tfidf[bow_query]]

    d = index.d
    xt = gensim.matutils.corpus2csc(output_query, num_terms=d).A.T.astype(np.float32)
    D, I = index.search(xt, limit)
    suggested = df.iloc[I.flatten()].copy()
    suggested["score"] = D.flatten() * 100
    suggested["tag"] = suggested.tags.apply(lambda x: [y for y in x.split(",") if y not in query])

    # now flatten and return top 5
    output = (
        suggested.explode("tag")
        .groupby("tag")
        .first()
        .reset_index()
        # .head(k)[["tag", "score"]]
    )

    # calculate tag_list average
    mean_query = [w2v.wv[x] for x in query if x in w2v.wv.key_to_index]
    if len(mean_query) == 0:
        return output.head(limit)[["tag", "score"]].to_dict(orient="records")
    else:
        mean_query = np.mean(mean_query, axis=0)
        w2v_tag = np.stack([w2v.wv[x] if x in w2v.wv.key_to_index else mean_query for x in output["tag"]], 0)
        mean_query = mean_query.reshape(1, -1)
        w2v_dist = distance.cdist(mean_query, w2v_tag).flatten()
        output["score"] = output["score"] * (w2v_dist + 1)

        # sort by score and output
        output = output.sort_values(by=["score"])
        return output.head(limit)[["tag", "score"]].to_dict(orient="records")


if __name__ == "__main__":
    model_core = train_or_load_model("core")
    model_other = train_or_load_model("other")
    model = {
        "core": model_core,
        "other": model_other,
    }
    joblib.dump(model, "notebooks/model_quick/model.joblib")
    del model

    model_loaded = joblib.load("notebooks/model_quick/model.joblib")
    model_loaded["fasttext"] = fasttext.load_model(FASTTEST_MODEL)
    model_loaded["core"]["index"] = faiss.read_index(INDEX_CORE_MODEL)
    model_loaded["other"]["index"] = faiss.read_index(INDEX_OTHER_MODEL)
    print(recsys(["dogs", "dog park"], limit=5, model=model_loaded))
    print(recsys(["广州"], limit=5, model=model_loaded))
