import os
import random
import string
from typing import List

import faiss
import gensim
import numpy as np
import pandas as pd
import pkg_resources
from gensim import corpora, models
from pydantic import BaseModel, validator


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


def recommend_random(query: List[str], limit: int = 5) -> List[str]:
    if len(query) < limit:
        # generate random strings
        for _ in range(limit - len(query)):
            query.append("".join(random.sample(string.ascii_letters, 7)))
    output = query[:limit]
    return output


def recommend_simple(
    query: List[str], limit: int = 5, df=None, dictionary=None, tfidf=None, lsi=None, index=None
) -> List[str]:
    # load all the models etc...
    if df is None:
        df = pd.read_pickle(pkg_resources.resource_stream(__name__, "metadata_simple/data.pkl"))
    if dictionary is None or tfidf is None or lsi is None or index is None:
        dictionary = corpora.Dictionary.load(os.path.join(os.path.dirname(__file__), "metadata_simple/dictionary"))
        tfidf = models.TfidfModel.load(os.path.join(os.path.dirname(__file__), "metadata_simple/tfidf"))
        lsi = models.LsiModel.load(os.path.join(os.path.dirname(__file__), "metadata_simple/lsi"))
        index = faiss.read_index(os.path.join(os.path.dirname(__file__), "metadata_simple/faiss.index"))

    bow_query = [dictionary.doc2bow(query)]
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
        .head(limit)[["tag", "score"]]
        .to_dict(orient="records")
    )
    return output


def recommend_lang(query: List[str], limit: int = 5) -> List[str]:
    if len(query) < limit:
        # generate random strings
        for _ in range(limit - len(query)):
            query.append("".join(random.sample(string.ascii_letters, 7)))
    output = query[:limit]
    return output


if __name__ == "__main__":
    pass
