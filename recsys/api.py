"""
uvicorn recsys:api --reload
"""
import os
from typing import List

import faiss
import pandas as pd
import pkg_resources
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.logger import logger
from gensim import corpora, models

from recsys.recsys import Output, Query, recommend_simple

app = FastAPI()
df = pd.read_pickle(pkg_resources.resource_stream(__name__, "metadata_simple/data.pkl"))
dictionary = corpora.Dictionary.load(os.path.join(os.path.dirname(__file__), "metadata_simple/dictionary"))
w2v = models.Word2Vec.load(os.path.join(os.path.dirname(__file__), "metadata_simple/w2v"))
tfidf = models.TfidfModel.load(os.path.join(os.path.dirname(__file__), "metadata_simple/tfidf"))
lsi = models.LsiModel.load(os.path.join(os.path.dirname(__file__), "metadata_simple/lsi"))
index = faiss.read_index(os.path.join(os.path.dirname(__file__), "metadata_simple/faiss.index"))


@app.post("/", response_model=List[Output])
async def root(query: Query):
    return recommend_simple(query.query, query.limit, df, dictionary, w2v, tfidf, lsi, index)
