"""
uvicorn recsys:api --reload
"""
import os
from typing import List

import faiss
import fasttext
import pandas as pd
import pkg_resources
from fastapi import FastAPI
from fastapi.logger import logger
from gensim import corpora, models

from recsys.recsys import Output, Query, recommend_lang

app = FastAPI()

df = pd.read_pickle(pkg_resources.resource_stream(__name__, "metadata_dual/data.pkl"))
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
fasttext_model = fasttext.load_model(os.path.join(os.path.dirname(__file__), "metadata_dual/lid.176.bin"))


@app.post("/", response_model=List[Output])
async def root(query: Query):
    return recommend_lang(query.query, query.limit, df=df, fasttext_model=fasttext_model, model=model)
