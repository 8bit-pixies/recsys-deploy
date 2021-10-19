"""
uvicorn recsys:api --reload
"""
import os
from typing import List

import faiss
import fasttext
import joblib
from fastapi import FastAPI

from recsys.recsys import Output, Query, recsys

app = FastAPI()

model_loaded = joblib.load(os.path.join(os.path.dirname(__file__), "model_quick/model.joblib"))
model_loaded["fasttext"] = fasttext.load_model(os.path.join(os.path.dirname(__file__), "model_quick/lid.176.ftz"))
model_loaded["core"]["index"] = faiss.read_index(os.path.join(os.path.dirname(__file__), "model_quick/core.index"))
model_loaded["other"]["index"] = faiss.read_index(os.path.join(os.path.dirname(__file__), "model_quick/other.index"))


@app.post("/", response_model=List[Output])
async def root(query: Query):
    output = recsys(query.query, limit=query.limit, model=model_loaded)
    return output
