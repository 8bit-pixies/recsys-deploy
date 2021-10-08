"""
uvicorn recsys:api --reload
"""
import os
from typing import List

import fasttext
import joblib
from fastapi import FastAPI

from recsys.recsys import Output, Query, recsys

app = FastAPI()

model_loaded = joblib.load(os.path.join(os.path.dirname(__file__), "model_quick/model.joblib"))
model_loaded["fasttext"] = fasttext.load_model(os.path.join(os.path.dirname(__file__), "model_quick/lid.176.ftz"))


@app.post("/", response_model=List[Output])
async def root(query: Query):
    return recsys(query.query, query.limit, model_loaded)
