import argparse
import json
import os

import faiss
import fasttext
import joblib

from recsys.recsys import Query, recsys

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data", nargs="*")

    model_loaded = joblib.load(os.path.join(os.path.dirname(__file__), "model_quick/model.joblib"))
    model_loaded["fasttext"] = fasttext.load_model(os.path.join(os.path.dirname(__file__), "model_quick/lid.176.ftz"))
    model_loaded["core"]["index"] = faiss.read_index(os.path.join(os.path.dirname(__file__), "model_quick/core.index"))
    model_loaded["other"]["index"] = faiss.read_index(
        os.path.join(os.path.dirname(__file__), "model_quick/other.index")
    )

    args = parser.parse_args()
    if args.data:
        input_data = Query(**json.loads(" ".join(args.data)))
        print(recsys(input_data.query, limit=input_data.limit, model=model_loaded))
    else:
        print(recsys(["dogs", "dog park"], limit=5, model=model_loaded))
        print(recsys(["广州"], limit=5, model=model_loaded))
