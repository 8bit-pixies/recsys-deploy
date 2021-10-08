import argparse
import json
import os

import fasttext
import joblib

from recsys.recsys import Query, recsys

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data", nargs="*")

    args = parser.parse_args()
    input_data = Query(**json.loads(" ".join(args.data)))

    model_loaded = joblib.load(os.path.join(os.path.dirname(__file__), "model_quick/model.joblib"))
    model_loaded["fasttext"] = fasttext.load_model(os.path.join(os.path.dirname(__file__), "model_quick/lid.176.ftz"))
    print(recsys(["dogs", "dog park"], limit=5, model=model_loaded))
    print(recsys(["广州"], limit=5, model=model_loaded))
