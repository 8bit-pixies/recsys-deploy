import argparse
import json
import os

import faiss
import pandas as pd
import pkg_resources
from gensim import corpora, models

from recsys.recsys import Query, recommend_simple

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data", nargs="*")

    args = parser.parse_args()
    input_data = Query(**json.loads(" ".join(args.data)))

    # print(recommend_random([]))
    # df = pd.read_pickle(pkg_resources.resource_stream(__name__, "metadata_simple/data.pkl"))
    # dictionary = corpora.Dictionary.load(os.path.join(os.path.dirname(__file__), "metadata_simple/dictionary"))
    # tfidf = models.TfidfModel.load(os.path.join(os.path.dirname(__file__), "metadata_simple/tfidf"))
    # lsi = models.LsiModel.load(os.path.join(os.path.dirname(__file__), "metadata_simple/lsi"))
    # index = faiss.read_index(os.path.join(os.path.dirname(__file__), "metadata_simple/faiss.index"))
    print(recommend_simple(input_data.query, input_data.limit))
