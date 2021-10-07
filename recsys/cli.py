import argparse
import json
import os

import faiss
import fasttext
import pandas as pd
import pkg_resources
from gensim import corpora, models

from recsys.recsys import Query, recommend_lang

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

    # load dataframe, fasttext mode, and model
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
    print("\n")
    print(recommend_lang(input_data.query, input_data.limit, df=df, fasttext_model=fasttext_model, model=model))
