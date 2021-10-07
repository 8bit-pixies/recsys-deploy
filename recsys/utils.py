import string
from typing import List

import gensim
import pandas as pd


def load_reference_data(lang=None, fasttext_model=None):
    # there are different languages - the naive approach is to train a different
    # model for each language; we'll deal with that detail later -
    NON_LATIN_LANGS = ["ja", "ko", "zh"]
    df = pd.read_csv("data/tags_on_posts_sample.csv")[["tags", "lang"]]
    # ignore countries....for now
    df["lan"] = df["lang"].apply(lambda x: x[:2])
    df["predict_lan"] = pd.Series(fasttext_model.predict(df["tags"].tolist())[0]).apply(lambda x: x[0][-2:])
    df["lan_split"] = df["predict_lan"].apply(lambda x: "en" if x not in NON_LATIN_LANGS else "other")
    df["clean_tags"] = [preprocess_tags(document.split(",")) for document in df.tags]

    if lang is None:
        return df
    else:
        return df[df["lan_split"] == lang]


def preprocess_tags(tag: List[str]):
    # remove punctuation
    return [
        " ".join(
            gensim.utils.simple_preprocess(x.replace("_", " ").translate(str.maketrans("", "", string.punctuation)))
        )
        for x in tag
    ]
