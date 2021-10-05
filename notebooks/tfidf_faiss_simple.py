"""
Usage:

https://github.com/facebookresearch/faiss
https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html

Running servers:
https://github.com/facebookresearch/faiss/blob/main/demos/demo_client_server_ivf.py

This one doesn't separate by language, just to simplify deployment on my 8gb ram machine
"""
from gensim import models
from gensim import corpora
import gensim
import pandas as pd
import numpy as np
import faiss
from colorama import Fore, Style
import tqdm

import os
import timeit

DICTIONARY_PATH = "notebooks/model_simple/dictionary"
TFIDF_PATH = "notebooks/model_simple/tfidf"
LSI_PATH = "notebooks/model_simple/lsi"
FAISS_PATH = "notebooks/model_simple/faiss.index"

# this one we create one model for all languages.
BATCH_SIZE = 2 ** 9  # for faiss training
MAX_DIM = 2 ** 7  # for gensim_lsi
MAX_DIM_NLIST = 2 ** 7  # for faiss


def load_reference_data(write_pickle=False):
    # there are different languages - the naive approach is to train a different
    # model for each language; we'll deal with that detail later -
    df = pd.read_csv("data/tags_on_posts_sample.csv")[["tags", "lang"]]
    # ignore countries....for now
    df["lan"] = df["lang"].apply(lambda x: x[:2])
    if write_pickle:
        df.to_pickle("notebooks/model_simple/data.pkl")
    return df


def train_or_load_model():
    df = load_reference_data()
    texts = [document.split(",") for document in df.tags]  # no processing as these are the raw tags
    if os.path.exists(DICTIONARY_PATH):
        dictionary = corpora.Dictionary(texts)
        dictionary.save(f"notebooks/model_simple/dictionary")
    else:
        dictionary = corpora.Dictionary.load(DICTIONARY_PATH)

    calculated_max_dim = min(min(MAX_DIM, max(dictionary.dfs.keys())), len(texts))
    calculated_nlist = min(calculated_max_dim, MAX_DIM_NLIST)
    bow_corpus = [dictionary.doc2bow(text) for text in texts]

    if not os.path.exists(TFIDF_PATH):
        print(f"..calculated dictionary size: {Style.BRIGHT + Fore.BLUE}{max(dictionary.cfs.keys())}{Style.RESET_ALL}.")
        tfidf = models.TfidfModel(bow_corpus)
        tfidf.save(TFIDF_PATH)
    else:
        tfidf = models.TfidfModel.load(TFIDF_PATH)

    if not os.path.exists(LSI_PATH):
        print(
            f"..finished tfidf...now training lsi with {Style.BRIGHT + Fore.BLUE}{calculated_max_dim}{Style.RESET_ALL}"
        )
        lsi = models.LsiModel(
            tfidf[bow_corpus],
            num_topics=calculated_max_dim,
            chunksize=BATCH_SIZE,
            power_iters=1,
            dtype=np.float32,
            extra_samples=0,
        )
        lsi.save(LSI_PATH)
    else:
        lsi = models.LsiModel.load(LSI_PATH)
    # need to dump and save tfidf, lsi, dictionary
    output = lsi[tfidf[bow_corpus]]
    # usage: gensim.matutils.corpus2csc(output[:5])
    # just don't try dumping everything or we'll have issues

    if not os.path.exists(FAISS_PATH):
        print(f"..using index FlatL2: {calculated_nlist}.")
        index = faiss.IndexFlatL2(calculated_nlist)
        # split into batches..

        for idx_set in tqdm.tqdm(range(df.shape[0] // BATCH_SIZE)):
            if idx_set == (df.shape[0] // BATCH_SIZE):
                output_array = gensim.matutils.corpus2csc(output[idx_set * BATCH_SIZE :], num_terms=index.d).T.A.astype(
                    np.float32
                )
            else:
                output_array = gensim.matutils.corpus2csc(
                    output[idx_set * BATCH_SIZE : (idx_set + 1) * (BATCH_SIZE)], num_terms=index.d
                ).T.A.astype(np.float32)
        index.add(output_array)

        faiss.write_index(index, FAISS_PATH)
    else:
        index = faiss.read_index(FAISS_PATH)
    return dictionary, tfidf, lsi, index


def generate_recommendations(query, k=5, df=None, dictionary=None, tfidf=None, lsi=None, index=None):
    if df is None:
        df = load_reference_data()
    if dictionary is None or tfidf is None or lsi is None or index is None:
        dictionary, tfidf, lsi, index = train_or_load_model()

    bow_query = [dictionary.doc2bow(query)]
    output_query = lsi[tfidf[bow_query]]

    d = index.d
    xt = gensim.matutils.corpus2csc(output_query, num_terms=d).A.T.astype(np.float32)
    D, I = index.search(xt, k)
    suggested = df.iloc[I.flatten()].copy()
    suggested["score"] = D.flatten() * 100
    suggested["tag"] = suggested.tags.apply(lambda x: [y for y in x.split(",") if y not in query])

    # now flatten and return top 5
    output = (
        suggested.explode("tag")
        .groupby("tag")
        .first()
        .reset_index()
        .head(k)[["tag", "score"]]
        .to_dict(orient="records")
    )
    return output


# benchmark
def benchmark_function(df, dictionary, tfidf, lsi, index):
    import random

    indx = random.choice(range(df.shape[0]))
    query = df.iloc[indx].tags.split(",")
    k = random.choice(range(5, 10))
    # df = load_reference_data
    # dictionary, tfidf, lsi, index = train_or_load_model()
    return generate_recommendations(query, k, df, dictionary, tfidf, lsi, index)


if __name__ == "__main__":
    df = load_reference_data()
    dictionary, tfidf, lsi, index = train_or_load_model()

    t = timeit.Timer("benchmark_function(df, dictionary, tfidf, lsi, index)", globals=globals())
    print(t.repeat(repeat=5, number=100))
    # this is fast enough
    # [1.1555010260001382, 1.888966438999887, 1.614154453999845, 1.390791598000078, 1.4472856519998913]
