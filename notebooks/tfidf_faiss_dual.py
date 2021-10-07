"""
Usage:

https://github.com/facebookresearch/faiss
https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html

Running servers:
https://github.com/facebookresearch/faiss/blob/main/demos/demo_client_server_ivf.py

We separate our model into two, using fasttext as a poc to show how we can split languages and load models
"""
import os
import string
import timeit
from typing import List

import faiss
import fasttext
import gensim
import joblib
import numpy as np
import pandas as pd
import tqdm
from colorama import Fore, Style
from gensim import corpora, models
from scipy.spatial import distance
from sklearn.decomposition import MiniBatchSparsePCA

DICTIONARY_PATH = "notebooks/model_dual/dictionary"
TFIDF_PATH = "notebooks/model_dual/tfidf"
LSI_PATH = "notebooks/model_dual/lsi"
FAISS_PATH = "notebooks/model_dual/faiss"
W2V_PATH = "notebooks/model_dual/w2v"
FASTTEST_MODEL = "notebooks/model_dual/lid.176.bin"

fmodel = fasttext.load_model(FASTTEST_MODEL)
# fmodel.predict([text2])  # ([['__label__en']], [array([0.9331119], dtype=float32)]

# this one we create one model for all languages.
BATCH_SIZE = 2 ** 9  # for faiss training
MAX_DIM = 800  # for gensim_lsi
MAX_DIM_NLIST = 800  # for faiss
MAX_TOKENS = 1000

# just going to pretend zh-cn and zh-tw are the same...
SUPPORTED_LANGS = ["de", "en", "es", "fr", "id", "it", "ja", "ko", "nl", "pl", "pt", "ru", "tr", "zh"]
NON_LATIN_LANGS = ["ja", "ko", "zh"]


def preprocess_tags(tag: List[str]):
    # remove punctuation
    return [
        " ".join(
            gensim.utils.simple_preprocess(x.replace("_", " ").translate(str.maketrans("", "", string.punctuation)))
        )
        for x in tag
    ]


def load_reference_data(lang=None, write_pickle=False):
    # there are different languages - the naive approach is to train a different
    # model for each language; we'll deal with that detail later -
    df = pd.read_csv("data/tags_on_posts_sample.csv")[["tags", "lang"]]
    # ignore countries....for now
    df["lan"] = df["lang"].apply(lambda x: x[:2])
    df["predict_lan"] = pd.Series(fmodel.predict(df["tags"].tolist())[0]).apply(lambda x: x[0][-2:])
    df["lan_split"] = df["predict_lan"].apply(lambda x: "en" if x not in NON_LATIN_LANGS else "other")
    df["clean_tags"] = [preprocess_tags(document.split(",")) for document in df.tags]

    if write_pickle:
        df.to_pickle("notebooks/model_dual/data.pkl")
    if lang is None:
        return df
    else:
        return df[df["lan_split"] == lang]


def train_or_load_model(lang="en"):
    df = load_reference_data()
    df = df[df["lan_split"] == lang]
    # texts = [preprocess_tags(document.split(",")) for document in df.tags]
    texts = df.clean_tags.tolist()
    print("... next")
    if not os.path.exists(DICTIONARY_PATH + "_" + lang):
        dictionary = corpora.Dictionary(texts)
        dictionary.filter_extremes(no_below=2)
        dictionary.compactify()
        dictionary.save(DICTIONARY_PATH + "_" + lang)
    else:
        dictionary = corpora.Dictionary.load(DICTIONARY_PATH + "_" + lang)

    calculated_max_dim = min(min(MAX_DIM, max(dictionary.dfs.keys())), len(texts))
    calculated_nlist = min(calculated_max_dim, MAX_DIM_NLIST)
    bow_corpus = [dictionary.doc2bow(text) for text in texts]

    if not os.path.exists(W2V_PATH + "_" + lang):
        print(
            f"..calculated dictionary size (w2v): {Style.BRIGHT + Fore.BLUE}{max(dictionary.cfs.keys())}{Style.RESET_ALL}."
        )
        w2v = models.Word2Vec(texts, vector_size=calculated_max_dim)
        w2v.save(W2V_PATH + "_" + lang)
    else:
        w2v = models.Word2Vec.load(W2V_PATH + "_" + lang)

    if not os.path.exists(TFIDF_PATH):
        print(f"..calculated dictionary size: {Style.BRIGHT + Fore.BLUE}{max(dictionary.cfs.keys())}{Style.RESET_ALL}.")
        tfidf = models.TfidfModel(bow_corpus)
        tfidf.save(TFIDF_PATH + "_" + lang)
    else:
        print(f"..calculated dictionary size: {Style.BRIGHT + Fore.BLUE}{max(dictionary.cfs.keys())}{Style.RESET_ALL}.")
        tfidf = models.TfidfModel.load(TFIDF_PATH + "_" + lang)

    if not os.path.exists(LSI_PATH + "_" + lang):
        print(
            f"..finished tfidf...now training lsi with {Style.BRIGHT + Fore.BLUE}{calculated_max_dim}{Style.RESET_ALL}"
        )
        lsi = models.LsiModel(
            num_topics=calculated_max_dim,
            chunksize=BATCH_SIZE * 2,
            power_iters=2,
            id2word=dictionary,
            dtype=np.float32,
        )
        tfidf_corpus = tfidf[bow_corpus]
        # chunk and merge - to avoid weird errors...
        for idx_set in tqdm.tqdm(range(len(tfidf_corpus) // BATCH_SIZE)):
            if idx_set == 0:
                lsi_proj = models.lsimodel.Projection(
                    m=max(dictionary.cfs.keys()) + 1,
                    k=calculated_max_dim,
                    docs=tfidf_corpus[idx_set * BATCH_SIZE : (idx_set + 1) * (BATCH_SIZE)],
                )
            elif idx_set == (df.shape[0] // BATCH_SIZE):
                try:
                    lsi_temp = models.lsimodel.Projection(
                        m=max(dictionary.cfs.keys()) + 1,
                        k=calculated_max_dim,
                        docs=tfidf_corpus[idx_set * BATCH_SIZE :],
                    )
                    lsi_proj.merge(lsi_temp)
                except Exception as e:
                    print(e)
            else:
                try:
                    lsi_temp = models.lsimodel.Projection(
                        m=max(dictionary.cfs.keys()) + 1,
                        k=calculated_max_dim,
                        docs=tfidf_corpus[idx_set * BATCH_SIZE : (idx_set + 1) * (BATCH_SIZE)],
                    )
                    lsi_proj.merge(lsi_temp)
                except Exception as e:
                    print(e)
        lsi.projection = lsi_proj
        lsi.save(LSI_PATH + "_" + lang)
    else:
        lsi = models.LsiModel.load(LSI_PATH + "_" + lang)
    # need to dump and save tfidf, lsi, dictionary
    output = lsi[tfidf[bow_corpus]]
    # usage: gensim.matutils.corpus2csc(output[:5])
    # just don't try dumping everything or we'll have issues

    if not os.path.exists(FAISS_PATH + "_" + lang + ".index"):
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
            # print(output_array.shape)
            index.add(output_array)

        faiss.write_index(index, FAISS_PATH + "_" + lang + ".index")
    else:
        index = faiss.read_index(FAISS_PATH + "_" + lang + ".index")
    return dictionary, w2v, tfidf, lsi, index


def generate_recommendation_language(
    query, k=5, df=None, model={"en": train_or_load_model("en"), "other": train_or_load_model("other")}
):
    lang = fmodel.predict([" ".join(query)])[0][0][0][-2:]
    if lang in NON_LATIN_LANGS:
        query_clean = query
        lang = "other"
    else:
        query_clean = preprocess_tags(query)
        lang = "en"

    dictionary, w2v, tfidf, lsi, index = model[lang]

    if df is None:
        df_lang = load_reference_data(lang)
    else:
        df_lang = df[df["lan_split"] == lang]
    return generate_recommendations(
        query, k=k, df=df_lang, lang=lang, dictionary=dictionary, w2v=w2v, tfidf=tfidf, lsi=lsi, index=index
    )


def generate_recommendations(
    query, k=5, df=None, lang=None, dictionary=None, w2v=None, tfidf=None, lsi=None, index=None
):
    if df is None:
        df = load_reference_data(lang)
    if dictionary is None or w2v is None or tfidf is None or lsi is None or index is None or lang is None:
        raise Exception("you should use the higher order function?")
        dictionary, w2v, tfidf, lsi, index = train_or_load_model("en")
        dictionary_other, w2v_other, tfidf_other, lsi_other, index_other = train_or_load_model("other")

    query_clean = preprocess_tags(query)
    bow_query = [dictionary.doc2bow(query_clean)]
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
        # .head(k)[["tag", "score"]]
    )

    # calculate tag_list average
    mean_query = [w2v.wv[x] for x in query if x in w2v.wv.key_to_index]
    if len(mean_query) == 0:
        return output.head(k)[["tag", "score"]].to_dict(orient="records")
    else:
        mean_query = np.mean(mean_query, axis=0)
        w2v_tag = np.stack([w2v.wv[x] if x in w2v.wv.key_to_index else mean_query for x in output["tag"]], 0)
        mean_query = mean_query.reshape(1, -1)
        w2v_dist = distance.cdist(mean_query, w2v_tag).flatten()
        output["score"] = output["score"] * (w2v_dist + 1)

        # sort by score and output
        output = output.sort_values(by=["score"])
        return output.head(k)[["tag", "score"]].to_dict(orient="records")


# benchmark
def benchmark_function(df, model):
    import random

    indx = random.choice(range(df.shape[0]))
    query = df.iloc[indx].tags.split(",")
    k = random.choice(range(5, 10))
    # df = load_reference_data
    # dictionary, tfidf, lsi, index = train_or_load_model()
    return generate_recommendation_language(query, k, df, model)


if __name__ == "__main__":
    df = load_reference_data()
    model = {
        "en": train_or_load_model("en"),
        "other": train_or_load_model("en"),
    }

    t = timeit.Timer("benchmark_function(df, model)", globals=globals())
    # print(t.repeat(repeat=5, number=100))
    # this is fast enough
    # [1.1555010260001382, 1.888966438999887, 1.614154453999845, 1.390791598000078, 1.4472856519998913]

    # print a sample:
    print(generate_recommendation_language(["dog", "park"], 5, df, model))
