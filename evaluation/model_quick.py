"""
This is a small training script to demonstrate how our processes work.
We will try to have the whole "object" placed into a joblib item
so that it can be uploaded to Github as a release(?)

Or at least easily distributed to platforms/email

Once this is done, we will also create a regular size one which reflects the
current MVP2 version.

This should run very quickly - no caching required.
"""

import os
import string
import timeit
from difflib import SequenceMatcher
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
from gensim.parsing.porter import PorterStemmer
from scipy.spatial import distance
from sklearn.model_selection import train_test_split


def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()


# putting these parameters here for now...
REC_MODEL = "notebooks/model_quick/model.joblib"
FASTTEST_MODEL = "notebooks/model_quick/lid.176.ftz"
INDEX_CORE_MODEL = "notebooks/model_quick/core.index"
INDEX_OTHER_MODEL = "notebooks/model_quick/other.index"
NON_CORE_LANG = ["ja", "ko", "zh"]  # renaming models to be "core" and "other"
MAX_DIM = 1024
BATCH_SIZE = 1024


def preprocess_tags(tag: List[str]):
    # remove punctuation
    p = PorterStemmer()
    stripped_list = [
        gensim.utils.simple_preprocess(p.stem(x.replace("_", " ").translate(str.maketrans("", "", string.punctuation))))
        for x in tag
    ]
    return [val for sublist in stripped_list for val in sublist]


def load_train_test_data(lang=None, output="train"):
    df = load_reference_data(lang)
    X_train, X_test = train_test_split(df, random_state=42)
    if output == "train":
        return X_train
    if output == "test":
        return X_test
    return X_train, X_test


def load_reference_data(lang=None):
    # there are different languages - the naive approach is to train a different
    # model for each language; we'll deal with that detail later -
    fmodel = fasttext.load_model(FASTTEST_MODEL)
    df = pd.read_csv("data/tags_on_posts_sample.csv")[["tags", "lang"]]
    # ignore countries....for now
    df["lan"] = df["lang"].apply(lambda x: x[:2])
    df["predict_lan"] = pd.Series(fmodel.predict(df["tags"].tolist())[0]).apply(lambda x: x[0][-2:])
    df["lan_split"] = df["predict_lan"].apply(lambda x: "core" if x not in NON_CORE_LANG else "other")
    df["clean_tags"] = [preprocess_tags(document.split(",")) for document in df.tags]
    df["split_tags"] = [document.split(",") for document in df.tags]

    if lang is None:
        return df
    else:
        return df[df["lan_split"] == lang]


def train_or_load_model(lang="core"):
    # df = load_reference_data()
    df = load_train_test_data(output="train")
    df = df[df["lan_split"] == lang]

    bag_of_tags = df.clean_tags
    dictionary = corpora.Dictionary(bag_of_tags)
    dictionary.filter_extremes(no_below=5)
    dictionary.compactify()
    f"..calculated dictionary size (w2v): {Style.BRIGHT + Fore.BLUE}{max(dictionary.cfs.keys())}{Style.RESET_ALL}."
    bow_corpus = [dictionary.doc2bow(text) for text in bag_of_tags]
    tfidf = models.TfidfModel(bow_corpus)
    lsi = models.LsiModel(
        num_topics=MAX_DIM,
        chunksize=BATCH_SIZE * 2,
        power_iters=2,
        id2word=dictionary,
        dtype=np.float32,
    )
    tfidf_corpus = tfidf[bow_corpus]
    # chunk and merge - to avoid weird errors...
    # set this to like 5 to speed up training...
    for idx_set in tqdm.tqdm(range(len(tfidf_corpus) // BATCH_SIZE)[:5]):
        if idx_set == 0:
            lsi_proj = models.lsimodel.Projection(
                m=max(dictionary.cfs.keys()) + 1,
                k=MAX_DIM,
                docs=tfidf_corpus[idx_set * BATCH_SIZE : (idx_set + 1) * (BATCH_SIZE)],
            )
        elif idx_set == (df.shape[0] // BATCH_SIZE):
            try:
                lsi_temp = models.lsimodel.Projection(
                    m=max(dictionary.cfs.keys()) + 1,
                    k=MAX_DIM,
                    docs=tfidf_corpus[idx_set * BATCH_SIZE :],
                )
                lsi_proj.merge(lsi_temp)
            except Exception as e:
                print(e)
        else:
            try:
                lsi_temp = models.lsimodel.Projection(
                    m=max(dictionary.cfs.keys()) + 1,
                    k=MAX_DIM,
                    docs=tfidf_corpus[idx_set * BATCH_SIZE : (idx_set + 1) * (BATCH_SIZE)],
                )
                lsi_proj.merge(lsi_temp)
            except Exception as e:
                print(e)
    lsi.projection = lsi_proj

    w2v = models.Word2Vec(bag_of_tags, vector_size=MAX_DIM)

    # this is for querying - save as the baseline if none of the tags have generated vectors
    df = load_reference_data()  # index all?
    bag_of_tags = df.clean_tags
    bow_corpus = [dictionary.doc2bow(text) for text in bag_of_tags]
    index = faiss.IndexFlatL2(MAX_DIM)
    output = lsi[tfidf[bow_corpus]]
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

    if lang == "core":
        faiss.write_index(index, INDEX_CORE_MODEL)
    else:
        faiss.write_index(index, INDEX_OTHER_MODEL)
    model = {"df": df, "dictionary": dictionary, "tfidf": tfidf, "lsi": lsi, "w2v": w2v}  # , "index": index}
    return model


def recsys(query, limit, model):
    """
    See cli.py or api.py - don't call this directly

    Parameters
    ----------

    query: List[str] - list of tags
    limit: int - the number of tags to recommend
    model: Dict - dictionary object which defines the model object.

    The format of the dictionary is as follows:

    {
        <language>: {
            'dictionary': gensim dictionary,
            'tfidf': gensim tfidf,
            'lsi': gensim lsi,
            'w2v': gensim w2v,
            'df': pandas database of tags used as reference,
            'index': faiss - approximate nearest neighbour learned,
        }
    }
    """
    lang = model["fasttext"].predict([" ".join(query)])[0][0][0][-2:]
    lang = "core" if lang not in NON_CORE_LANG else "other"
    query_clean = preprocess_tags(query)
    model_sub = model[lang]
    dictionary = model_sub["dictionary"]
    tfidf = model_sub["tfidf"]
    lsi = model_sub["lsi"]
    w2v = model_sub["w2v"]
    df = model_sub["df"]
    index = model_sub["index"]

    bow_query = [dictionary.doc2bow(query_clean)]
    output_query = lsi[tfidf[bow_query]]

    d = index.d
    xt = gensim.matutils.corpus2csc(output_query, num_terms=d).A.T.astype(np.float32)
    D, I = index.search(xt, limit)
    suggested = df.iloc[I.flatten()].copy()
    suggested["score"] = D.flatten()
    suggested["tag"] = suggested.tags.apply(lambda x: [y for y in x.split(",") if y not in query])

    # now flatten and return top k
    # we need exception handling or for it to do something if it returns less than k
    output = suggested.explode("tag").groupby("tag").first().reset_index()[["tag", "score"]]
    output["score"] /= output["score"].max()
    output["score"] *= 100
    output["score"] = np.nan_to_num(output["score"], nan=100.0)

    target_limit = int(output.shape[0] * (output.shape[0] / limit * 2)) + 1
    while output.shape[0] < limit:
        # do stuff here to expand.
        D, I = index.search(xt, target_limit * 2)
        suggested = df.iloc[I.flatten()].copy()
        suggested["score"] = D.flatten()
        suggested["tag"] = suggested.tags.apply(lambda x: [y for y in x.split(",") if y not in query])
        output_next = suggested.explode("tag").groupby("tag").first().reset_index()[["tag", "score"]]
        target_limit = target_limit * 2
        output_next["score"] += output["score"].max()
        output = pd.concat([output, output_next])
        output = output.groupby("tag").first().reset_index()[["tag", "score"]]
        output["score"] /= output["score"].max()
        output["score"] *= 100
        output["score"] = np.nan_to_num(output["score"], nan=1e-6)
        if output.shape[0] >= limit:
            output = output.sort_values("score").reset_index(drop=True).head(limit)

    # calculate tag_list average
    mean_query = [w2v.wv[x] for x in query if x in w2v.wv.key_to_index]
    if len(mean_query) == 0:
        return output.head(limit)[["tag", "score"]].to_dict(orient="records")
    else:
        mean_query = np.mean(mean_query, axis=0)
        try:
            w2v_tag = np.stack([w2v.wv[x] if x in w2v.wv.key_to_index else mean_query for x in output["tag"]], 0)
        except:
            w2v_tag = mean_query.reshape(1, -1)
        mean_query = mean_query.reshape(1, -1)
        w2v_dist = distance.cdist(mean_query, w2v_tag).flatten()
        output["score"] = output["score"] * (np.nan_to_num(w2v_dist, nan=0.0) + 1)

        # sort by score and output
        output = output.sort_values(by=["score"])
        return output.head(limit)[["tag", "score"]].to_dict(orient="records")


if __name__ == "__main__":
    model_core = train_or_load_model("core")
    model_other = train_or_load_model("other")
    model = {
        "core": model_core,
        "other": model_other,
    }
    joblib.dump(model, "notebooks/model_quick/model.joblib")
    del model

    model_loaded = joblib.load("notebooks/model_quick/model.joblib")
    model_loaded["fasttext"] = fasttext.load_model(FASTTEST_MODEL)
    model_loaded["core"]["index"] = faiss.read_index(INDEX_CORE_MODEL)
    model_loaded["other"]["index"] = faiss.read_index(INDEX_OTHER_MODEL)

    threshold = 0.8  # tag similarity for it to be "same"
    df_test = load_train_test_data(output="test")
    rnd = np.random.RandomState(42)
    num_tags_to_sample = rnd.randint(1, high=3, size=df_test.shape[0])
    output_stats = []
    for idx, row in enumerate(tqdm.tqdm(df_test.to_dict(orient="records"))):
        tags = list(set(row["tags"].split(",")))
        if len(tags) == 1:
            continue
        if len(tags) > 2:
            num_sample = num_tags_to_sample[idx]
        else:
            num_sample = 1
        remove_tags = rnd.choice(tags, num_sample)
        input_tags = [x for x in tags if x not in remove_tags]
        pred_tags = [x["tag"] for x in recsys(input_tags, limit=5, model=model_loaded)]
        metric = False
        for input_tag in input_tags:
            for pred_tag in pred_tags:
                if similar(input_tag, pred_tag) >= threshold:
                    metric = True
                    break

        output_stats.append(metric)

    print("performance on test: ", np.mean(output_stats))
    # 0.196
