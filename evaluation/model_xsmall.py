"""
This is a small training script to demonstrate how our processes work.
We will try to have the whole "object" placed into a joblib item
so that it can be uploaded to Github as a release(?)

Or at least easily distributed via email or otherwise. This one
uses only w2v rather than a pipeline.

Once this is done, we will also create a regular size one which reflects the
current MVP2 version.

This should run very quickly - no caching required.
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

from evaluation.train_test_generator import get_data, evaluate_single, evaluate_fast



# putting these parameters here for now...
REC_MODEL = "notebooks/model_xsmall/model.joblib"
FASTTEST_MODEL = "notebooks/model_xsmall/lid.176.ftz"
NON_CORE_LANG = ["ja", "ko", "zh"]  # renaming models to be "core" and "other"
MAX_DIM = 1024
BATCH_SIZE = 1024


def preprocess_tags(tag: List[str]):
    # remove punctuation
    return [
        " ".join(
            gensim.utils.simple_preprocess(x.replace("_", " ").translate(str.maketrans("", "", string.punctuation)))
        )
        for x in tag
    ]


def preprocess_df(df):
    fmodel = fasttext.load_model(FASTTEST_MODEL)
    df["lan"] = df["lang"].apply(lambda x: x[:2])
    df["predict_lan"] = pd.Series(fmodel.predict(df["tags"].tolist())[0]).apply(lambda x: x[0][-2:])
    df["lan_split"] = df["predict_lan"].apply(lambda x: "core" if x not in NON_CORE_LANG else "other")
    df["clean_tags"] = [preprocess_tags(document.split(",")) for document in df.tags]
    df["split_tags"] = [document.split(",") for document in df.tags]
    return df

def load_reference_data(lang=None,):
    # there are different languages - the naive approach is to train a different
    # model for each language; we'll deal with that detail later -
    df, _ = get_data()
    # ignore countries....for now
    df = preprocess_df(df)

    if lang is None:
        return df
    else:
        return df[df["lan_split"] == lang]


def train_or_load_model(lang="core"):
    df = load_reference_data()
    df = df[df["lan_split"] == lang]

    if lang == "core":
        bag_of_tags = df.clean_tags
    else:
        bag_of_tags = df.split_tags
    w2v = models.Word2Vec(bag_of_tags, vector_size=MAX_DIM)

    # this is for querying - save as the baseline if none of the tags have generated vectors
    base_vector = np.mean([w2v.wv[x] for x in w2v.wv.key_to_index], axis=0)
    # bag_of_tags["vector_of_tags"] = bag_of_tags.apply(
    #     lambda query: np.mean([w2v.wv[x] for x in query if x in w2v.wv.key_to_index], axis=0)
    # )
    # bag_of_tags["vector_of_tags"] = bag_of_tags["vector_of_tags"].apply(
    #     lambda x: base_vector if np.isnan(x).all() else x
    # )

    # index all
    del df
    df = pd.read_csv("data/tags_on_posts_sample.csv")[["tags", "lang"]]
    df = preprocess_df(df)
    df = df[df["lan_split"] == lang]
    if lang == "core":
        bag_of_tags = df.clean_tags
    else:
        bag_of_tags = df.split_tags
    bag_of_tags["vector_of_tags"] = bag_of_tags.apply(
        lambda query: np.mean([w2v.wv[x] for x in query if x in w2v.wv.key_to_index], axis=0)
    )
    bag_of_tags["vector_of_tags"] = bag_of_tags["vector_of_tags"].apply(
        lambda x: base_vector if np.isnan(x).all() else x
    )

    index = faiss.IndexFlatL2(MAX_DIM)
    vector_of_tags = np.stack(bag_of_tags["vector_of_tags"].tolist(), axis=0)
    for idx_set in tqdm.tqdm(range(df.shape[0] // BATCH_SIZE)):
        if idx_set == (df.shape[0] // BATCH_SIZE):
            output_array = vector_of_tags[idx_set * BATCH_SIZE :]
        else:
            output_array = vector_of_tags[idx_set * BATCH_SIZE : (idx_set + 1) * (BATCH_SIZE)]
        index.add(output_array)

    model = {"df": df, "w2v": w2v, "base_vector": base_vector, "index": index}
    return model


def recsys_small(query, limit, model, only_tags = False):
    lang = model["fasttext"].predict([" ".join(query)])[0][0][0][-2:]
    lang = "core" if lang not in NON_CORE_LANG else "other"
    if lang == "core":
        query_clean = preprocess_tags(query)
    else:
        query_clean = query

    model_sub = model[lang]
    w2v = model_sub["w2v"]
    df = model_sub["df"]
    index = model_sub["index"]
    base_vector = model_sub["base_vector"]

    mean_query = [w2v.wv[x] for x in query_clean if x in w2v.wv.key_to_index]
    if len(mean_query) == 0:
        mean_query = base_vector
    else:
        mean_query = np.mean(mean_query, axis=0)

    # do ANN - 
    xt = mean_query.reshape(1, -1)
    D, I = index.search(xt, limit)
    suggested = df.iloc[I.flatten()].copy()
    suggested["score"] = D.flatten()
    suggested["tag"] = suggested.tags.apply(lambda x: [y for y in x.split(",") if y not in query])

    # now flatten and return top 5
    output = suggested.explode("tag").groupby("tag").first().reset_index()
    if output.shape[0] < limit:
        # have to search more...
        D, I = index.search(xt, limit * 50)
        suggested = df.iloc[I.flatten()].copy()
        suggested["score"] = D.flatten()
        suggested["tag"] = suggested.tags.apply(lambda x: [y for y in x.split(",") if y not in query])

        # now flatten and return top 5
        output = suggested.explode("tag").groupby("tag").first().reset_index()

    if only_tags:
        return output.head(limit)["tag"].tolist()
    return output.head(limit)[["tag", "score"]].to_dict(orient="records")


def recsys_batch(tags: List[str], limit=5, model=None):
    df_eval = pd.DataFrame({
        'tags': [','.join(x) for x in tags],  # just to make sure I don't do something silly...
        'tags_eval': tags  # just to make sure I don't do something silly...
    })
    df_eval["predict_lan"] = pd.Series(model['fasttext'].predict(df_eval["tags"].tolist())[0]).apply(lambda x: x[0][-2:])
    df_eval["lan_split"] = df_eval["predict_lan"].apply(lambda x: "core" if x not in NON_CORE_LANG else "other")
    df_eval["clean_tags"] = [preprocess_tags(document.split(",")) for document in df_eval.tags]
    df_eval["split_tags"] = [document.split(",") for document in df_eval.tags]
    df_eval = df_eval.reset_index()  # so I have something to join back on.

    df = pd.read_csv("data/tags_on_posts_sample.csv")[["tags", "lang"]]
    df = preprocess_df(df)

    pred_df = {}

    for lang in ["other", "core"]:
        df_sub = df_eval[df_eval['lan_split'] == lang]
        model_sub = model[lang]

        if lang == "core":
            bag_of_tags = df_sub.clean_tags
        else:
            bag_of_tags = df_sub.split_tags

        w2v = model_sub["w2v"]
        # df = model_sub["df"]
        index = model_sub["index"]
        base_vector = model_sub["base_vector"]

        bag_of_tags["vector_of_tags"] = bag_of_tags.apply(
            lambda query: np.mean([w2v.wv[x] for x in query if x in w2v.wv.key_to_index], axis=0)
        )
        bag_of_tags["vector_of_tags"] = bag_of_tags["vector_of_tags"].apply(
            lambda x: base_vector if np.isnan(x).all() else x
        )

        vector_of_tags = np.stack(bag_of_tags["vector_of_tags"].tolist(), axis=0)
        D, I = index.search(vector_of_tags, limit)
        df_sub['other_index'] = I.tolist()
        # return df_sub, df.reset_index(drop=True).reset_index(), index
        df_sub = df_sub.explode('other_index')
        df_sub = pd.merge(df_sub, df[df['lan_split']==lang].reset_index(drop=True).reset_index(), left_on='other_index', right_on='index')
        # something evaluate_single
        pred_df[lang] = df_sub

    # now join back and do comparison
    return pred_df



if __name__ == "__main__":
    model_core = train_or_load_model("core")
    model_other = train_or_load_model("other")
    model = {
        "core": model_core,
        "other": model_other,
    }
    model["fasttext"] = fasttext.load_model(FASTTEST_MODEL)

    _, df_test = get_data()
    target_size = df_test.shape[0]
    df_test = df_test.sample(df_test.shape[0], axis=0, random_state=42)

    output = recsys_batch(df_test.tags_eval, model=model)  # compare with df_test.tags_drop
    output_stack = pd.concat([output['core'], output['other']])  # output is split_tags_y, join on tags_eval

    output_stack = output_stack[['split_tags_y', 'tags_eval']]
    df_test = df_test[['tags_drop', 'tags_eval']]
    output_stack['join_key'] = output_stack['tags_eval'].apply(lambda x: ','.join(x))
    df_test['join_key'] = df_test['tags_eval'].apply(lambda x: ','.join(x))

    output_stack = output_stack.drop(columns=["tags_eval"])
    df_test = df_test.drop(columns=["tags_eval"])
    df_compare = pd.merge(df_test, output_stack, on='join_key')
    df_compare = df_compare.sample(target_size*2, axis=0, random_state=42)


    eval_score = np.vectorize(evaluate_single)(df_compare['tags_drop'], df_compare['split_tags_y'])
    eval_score = pd.DataFrame({"eval": eval_score})
    eval_score = eval_score[eval_score['eval'] > 0]
    eval_score = eval_score.sample(target_size, axis=0, random_state=42)
    print("MAP@5:\n", eval_score.describe())
    # MAP@5:
    #                 eval
    # count  76296.000000
    # mean       0.289154
    # std        0.197114
    # min        0.012500
    # 25%        0.200000
    # 50%        0.200000
    # 75%        0.333333
    # max        1.000000
    ax = eval_score['eval'].hist()  # s is an instance of Series
    fig = ax.get_figure()
    fig.savefig('evaluation/xsmall.png')