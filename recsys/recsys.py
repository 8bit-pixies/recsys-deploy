import string
from typing import List

import gensim
import numpy as np
import pandas as pd
from gensim.parsing.porter import PorterStemmer
from pydantic import BaseModel, validator
from scipy.spatial import distance

NON_CORE_LANG = ["ja", "ko", "zh"]


class Query(BaseModel):
    query: List[str] = []
    limit: int = 5

    @validator("limit")
    def limit_is_positive(cls, v):
        if v <= 0:
            raise ValueError("limit needs to be positive integer")
        return v


class Output(BaseModel):
    tag: str
    score: float


def preprocess_tags(tag: List[str]):
    # remove punctuation
    p = PorterStemmer()
    stripped_list = [
        gensim.utils.simple_preprocess(p.stem(x.replace("_", " ").translate(str.maketrans("", "", string.punctuation))))
        for x in tag
    ]
    return [val for sublist in stripped_list for val in sublist]


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
    output = (
        suggested.explode("tag")
        .groupby("tag")
        .first()
        .reset_index()
        [["tag", "score"]]
    )

    target_limit = int(output.shape[0] * (output.shape[0] / limit * 2)) + 1
    while output.shape[0] < limit:
        # do stuff here to expand.
        D, I = index.search(xt, target_limit * 2)
        suggested = df.iloc[I.flatten()].copy()
        suggested["score"] = D.flatten()
        suggested["tag"] = suggested.tags.apply(lambda x: [y for y in x.split(",") if y not in query])
        output_next = (
            suggested.explode("tag")
            .groupby("tag")
            .first()
            .reset_index()
            [["tag", "score"]]
        )
        target_limit = target_limit * 2
        output_next["score"] += output["score"].max()
        output = pd.concat([output, output_next])
        output = (
            output
            .groupby("tag")
            .first()
            .reset_index()
            [["tag", "score"]]
        )
        output["score"] /= output["score"].max()
        output["score"] *= 100
        output["score"] = np.nan_to_num(output["score"], nan=100.0)
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
