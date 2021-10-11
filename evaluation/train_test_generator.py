"""
This script generates the training and test splits
"""

from typing import List
import pandas as pd

from sklearn.model_selection import train_test_split
import difflib

def get_data():
    df = pd.read_csv("data/tags_on_posts_sample.csv")[["tags", "lang"]]
    X_train, X_test = train_test_split(df, random_state=42)

    # generate the eval from X_test
    X_test["tags"] = X_test["tags"].apply(lambda x: x.split(","))
    X_test["tags_drop"] = X_test["tags"].copy()
    X_test = X_test.explode("tags_drop")
    X_test['tags_eval'] = X_test.apply(lambda x: [y for y in x['tags'] if y != x['tags_drop']], axis=1)
    X_test = X_test[X_test['tags_eval'].apply(lambda x: len(x)) > 0]
    return X_train, X_test

def get_all():
    df = pd.read_csv("data/tags_on_posts_sample.csv")[["tags", "lang"]]
    df['tags_eval'] = df['tags'].apply(lambda x: x.split(","))
    return df

def evaluate_single(y_true: str, y_pred: List[str]) -> float:
    output = []
    # print(y_true, y_pred)
    for idx, tag in enumerate(y_pred):
        weight = 1/(idx+1)
        try:
            if len(tag) <= len(y_true):
                s = difflib.SequenceMatcher(None, y_true, tag)
                score = sum(n for i,j,n in s.get_matching_blocks()) / float(len(tag))
            else:
                s = difflib.SequenceMatcher(None, tag, y_true)
                score = sum(n for i,j,n in s.get_matching_blocks()) / float(len(y_true))
        except:
            score = 0
        score *= weight
        output.append(score)
    return max(output)

def evaluate_fast(y_true:str, y_pred: List[str]) -> float:
    tag = ' '.join(y_pred)
    s = difflib.SequenceMatcher(None, y_true, tag)
    score = sum(n for i,j,n in s.get_matching_blocks()) / float(len(tag))
    return score