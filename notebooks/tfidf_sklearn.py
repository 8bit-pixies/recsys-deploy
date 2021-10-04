"""
Usage:

https://github.com/facebookresearch/faiss
https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html

Running servers:
https://github.com/facebookresearch/faiss/blob/main/demos/demo_client_server_ivf.py
"""
from sklearn.feature_extraction.text import TfidfVectorizer

# technically speaking should be using SVD for LSI - but use something that can scale in scikit learn
# we can redo this "properly" in gensim later.
from sklearn.decomposition import IncrementalPCA
from sklearn.pipeline import make_pipeline
import pandas as pd
import faiss
from joblib import dump, load

# there are different languages - the naive approach is to train a different
# model for each language; we'll deal with that detail later -
df = pd.read_csv("data/tags_on_posts_sample.csv")
# ignore countries....for now
df["lan"] = df["lang"].apply(lambda x: x[:2])


# KISS - and only have languages, and assume countries don't matter...
"""
Counter({'en': 92029,
         'ja': 393,
         'pt': 1003,
         'de': 1225,
         'fr': 894,
         'es': 2054,
         'it': 583,
         'pl': 323,
         'ru': 352,
         'tr': 369,
         'zh': 107,
         'nl': 442,
         'ko': 110,
         'id': 115})
"""
langs = df["lan"].unique()

for lang in langs:
    # build TFIDF
    try:
        tfidf_mod = make_pipeline(TfidfVectorizer(), IncrementalPCA(10000, batch_size=10000))
        xt = tfidf_mod.fit_transform(df[df["lan"] == lang]["tags"])
    except:
        tfidf_mod = TfidfVectorizer()
        xt = tfidf_mod.fit_transform(df[df["lan"] == lang]["tags"])
    dump(tfidf_mod, f"notebooks/model/tfidf_{lang}.joblib")

    index = faiss.index_factory(xt.shape[1], "IVF4096,Flat")
    index.train(xt)
    faiss.write_index(index, f"notebooks/model/tfidf_faiss_{lang}.joblib")
