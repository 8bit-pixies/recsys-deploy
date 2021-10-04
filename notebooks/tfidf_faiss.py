"""
Usage:

https://github.com/facebookresearch/faiss
https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html

Running servers:
https://github.com/facebookresearch/faiss/blob/main/demos/demo_client_server_ivf.py
"""
from gensim import models
from gensim import corpora
import gensim
import pandas as pd
import numpy as np
import faiss
from colorama import Fore, Style


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


batch_size = 2 ** 12
max_dim = 2 ** 9
nlist = 4096
faiss_factory_builder = "IVF{nlist},Flat"  # "IVF4096,Flat", "IVF100,PQ8"

print("Training...")

skip = ["en"]
print(langs)

for lang in langs:
    # build TFIDF
    if lang in skip:
        continue
    print(f"\nTraining on language: {Style.BRIGHT + Fore.BLUE}{lang}{Style.RESET_ALL}.")
    subset_df = df[df["lan"] == lang]
    subset_df_len = subset_df.shape[0]
    texts = [document.split(",") for document in subset_df.tags]  # no processing as these are the raw tags
    dictionary = corpora.Dictionary(texts)

    calculated_max_dim = min(min(max_dim, max(dictionary.dfs.keys())), len(texts))
    calculated_nlist = min(calculated_max_dim, nlist)
    bow_corpus = [dictionary.doc2bow(text) for text in texts]
    print(f"..calculated dictionary size: {Style.BRIGHT + Fore.BLUE}{max(dictionary.cfs.keys())}{Style.RESET_ALL}.")
    tfidf = models.TfidfModel(bow_corpus)
    print(f"..finished tfidf...now training lsi with {Style.BRIGHT + Fore.BLUE}{calculated_max_dim}{Style.RESET_ALL}")
    lsi = models.LsiModel(
        tfidf[bow_corpus],
        num_topics=calculated_max_dim,
        chunksize=batch_size,
        power_iters=1,
        dtype=np.float32,
        extra_samples=0,
    )
    print("..finished lsi...now saving...")
    # need to dump and save tfidf, lsi, dictionary
    output = lsi[tfidf[bow_corpus]]
    # usage: gensim.matutils.corpus2csc(output[:5])
    # just don't try dumping everything or we'll have issues
    tfidf.save(f"notebooks/model/tfidf_{lang}")
    lsi.save(f"notebooks/model/lsi_{lang}")
    dictionary.save(f"notebooks/model/dictionary_{lang}")

    print(f"..using index factory: {faiss_factory_builder.format(nlist=calculated_nlist)}.")
    index = faiss.index_factory(calculated_max_dim, faiss_factory_builder.format(nlist=calculated_nlist))
    # split into batches..
    for idx_set in range(subset_df_len // batch_size):
        xt = gensim.matutils.corpus2dense(
            output[idx_set * batch_size : (idx_set + 1) * (batch_size)], num_terms=calculated_max_dim
        ).T.astype(np.float32)
        if xt.shape[0] >= calculated_max_dim:
            # print(xt, xt.shape)
            index.train(xt)
    faiss.write_index(index, f"notebooks/model/faiss_{lang}")
