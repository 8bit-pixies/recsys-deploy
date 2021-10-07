import random
import time

import httpx
import numpy as np
import pandas as pd
import tqdm

df = pd.read_csv("data/tags_on_posts_sample.csv")[["tags", "lang"]]


def benchmark_function(df=df, repeats=2000):

    output = []

    for _ in tqdm.tqdm(range(repeats)):
        indx = random.choice(range(df.shape[0]))
        indx2 = random.choice(range(df.shape[0]))
        query1 = df.iloc[indx].tags.split(",")
        query2 = df.iloc[indx2].tags.split(",")
        query_all = query1 + query2
        # random sample
        if len(query_all) > 6:
            k = random.choice(range(5, len(query_all)))
            indx = random.sample(range(len(query_all)), k)
            query_all = np.array(query_all)[indx].tolist()

        limit = random.choice(range(50, 100))

        t0 = round(time.time() * 1000.0)
        _ = httpx.post("http://127.0.0.1:8000/", json={"query": query_all, "limit": limit})
        time_taken = round(time.time() * 1000.0) - t0
        output.append(time_taken)

    return output


sample_benchmark = benchmark_function()
# return statistics:

df = pd.DataFrame({"percentile": [50, 75, 90, 95, 99], "score": np.percentile(sample_benchmark, [50, 75, 90, 95, 99])})
print(df)
