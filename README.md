# recsys-deploy

Recommendation System Deployment

# Training

Here is sample output from training

```
$ python notebooks/tfidf_faiss_dual.py
Warning : `load_model` does not return WordVectorModel or SupervisedModel any more, but a `FastText` object which is very similar.
... next
..calculated dictionary size (w2v): 27917.
..calculated dictionary size: 27917.
..finished tfidf...now training lsi with 800
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████| 190/190 [42:20<00:00, 13.37s/it]
..using index FlatL2: 800.
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████| 190/190 [00:32<00:00,  5.86it/s]
... next
..calculated dictionary size (w2v): 1236.
..calculated dictionary size: 1236.
..finished tfidf...now training lsi with 800
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████| 5/5 [00:04<00:00,  1.24it/s]
..using index FlatL2: 800.
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████| 5/5 [00:00<00:00, 13.74it/s]
... next
..calculated dictionary size: 27917.
... next
..calculated dictionary size: 27917.
[{'tag': 'doggy', 'score': 0.41756876620422556}, {'tag': 'pitbull rules', 'score': 1.0999268293380737}, {'tag': 'pitbull love', 'score': 1.0999268293380737}, {'tag': 'pitbull life', 'score': 1.0999268293380737}, {'tag': 'pitbull dad', 'score': 1.0999268293380737}]
```

# Installation and testing

You can test and run (assuming that the model data created from the training step is moved to `smetadata_dual`). The folder structure under `recsys` is:

```sh
recsys
├── api.py
├── cli.py
├── __init__.py
├── metadata_dual
│   ├── data.pkl
│   ├── dictionary_en
│   ├── dictionary_other
│   ├── faiss_en.index
│   ├── faiss_other.index
│   ├── lid.176.bin
│   ├── lsi_en
│   ├── lsi_en.projection
│   ├── lsi_en.projection.u.npy
│   ├── lsi_other
│   ├── lsi_other.projection
│   ├── tfidf_en
│   ├── tfidf_other
│   ├── w2v_en
│   └── w2v_other
├── recsys.py
└── utils.py
```

```sh
pip install -e .
```

```sh
python -m recsys.cli '{"query":["hello", "world", "hello world"], "limit": 5}'

python -m recsys.cli '{"query":["肇庆混凝土","美灼物资"], "limit": 5}'
```

# Testing via FastAPI

You can build and start the API server as follows (`podman` was used on Ubuntu 21.04)

```sh
make podman_build
make podman_run
```

The benchmark results can be run through `python benchmark/benchmark.py`, which presumes port `8000` is used (as hard-coded in the `Makefile`)

# Benchmark Results

The below were based on running off the container - not "locally"

```
$ python benchmark/benchmark.py
100%|█████████████████████████████████████████████████████████| 2000/2000 [04:22<00:00,  7.62it/s]
   percentile  score
0          50  111.0
1          75  119.0
2          90  204.1
3          95  246.0
4          99  351.1
```

# Sample Outputs

This demonstrates the multi-language support (albeit, not necessarily sensible)

```
$ python -m recsys.cli '{"query":["广州"], "limit": 5}'
[{'tag': 'THE MOOD', 'score': 9.871350288391113}, {'tag': 'みずの', 'score': 9.871350288391113}, {'tag': '三遊亭わん丈', 'score': 9.871350288391113}, {'tag': '塩大福', 'score': 9.871350288391113}, {'tag': '春風亭一蔵', 'score': 9.871350288391113}]
```

For english side - it is somewhat better. There are still limitations as the tags were built on exact match at this stage, and don't tokenize tags.

```
$ python -m recsys.cli '{"query":["dog", "dog park"], "limit": 5}'
[{'tag': ':00 doggie!!!', 'score': 31.603696823120117}, {'tag': 'going 2 CRYYY', 'score': 33.004329681396484}, {'tag': 'mutastanaccount', 'score': 37.02174377441406}, {'tag': 'is it different depending on the size of the cat like it is for dog', 'score': 37.02174377441406}, {'tag': 'jim jarmusch', 'score': 38.43353271484375}]
```
