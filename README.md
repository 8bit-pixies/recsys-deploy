# recsys-deploy

Recommendation System Deployment

# Training

The easiest way to try and run the whole thing end to end is to

*  move `tags_on_posts_sample.csv` to `data/tags_on_posts_sample.csv`
*  automatically install package + dependencies, and train the model via `make train_quick_run`

```
wget -O data/tags_on_posts_sample.csv <https://path/to/tags_on_posts_sample.csv>
make train_quick_run
```

This won't train `LSI` to completion, but shouldn't take longer than a few minutes to train. It will then build and run the container. Longer form:

```sh
wget -O data/tags_on_posts_sample.csv <https://path/to/tags_on_posts_sample.csv>
pip install -e . 
mkdir -p notebooks/model_quick
wget -O notebooks/model_quick/lid.176.ftz https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz
python notebooks/model_quick.py
cp -r notebooks/model_quick recsys/model_quick
docker build -t recsys -f docker/Dockerfile.api .
docker run --rm -it -p 8000:8000 recsys
```

In a separate shell

```sh
pip install -e ".[dev]"
python benchmark/benchmark.py
```

# Usage

You can try the cli as shown below

````sh
$ python -m recsys.cli '{"query":["dog", "dog park"], "limit": 5}'
[{'tag': 'dogsarefamily', 'score': 31.691408157348633}, {'tag': 'cute dogs', 'score': 31.691408157348633}, {'tag': 'huskylove', 'score': 31.691408157348633}, {'tag': 'petlovers', 'score': 31.691408157348633}, {'tag': 'mtblife', 'score': 35.07106399536133}]```
````

```sh
$ python -m recsys.cli '{"query":["广州"], "limit": 5}'
[{'tag': 'c25', 'score': 98.29067993164062}, {'tag': '混凝土直销', 'score': 98.29067993164062}, {'tag': 'sherlolly', 'score': 99.99999237060547}, {'tag': '三遊亭わん丈', 'score': 99.99999237060547}, {'tag': '春風亭一蔵', 'score': 99.99999237060547}]
```

# Benchmark results

You can build and start the API server as follows (`podman` was used on Ubuntu 21.04)

```sh
make podman_build
make podman_run
```

The benchmark results can be run through `python benchmark/benchmark.py`, which presumes port `8000` is used (as hard-coded in the `Makefile`)

The below were based on running off the container as above

```
$ python benchmark/benchmark.py 
100%|███████████████████████████████████████████████████████████████████████████| 2000/2000 [03:14<00:00, 10.29it/s]
   percentile   score
0          50   92.00
1          75  105.00
2          90  122.00
3          95  133.00
4          99  158.01
```
