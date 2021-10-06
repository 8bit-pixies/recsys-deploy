# recsys-deploy

Recommendation System Deployment

# Installation and testing

You can test and run (assuming that the metadata is included...)

```sh
pip install -e .
```

```sh
python -m recsys.cli '{"query":["hello", "world", "hello world"], "limit": 5}'
```

# Testing via FastAPI

You can build and start the API server as follows (`podman` was used on Ubuntu 21.04)

```sh
make podman_build
make podman_run
```

The benchmark results can be run through `python benchmark/benchmark.py`, which presumes port `8000` is used (as hard-coded in the `Makefile`)
