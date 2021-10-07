# I'm using podman - change this where appropriate!
docker = podman

train:
	# move training data to data/
	mkdir notebooks/model_dual
	wget -O notebooks/model_dual/lid.176.bin https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin
	python notebooks/tfidf_faiss_dual.pytest
	cp -r notebooks/model_dual/* recsys/model_dual/*

install:
	pip install -e . 

format: format-docs format-python

format-docs:
	npx prettier --write . 

format-python:
	isort .;
	black .;

test-python:
	pytest tests

lint-python:
	flake8 .;
	isort . --check-only;
	black . --check;

podman_build:
	# buildah containers
	$(docker) build -t recsys -f docker/Dockerfile.api .

podman_run:
	# podman run recsys '<json blob>'
	# podman run recsys
	# $(docker) run recsys
	$(docker) run --rm -it -p 8000:8000 recsys
