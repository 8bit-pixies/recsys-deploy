# I'm using podman - change this where appropriate!
docker = podman

train_quick: install
	# move training data to data/
	mkdir -p notebooks/model_quick
	wget -O notebooks/model_quick/lid.176.ftz https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz
	python notebooks/model_quick.py
	cp -r notebooks/model_quick recsys

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
	$(docker) build -t recsys -f docker/Dockerfile.api .

podman_run:
	$(docker) run --rm -it -p 8000:8000 recsys

podman_run_build: podman_build podman_run

train_quick_run: train_quick podman_build podman_run