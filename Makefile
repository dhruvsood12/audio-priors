.PHONY: install install-dev lint format test data data-demo train interpret recommend-eval eval prepare-app app docker-build docker-run clean

PYTHON ?= python3
PIP ?= $(PYTHON) -m pip

install:
	$(PIP) install -e .

install-dev:
	$(PIP) install -e ".[dev,notebooks,app]"
	pre-commit install

lint:
	ruff check .
	ruff format --check .

format:
	ruff check . --fix
	ruff format .

test:
	pytest -q --cov=src/audio_priors --cov-fail-under=70 --cov-report=term-missing

data:
	$(PYTHON) scripts/download_data.py

data-demo:
	$(PYTHON) scripts/make_demo_data.py

train:
	$(PYTHON) scripts/train.py

interpret:
	$(PYTHON) scripts/interpret.py

recommend-eval:
	$(PYTHON) scripts/recommend_eval.py

# Alias kept for habit; previously pointed at a non-existent scripts/evaluate.py.
eval: recommend-eval

prepare-app:
	$(PYTHON) scripts/prepare_app.py

app: prepare-app
	streamlit run app/streamlit_app.py

docker-build:
	docker build -t audio-priors .

docker-run:
	docker run --rm -it audio-priors --help

clean:
	rm -rf build/ dist/ *.egg-info/ .pytest_cache/ .mypy_cache/ .ruff_cache/ htmlcov/ .coverage
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name .ipynb_checkpoints -exec rm -rf {} +
