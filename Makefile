# Makefile for the telemetry-anomaly-detector.
#
# Thin, self-documenting wrappers around the project's day-to-day commands.
# Run `make help` (the default target) to list everything.

# Use the interpreter from the active virtualenv when present, else the system
# python3. Override on the command line, e.g. `make test PYTHON=python`.
PYTHON ?= python
PIP := $(PYTHON) -m pip

# Source trees that linting, formatting, and type checks apply to.
CODE_DIRS := src api tests streamlit_app.py

.DEFAULT_GOAL := help

.PHONY: help install install-dev lock lint format format-check test data train compare evaluate serve demo clean

help: ## Show this help message.
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| sort \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

install: ## Install the exact, reproducible locked environment.
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements-lock.txt

install-dev: ## Install loose top-level dependencies for development.
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

lock: ## Regenerate requirements-lock.txt from the current environment.
	$(PIP) freeze > requirements-lock.txt

lint: ## Lint all source with ruff.
	$(PYTHON) -m ruff check $(CODE_DIRS)

format: ## Auto-format all source with ruff.
	$(PYTHON) -m ruff format $(CODE_DIRS)

format-check: ## Verify formatting without modifying files (CI mode).
	$(PYTHON) -m ruff format --check $(CODE_DIRS)

test: ## Run the test suite.
	$(PYTHON) -m pytest

data: ## Generate a synthetic telemetry dataset at data/raw/sessions.jsonl.
	$(PYTHON) -m src.generate_data --n-normal 1200 --n-cheater 300 --out data/raw/sessions.jsonl

train: ## Generate data (in memory) and train the anomaly-detection model.
	$(PYTHON) -m src.train

compare: ## Benchmark the Isolation Forest vs the autoencoder (table + ROC figure).
	$(PYTHON) -m src.compare

evaluate: ## Render ROC / score / feature-distribution figures into reports/.
	$(PYTHON) -m src.evaluate --dataset data/raw/sessions.jsonl --model models/model.joblib

serve: ## Run the FastAPI inference server with autoreload.
	$(PYTHON) -m uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

demo: ## Launch the interactive Streamlit demo (needs: pip install -r requirements-demo.txt).
	$(PYTHON) -m streamlit run streamlit_app.py

clean: ## Remove caches and generated pipeline artifacts.
	rm -rf .pytest_cache .ruff_cache .mypy_cache .coverage htmlcov coverage.xml
	rm -rf build dist *.egg-info
	rm -rf data/raw data/processed reports
	rm -f models/*.joblib models/metrics.json
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
