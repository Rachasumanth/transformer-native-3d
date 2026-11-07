PYTHON ?= python3.12
VENV ?= .venv
PIP := $(VENV)/bin/pip
PY := $(VENV)/bin/python

.PHONY: install test doctor serve docker-build docker-run clean

install:
	$(PYTHON) -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install --index-url https://download.pytorch.org/whl/cpu torch
	$(PIP) install -e ".[service]"

test:
	PYTHONPATH=. $(PY) -m pytest -q

doctor:
	PYTHONPATH=. $(PY) -m t3d.cli doctor

serve:
	PYTHONPATH=. $(PY) -m t3d.cli serve --host 0.0.0.0 --port 8000

docker-build:
	docker build -t t3d:latest .

docker-run:
	docker run --rm -p 8000:8000 \
		-v "$$(pwd)/artifacts:/models:ro" \
		-v "$${HOME}/.cache/huggingface:/cache/huggingface" \
		t3d:latest

clean:
	rm -rf .pytest_cache __pycache__ t3d/__pycache__ tools/__pycache__ tests/__pycache__
