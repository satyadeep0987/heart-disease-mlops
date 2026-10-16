PYTHON ?= python3

.PHONY: install check lint test
install:
	$(PYTHON) -m pip install -r requirements-dev.txt

check:
	$(PYTHON) -m scripts.check_setup

lint:
	ruff check .

test:
	pytest -q
