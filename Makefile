PYTHON ?= python3

.PHONY: help install test lint format results report clean

help:
	@echo "install  install the package with its development dependencies"
	@echo "test     run the test suite with coverage"
	@echo "lint     run ruff over the sources"
	@echo "format   apply ruff's automatic fixes"
	@echo "results  regenerate docs/RESULTS.md and every figure in docs/figures"
	@echo "report   build report/report.pdf with latexmk (needs a TeX distribution)"
	@echo "clean    remove build artifacts"

install:
	$(PYTHON) -m pip install -e ".[dev]"

test:
	$(PYTHON) -m pytest --cov=euler1d --cov-report=term-missing

lint:
	$(PYTHON) -m ruff check .

format:
	$(PYTHON) -m ruff check --fix .

results:
	$(PYTHON) scripts/generate_results.py

report:
	cd report && latexmk -pdf -interaction=nonstopmode master.tex

clean:
	rm -rf build dist .pytest_cache .ruff_cache htmlcov .coverage
	find . -name '__pycache__' -type d -prune -exec rm -rf {} +
	cd report && latexmk -C || true
