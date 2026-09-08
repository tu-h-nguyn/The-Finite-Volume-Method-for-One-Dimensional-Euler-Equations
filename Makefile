PYTHON ?= python3

.PHONY: help install test lint types format results animation notebook report clean

help:
	@echo "install  install the package with its development dependencies"
	@echo "test     run the test suite with coverage"
	@echo "lint     run ruff over the sources"
	@echo "types    run mypy over the package"
	@echo "format   apply ruff's automatic fixes"
	@echo "results  regenerate docs/RESULTS.md and every figure in docs/figures"
	@echo "animation regenerate docs/figures/sod_evolution.gif"
	@echo "notebook regenerate and execute notebooks/demo.ipynb"
	@echo "report   build report/report.pdf with latexmk (needs a TeX distribution)"
	@echo "clean    remove build artifacts"

install:
	$(PYTHON) -m pip install -e ".[dev]"

test:
	$(PYTHON) -m pytest --cov=euler1d --cov-report=term-missing

lint:
	$(PYTHON) -m ruff check .

types:
	$(PYTHON) -m mypy

format:
	$(PYTHON) -m ruff check --fix .

results:
	$(PYTHON) scripts/generate_results.py

animation:
	$(PYTHON) scripts/make_animation.py

notebook:
	$(PYTHON) scripts/build_demo_notebook.py

report:
	cd report && latexmk -pdf -interaction=nonstopmode master.tex

clean:
	rm -rf build dist .pytest_cache .ruff_cache htmlcov .coverage
	find . -name '__pycache__' -type d -prune -exec rm -rf {} +
	cd report && latexmk -C || true
