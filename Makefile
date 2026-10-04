.PHONY: install setup check test lint serve eval

install:        ## install everything needed to run and test the app
	pip install -e ".[serve,dev]"

setup:          ## download the data and build the database, model and indexes (idempotent)
	python -m src.bootstrap

check:          ## report which build steps are done
	python -m src.bootstrap --check

test:           ## the full test suite (hermetic: needs no data or API key)
	pytest -q

lint:
	ruff check src tests eval

serve:          ## http://127.0.0.1:8000 (needs ANTHROPIC_API_KEY for chat)
	python -m src.api

eval:           ## live honesty / tool-use eval (spends API credit)
	python eval/run_eval.py
