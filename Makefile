.PHONY: install test run demo report clean

install:
	pip install -r requirements.txt

test:
	pytest -v

run:
	python -m volfc.run --config configs/sp500.yaml

run-nsei:
	python -m volfc.run --config configs/nsei.yaml

demo:
	python leakage_demo.py --source yfinance --ticker ^GSPC --horizons 1 5 22

report: run run-nsei demo

clean:
	rm -rf __pycache__ .pytest_cache *.pyc volfc/__pycache__ tests/__pycache__
