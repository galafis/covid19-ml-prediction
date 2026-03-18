.PHONY: install data train evaluate serve test lint clean

install:
	pip install -r requirements.txt

data:
	python -m src.data_ingestion

train:
	python -m src.models.train_all

evaluate:
	python -m src.evaluation

serve:
	uvicorn src.prediction_api:app --host 0.0.0.0 --port 8000 --reload

test:
	pytest tests/ -v --cov=src --cov-report=term-missing

lint:
	black src/ tests/
	isort src/ tests/
	flake8 src/ --max-line-length=120
	mypy src/ --ignore-missing-imports

clean:
	rm -rf __pycache__ .pytest_cache .mypy_cache htmlcov
	find . -type d -name __pycache__ -exec rm -rf {} +

all: install data train evaluate
