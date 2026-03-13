setup:
	pip install -r requirements.dev.txt
	pip install -e .
	
train:
	python -m credit_ml.modeling.train

test:
	pytest -q

serve:
	python -m uvicorn credit_ml.api.main:app --reload --port 8010

lint:
	ruff check .

format:
	ruff format src tests