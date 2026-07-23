.PHONY: install run eval test fmt

install:
	pip install -r requirements.txt

run:
	uvicorn app.main:app --reload

eval:
	python -m eval.run_eval

compare:
	python -m eval.compare_retrieval

test:
	pytest -q
