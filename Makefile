style:
	isort src 
	ruff --fix src 
	black src 

check:
	black --check src 
	ruff src 
	pylint -j 8 --fail-under=8 src

test:
	rm -r build
	rm -r src/llm_prompt.egg-info
	pip install -e .
	pytest tests