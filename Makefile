.PHONY: install test lint demo benchmark clean

install:
	pip install -e .

test:
	pytest tests/

lint:
	@echo "lint placeholder — no linter configured yet"

demo:
	python scripts/demo.py

benchmark:
	python scripts/benchmark.py

clean:
	python -c "import shutil, pathlib; [shutil.rmtree(p) for p in pathlib.Path('.').rglob('__pycache__')]"
	python -c "import shutil, pathlib; [shutil.rmtree(p) for p in pathlib.Path('.').rglob('*.egg-info')]"
