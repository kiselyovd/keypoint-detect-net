.PHONY: setup data train evaluate serve test lint fmt typecheck docker docs publish-hf clean

setup:
	uv sync --all-groups
	uv run pre-commit install

data:
	uv run python -m vehicle_keypoints.data.prepare --raw data/raw --out data/processed

train:
	uv run python -m vehicle_keypoints.training.train

evaluate:
	uv run python -m vehicle_keypoints.evaluation.evaluate --checkpoint artifacts/checkpoints/best.ckpt --data data/processed

serve:
	MODEL_PATH=artifacts/checkpoints/best.ckpt uv run uvicorn vehicle_keypoints.serving.main:app --reload --port 8000
test:
	uv run pytest

lint:
	uv run ruff check .

fmt:
	uv run ruff format .

typecheck:
	uv run mypy src

docker:
	docker compose build

docs:
	uv run mkdocs serve

publish-hf:
	uv run python scripts/publish_to_hf.py

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage coverage.xml dist build
