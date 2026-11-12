.PHONY: install lint typecheck security test smoke build up down clean

install:
	pip install -r requirements.txt
	pip install -r requirements-dev.txt

lint:
	ruff check src/ tests/
	ruff format --check src/ tests/

typecheck:
	mypy src/ --ignore-missing-imports --strict

security:
	bandit -r src/ -f json -o bandit.json

test:
	pytest -q --cov=src --cov-report=xml --cov-fail-under=80

smoke:
	python tests/integration/smoke_test.py

build:
	docker-compose -f docker/docker-compose.yml build

up:
	docker-compose -f docker/docker-compose.yml up --build -d agent ui redis worker

down:
	docker-compose -f docker/docker-compose.yml down

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov coverage.xml bandit.json


.PHONY: demo demo-stop demo-prewarm demo-gpu demo-test

demo:
	@echo "=== Launching PDFMate AI Demo ==="
	python scripts/demo_launcher.py --demo

demo-stop:
	@echo "Stopping all services..."
	python scripts/demo_launcher.py --stop

demo-prewarm:
	@echo "Pre-warming model into VRAM..."
	python scripts/demo_launcher.py --prewarm-only

demo-gpu:
	@echo "Starting GPU dashboard..."
	python scripts/gpu_dashboard.py

demo-test:
	@echo "Running end-to-end pipeline test..."
	pytest tests/integration/test_full_pipeline.py -v
