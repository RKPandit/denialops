.PHONY: install lint format typecheck test eval run clean docker-build docker-run help

# Default target
help:
	@echo "DenialOps Development Commands"
	@echo "=============================="
	@echo "make install     - Install dependencies (dev mode)"
	@echo "make lint        - Run linter checks"
	@echo "make format      - Format code"
	@echo "make typecheck   - Run type checker"
	@echo "make test        - Run tests with coverage"
	@echo "make eval        - Run evaluation harness"
	@echo "make run         - Start development server"
	@echo "make clean       - Remove build artifacts"
	@echo "make docker-build - Build Docker image"
	@echo "make docker-run  - Run with Docker Compose"

# Install dependencies
install:
	pip install -e ".[dev]"

# Linting
lint:
	ruff check src tests
	ruff format --check src tests

# Format code
format:
	ruff format src tests
	ruff check --fix src tests

# Type checking
typecheck:
	mypy src

# Run tests
test:
	PYTHONPATH=src pytest tests -v --cov=src/denialops --cov-report=term-missing

# Run tests without coverage (faster)
test-quick:
	PYTHONPATH=src pytest tests -v

# Run evaluation harness
eval:
	python -m eval.run_eval

# Start development server
run:
	PYTHONPATH=src uvicorn denialops.main:app --reload --host 0.0.0.0 --port 8000

# Clean build artifacts
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

# Docker
docker-build:
	docker build -t denialops:latest .

docker-run:
	docker compose up

# CI target (runs all checks)
ci: lint typecheck test
	@echo "All CI checks passed!"
