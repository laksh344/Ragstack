.PHONY: dev test lint docker-up docker-down ingest eval clean

# Start FastAPI dev server
dev:
	uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000

# Run tests
test:
	pytest tests/ -v --cov=backend --cov-report=term-missing

# Lint & format
lint:
	ruff check backend/ tests/ --fix
	ruff format backend/ tests/

# Type check
typecheck:
	mypy backend/

# Start infrastructure services
docker-up:
	docker compose up -d
	@echo "Waiting for services to be healthy..."
	@sleep 5
	@echo "Qdrant:        http://localhost:6333/dashboard"
	@echo "Elasticsearch: http://localhost:9200"
	@echo "Redis:         localhost:6379"

# Stop infrastructure
docker-down:
	docker compose down

# Seed sample documents
seed:
	python scripts/seed_data.py

# Run evaluation suite
eval:
	python eval/run_eval.py

# Clean uploads and caches
clean:
	rm -rf uploads/*
	rm -rf __pycache__ .pytest_cache .mypy_cache
