.PHONY: install run test lint clean

install:
	cd backend && pip install -r requirements.txt

run:
	cd backend && uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

test:
	pytest -v

lint:
	cd backend && python -m py_compile app/main.py
	cd backend && python -m py_compile app/config.py
	cd backend && python -m py_compile app/models/database.py
	cd backend && python -m py_compile app/models/portfolio.py
	cd backend && python -m py_compile app/schemas/portfolio.py
	cd backend && python -m py_compile app/routers/portfolio.py
	cd backend && python -m py_compile app/routers/journal.py
	cd backend && python -m py_compile app/routers/market.py
	cd backend && python -m py_compile app/routers/alerts.py
	cd backend && python -m py_compile app/services/price_service.py
	@echo "All files compile OK"

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	rm -f backend/data/portfolio.db
