# InvestMentor

Personal investment tracker, journal, and AI mentor for BIST, US stocks, and crypto.

## Features (Roadmap)

### Phase 1 — Portfolio Tracker (current)
- Track holdings across BIST, NYSE/NASDAQ, and crypto
- Live prices via yfinance (stocks) and CoinGecko (crypto)
- P&L calculation with commission tracking, portfolio summary
- Portfolio distribution analysis (by market, currency)
- Investment journal with emotion tagging
- Price alerts ("THYAO ₺300'ü geçerse bildir")

### Phase 2 — AI Mentor
- LLM-powered investment conversations
- RAG over investment books and personal notes
- Emotional investing alerts ("Bu hafta 3 alım yaptın, 2'si FOMO")
- Weekly AI-generated portfolio reports

### Phase 3 — Simulation & Strategy Game
- Paper trading with real prices
- Backtesting engine (DCA, Value, Momentum)
- Historical scenario replays
- Strategy tournament mode

### Phase 4 — Social + Advanced AI
- Technical indicators (RSI, MACD, Bollinger Bands)
- Macro data correlation (USD/TRY, interest, inflation)
- Anonymous portfolio sharing
- Mobile PWA, push notifications
- PDF/Excel export for tax reporting

## Tech Stack

| Layer | Technology |
|---|---|
| Backend | FastAPI, SQLAlchemy, Pydantic |
| Data | yfinance, CoinGecko API, SQLite → PostgreSQL |
| AI (Phase 2) | Claude API, LangChain, Qdrant |
| Frontend | Streamlit (Phase 1) → React (Phase 3) |
| Deploy | Docker, Hetzner / Railway |

## Quick Start

```bash
cd invest-mentor

# Option 1: Using Makefile
make install
make run

# Option 2: Manual
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload

# Open docs
open http://localhost:8000/docs

# Run tests (from invest-mentor root)
make test
# or: pytest -v
```

## API Endpoints

### Portfolio

| Method | Endpoint | Description |
|---|---|---|
| POST | `/api/v1/portfolio/holdings` | Add a new position |
| GET | `/api/v1/portfolio/holdings` | List holdings (filter: `?market=BIST`) |
| GET | `/api/v1/portfolio/holdings/{id}` | Get holding with live price |
| PUT | `/api/v1/portfolio/holdings/{id}` | Update a position (partial) |
| DELETE | `/api/v1/portfolio/holdings/{id}` | Remove a position |
| GET | `/api/v1/portfolio/summary` | Full portfolio P&L summary |
| GET | `/api/v1/portfolio/distribution` | Allocation by market & currency |

### Market Data

| Method | Endpoint | Description |
|---|---|---|
| GET | `/api/v1/market/price/{market}/{ticker}` | Live price (BIST/US/CRYPTO) |
| GET | `/api/v1/market/history/{market}/{ticker}` | Historical prices (`?period=1y`) |

### Journal

| Method | Endpoint | Description |
|---|---|---|
| POST | `/api/v1/journal/` | Create journal entry |
| GET | `/api/v1/journal/` | List entries (filter: `?entry_type=lesson`) |
| GET | `/api/v1/journal/{id}` | Get single entry |
| DELETE | `/api/v1/journal/{id}` | Delete entry |

### Price Alerts

| Method | Endpoint | Description |
|---|---|---|
| POST | `/api/v1/alerts/` | Create alert (above/below threshold) |
| GET | `/api/v1/alerts/` | List alerts (`?active_only=true`) |
| GET | `/api/v1/alerts/check` | Check all active alerts against live prices |
| DELETE | `/api/v1/alerts/{id}` | Delete alert |

## Project Structure

```
invest-mentor/
├── backend/
│   ├── app/
│   │   ├── main.py           # FastAPI application
│   │   ├── config.py         # Settings and constants
│   │   ├── models/
│   │   │   ├── database.py   # SQLAlchemy engine, session, Base
│   │   │   └── portfolio.py  # Holding, JournalEntry, PriceAlert
│   │   ├── schemas/
│   │   │   └── portfolio.py  # Pydantic request/response models
│   │   ├── routers/
│   │   │   ├── portfolio.py  # CRUD + summary + distribution
│   │   │   ├── journal.py    # Investment journal
│   │   │   ├── market.py     # Live prices + history
│   │   │   └── alerts.py     # Price alerts
│   │   └── services/
│   │       └── price_service.py  # yfinance + CoinGecko
│   └── requirements.txt
├── tests/
│   ├── conftest.py           # Shared fixtures, test DB
│   ├── test_portfolio.py
│   └── test_alerts.py
├── Makefile
├── pytest.ini
└── README.md
```

## Author

Built by Yetkin Eser — Data Scientist transitioning to AI Engineering.
