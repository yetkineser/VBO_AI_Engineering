import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{BASE_DIR / 'data' / 'portfolio.db'}")

SUPPORTED_MARKETS = ["BIST", "US", "CRYPTO"]

# yfinance handles US stocks natively (e.g. "AAPL")
# BIST tickers use ".IS" suffix in yfinance (e.g. "THYAO.IS")
# Crypto uses CoinGecko API (free, no key needed)

COINGECKO_BASE_URL = "https://api.coingecko.com/api/v3"

LABEL_MAP = {
    "BIST": "Borsa Istanbul",
    "US": "US Stock (NYSE/NASDAQ)",
    "CRYPTO": "Cryptocurrency",
}

CURRENCY_SYMBOLS = {
    "TRY": "₺",
    "USD": "$",
}
