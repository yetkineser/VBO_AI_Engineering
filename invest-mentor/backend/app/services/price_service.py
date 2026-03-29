"""
Unified price service for BIST, US stocks, and crypto.

- US stocks & BIST: yfinance (BIST tickers use ".IS" suffix)
- Crypto: CoinGecko free API (no key needed)
"""

import httpx
import yfinance as yf
from typing import Optional

from app.config import COINGECKO_BASE_URL


def _normalize_ticker(ticker: str, market: str) -> str:
    """Convert our internal ticker to the format the data source expects."""
    ticker = ticker.upper()
    if market == "BIST" and not ticker.endswith(".IS"):
        return f"{ticker}.IS"
    return ticker


def get_stock_price(ticker: str, market: str) -> Optional[float]:
    """Get current price for a BIST or US stock via yfinance."""
    yf_ticker = _normalize_ticker(ticker, market)
    try:
        info = yf.Ticker(yf_ticker).fast_info
        return float(info.get("lastPrice", 0) or info.get("previousClose", 0))
    except Exception:
        return None


def get_crypto_price(coin_id: str, vs_currency: str = "usd") -> Optional[float]:
    """Get current price for a cryptocurrency via CoinGecko."""
    url = f"{COINGECKO_BASE_URL}/simple/price"
    params = {"ids": coin_id.lower(), "vs_currencies": vs_currency}
    try:
        resp = httpx.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        return data.get(coin_id.lower(), {}).get(vs_currency)
    except Exception:
        return None


def get_price(ticker: str, market: str) -> Optional[float]:
    """Unified entry point — routes to the right data source."""
    if market == "CRYPTO":
        return get_crypto_price(ticker)
    return get_stock_price(ticker, market)


def get_stock_history(ticker: str, market: str, period: str = "1y"):
    """Get historical price data as a pandas DataFrame (stocks only)."""
    yf_ticker = _normalize_ticker(ticker, market)
    try:
        return yf.Ticker(yf_ticker).history(period=period)
    except Exception:
        return None


def get_crypto_history(
    coin_id: str, vs_currency: str = "usd", days: int = 365
) -> Optional[list[dict]]:
    """Get historical price data for a cryptocurrency via CoinGecko.

    Returns a list of {"date": "YYYY-MM-DD", "close": float} dicts.
    CoinGecko free tier: max 365 days, daily granularity for > 90 days.
    """
    url = f"{COINGECKO_BASE_URL}/coins/{coin_id.lower()}/market_chart"
    params = {"vs_currency": vs_currency, "days": days}
    try:
        resp = httpx.get(url, params=params, timeout=15)
        resp.raise_for_status()
        prices = resp.json().get("prices", [])
        from datetime import datetime
        records = []
        seen_dates: set[str] = set()
        for timestamp_ms, price in prices:
            date_str = datetime.utcfromtimestamp(timestamp_ms / 1000).strftime("%Y-%m-%d")
            if date_str not in seen_dates:
                seen_dates.add(date_str)
                records.append({"date": date_str, "close": round(price, 2)})
        return records
    except Exception:
        return None


PERIOD_TO_DAYS = {
    "7d": 7,
    "1mo": 30,
    "3mo": 90,
    "6mo": 180,
    "1y": 365,
    "max": 365,
}
