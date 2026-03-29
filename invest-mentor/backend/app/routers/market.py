from fastapi import APIRouter, HTTPException

from app.services.price_service import (
    get_price,
    get_stock_history,
    get_crypto_history,
    PERIOD_TO_DAYS,
)

router = APIRouter(prefix="/market", tags=["market"])


@router.get("/price/{market}/{ticker}")
def live_price(market: str, ticker: str):
    """Get the current price for any asset."""
    market = market.upper()
    price = get_price(ticker, market)
    if price is None:
        raise HTTPException(404, f"Could not fetch price for {ticker} ({market})")
    return {"ticker": ticker, "market": market, "price": price}


@router.get("/history/{market}/{ticker}")
def price_history(market: str, ticker: str, period: str = "1y"):
    """Get historical prices (returns dates and closing prices).

    Supported periods: 7d, 1mo, 3mo, 6mo, 1y, max
    """
    market = market.upper()

    if market == "CRYPTO":
        days = PERIOD_TO_DAYS.get(period, 365)
        records = get_crypto_history(ticker, days=days)
        if not records:
            raise HTTPException(404, f"No history for {ticker} ({market})")
        return {"ticker": ticker, "market": market, "period": period, "data": records}

    df = get_stock_history(ticker, market, period=period)
    if df is None or df.empty:
        raise HTTPException(404, f"No history for {ticker} ({market})")

    records = [
        {"date": idx.strftime("%Y-%m-%d"), "close": round(row["Close"], 2)}
        for idx, row in df.iterrows()
    ]
    return {"ticker": ticker, "market": market, "period": period, "data": records}
