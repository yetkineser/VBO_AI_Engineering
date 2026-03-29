from collections import defaultdict

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import Optional

from app.models.database import get_db
from app.models.portfolio import Holding
from app.schemas.portfolio import (
    HoldingCreate,
    HoldingUpdate,
    HoldingResponse,
    HoldingWithPrice,
    PortfolioSummary,
    PortfolioDistribution,
    AllocationItem,
)
from app.services.price_service import get_price

router = APIRouter(prefix="/portfolio", tags=["portfolio"])


@router.post("/holdings", response_model=HoldingResponse, status_code=201)
def add_holding(holding: HoldingCreate, db: Session = Depends(get_db)):
    """Add a new position to the portfolio."""
    db_holding = Holding(**holding.model_dump())
    db.add(db_holding)
    db.commit()
    db.refresh(db_holding)
    return db_holding


@router.get("/holdings", response_model=list[HoldingResponse])
def list_holdings(
    market: Optional[str] = None,
    db: Session = Depends(get_db),
):
    """List all holdings, optionally filtered by market."""
    query = db.query(Holding)
    if market:
        query = query.filter(Holding.market == market.upper())
    return query.order_by(Holding.buy_date.desc()).all()


@router.get("/holdings/{holding_id}", response_model=HoldingWithPrice)
def get_holding(holding_id: int, db: Session = Depends(get_db)):
    """Get a single holding with live price data."""
    holding = db.query(Holding).filter(Holding.id == holding_id).first()
    if not holding:
        raise HTTPException(404, "Holding not found")

    current_price = get_price(holding.ticker, holding.market)
    return _enrich_holding(holding, current_price)


@router.put("/holdings/{holding_id}", response_model=HoldingResponse)
def update_holding(holding_id: int, updates: HoldingUpdate, db: Session = Depends(get_db)):
    """Update an existing holding (partial update)."""
    holding = db.query(Holding).filter(Holding.id == holding_id).first()
    if not holding:
        raise HTTPException(404, "Holding not found")

    update_data = updates.model_dump(exclude_unset=True)
    for field, value in update_data.items():
        setattr(holding, field, value)

    db.commit()
    db.refresh(holding)
    return holding


@router.delete("/holdings/{holding_id}", status_code=204)
def delete_holding(holding_id: int, db: Session = Depends(get_db)):
    """Remove a position from the portfolio."""
    holding = db.query(Holding).filter(Holding.id == holding_id).first()
    if not holding:
        raise HTTPException(404, "Holding not found")
    db.delete(holding)
    db.commit()


@router.get("/summary", response_model=PortfolioSummary)
def portfolio_summary(db: Session = Depends(get_db)):
    """Full portfolio overview with live prices and P&L."""
    holdings = db.query(Holding).all()

    enriched = []
    total_invested = 0.0
    total_commission = 0.0
    current_value = 0.0

    for h in holdings:
        price = get_price(h.ticker, h.market)
        enriched_h = _enrich_holding(h, price)
        enriched.append(enriched_h)

        cost = h.buy_price * h.quantity
        total_invested += cost
        total_commission += h.commission or 0.0
        if price:
            current_value += price * h.quantity
        else:
            current_value += cost

    pnl = current_value - total_invested - total_commission
    pnl_pct = (pnl / total_invested * 100) if total_invested > 0 else 0.0

    return PortfolioSummary(
        total_invested=round(total_invested, 2),
        total_commission=round(total_commission, 2),
        current_value=round(current_value, 2),
        total_profit_loss=round(pnl, 2),
        total_profit_loss_pct=round(pnl_pct, 2),
        holdings_count=len(holdings),
        holdings=enriched,
    )


@router.get("/distribution", response_model=PortfolioDistribution)
def portfolio_distribution(db: Session = Depends(get_db)):
    """Portfolio allocation breakdown by market and currency."""
    holdings = db.query(Holding).all()

    if not holdings:
        return PortfolioDistribution(
            by_market=[], by_currency=[], total_invested=0.0
        )

    by_market: dict[str, float] = defaultdict(float)
    by_currency: dict[str, float] = defaultdict(float)
    total = 0.0

    for h in holdings:
        cost = h.buy_price * h.quantity
        total += cost
        market_label = h.market.value if hasattr(h.market, "value") else h.market
        by_market[market_label] += cost
        by_currency[h.buy_currency] += cost

    def to_allocation(data: dict[str, float]) -> list[AllocationItem]:
        return [
            AllocationItem(
                label=label,
                value=round(value, 2),
                percentage=round(value / total * 100, 2) if total > 0 else 0.0,
            )
            for label, value in sorted(data.items(), key=lambda x: -x[1])
        ]

    return PortfolioDistribution(
        by_market=to_allocation(by_market),
        by_currency=to_allocation(by_currency),
        total_invested=round(total, 2),
    )


def _enrich_holding(holding: Holding, current_price: Optional[float]) -> HoldingWithPrice:
    """Attach live price and P&L to a holding."""
    cost = holding.buy_price * holding.quantity
    commission = holding.commission or 0.0
    total_cost = cost + commission
    cur_val = (current_price * holding.quantity) if current_price else None
    pnl = (cur_val - total_cost) if cur_val else None
    pnl_pct = (pnl / total_cost * 100) if pnl and total_cost > 0 else None

    return HoldingWithPrice(
        id=holding.id,
        ticker=holding.ticker,
        market=holding.market,
        quantity=holding.quantity,
        buy_price=holding.buy_price,
        buy_currency=holding.buy_currency,
        commission=commission,
        buy_date=holding.buy_date,
        notes=holding.notes,
        emotion_tag=holding.emotion_tag,
        created_at=holding.created_at,
        current_price=current_price,
        current_value=round(cur_val, 2) if cur_val else None,
        total_cost=round(total_cost, 2),
        profit_loss=round(pnl, 2) if pnl else None,
        profit_loss_pct=round(pnl_pct, 2) if pnl_pct else None,
    )
