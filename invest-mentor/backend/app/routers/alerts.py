from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import Optional

from app.models.database import get_db
from app.models.portfolio import PriceAlert
from app.schemas.portfolio import (
    PriceAlertCreate,
    PriceAlertResponse,
    PriceAlertCheck,
)
from app.services.price_service import get_price

router = APIRouter(prefix="/alerts", tags=["alerts"])


@router.post("/", response_model=PriceAlertResponse, status_code=201)
def create_alert(alert: PriceAlertCreate, db: Session = Depends(get_db)):
    """Create a new price alert (e.g. 'THYAO above ₺300')."""
    db_alert = PriceAlert(**alert.model_dump())
    db.add(db_alert)
    db.commit()
    db.refresh(db_alert)
    return db_alert


@router.get("/", response_model=list[PriceAlertResponse])
def list_alerts(
    active_only: bool = True,
    ticker: Optional[str] = None,
    db: Session = Depends(get_db),
):
    """List price alerts, optionally filtered."""
    query = db.query(PriceAlert)
    if active_only:
        query = query.filter(PriceAlert.is_active == True)  # noqa: E712
    if ticker:
        query = query.filter(PriceAlert.ticker == ticker.upper())
    return query.order_by(PriceAlert.created_at.desc()).all()


@router.get("/check", response_model=list[PriceAlertCheck])
def check_alerts(db: Session = Depends(get_db)):
    """Check all active alerts against current prices. Returns triggered ones."""
    active = db.query(PriceAlert).filter(PriceAlert.is_active == True).all()  # noqa: E712
    results = []

    for alert in active:
        current_price = get_price(alert.ticker, alert.market)
        triggered = False

        if current_price is not None:
            if alert.direction == "above" and current_price >= alert.target_price:
                triggered = True
            elif alert.direction == "below" and current_price <= alert.target_price:
                triggered = True

        if triggered:
            alert.triggered_at = datetime.utcnow()
            alert.is_active = False
            db.commit()
            db.refresh(alert)

        results.append(PriceAlertCheck(
            id=alert.id,
            ticker=alert.ticker,
            market=alert.market,
            target_price=alert.target_price,
            direction=alert.direction,
            is_active=alert.is_active,
            triggered_at=alert.triggered_at,
            created_at=alert.created_at,
            note=alert.note,
            current_price=current_price,
            is_triggered=triggered,
        ))

    return results


@router.delete("/{alert_id}", status_code=204)
def delete_alert(alert_id: int, db: Session = Depends(get_db)):
    alert = db.query(PriceAlert).filter(PriceAlert.id == alert_id).first()
    if not alert:
        raise HTTPException(404, "Alert not found")
    db.delete(alert)
    db.commit()
