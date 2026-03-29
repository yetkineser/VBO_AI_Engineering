from datetime import datetime, timezone
from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean, Enum as SAEnum
import enum


def _utcnow():
    return datetime.now(timezone.utc)

from app.models.database import Base


class MarketType(str, enum.Enum):
    BIST = "BIST"
    US = "US"
    CRYPTO = "CRYPTO"


class EmotionTag(str, enum.Enum):
    RESEARCH = "research"
    CONVICTION = "conviction"
    FOMO = "fomo"
    PANIC = "panic"
    INTUITION = "intuition"
    DCA = "dca"


class Holding(Base):
    """A single position in the portfolio (e.g. 10 shares of THYAO bought on 2025-01-15)."""

    __tablename__ = "holdings"

    id = Column(Integer, primary_key=True, index=True)
    ticker = Column(String, nullable=False, index=True)
    market = Column(SAEnum(MarketType), nullable=False)
    quantity = Column(Float, nullable=False)
    buy_price = Column(Float, nullable=False)
    buy_currency = Column(String, default="TRY")
    commission = Column(Float, default=0.0)
    buy_date = Column(DateTime, default=_utcnow)
    notes = Column(String, nullable=True)
    emotion_tag = Column(String, nullable=True)
    created_at = Column(DateTime, default=_utcnow)
    updated_at = Column(DateTime, default=_utcnow, onupdate=_utcnow)


class JournalEntry(Base):
    """Investment journal — thoughts, learnings, decisions."""

    __tablename__ = "journal_entries"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String, nullable=False)
    content = Column(String, nullable=False)
    entry_type = Column(String, default="thought")
    related_ticker = Column(String, nullable=True)
    created_at = Column(DateTime, default=_utcnow)


class PriceAlert(Base):
    """Price alert — notify when a ticker crosses a threshold."""

    __tablename__ = "price_alerts"

    id = Column(Integer, primary_key=True, index=True)
    ticker = Column(String, nullable=False, index=True)
    market = Column(SAEnum(MarketType), nullable=False)
    target_price = Column(Float, nullable=False)
    direction = Column(String, nullable=False)  # "above" or "below"
    is_active = Column(Boolean, default=True)
    triggered_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=_utcnow)
    note = Column(String, nullable=True)
