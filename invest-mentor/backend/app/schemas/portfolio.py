from datetime import datetime
from pydantic import BaseModel, Field
from typing import Optional
from enum import Enum


class MarketType(str, Enum):
    BIST = "BIST"
    US = "US"
    CRYPTO = "CRYPTO"


class EmotionTag(str, Enum):
    RESEARCH = "research"
    CONVICTION = "conviction"
    FOMO = "fomo"
    PANIC = "panic"
    INTUITION = "intuition"
    DCA = "dca"


# ─── Holdings ───────────────────────────────────────────

class HoldingCreate(BaseModel):
    ticker: str = Field(..., min_length=1, max_length=20, examples=["THYAO"])
    market: MarketType = Field(..., examples=["BIST"])
    quantity: float = Field(..., gt=0, examples=[100])
    buy_price: float = Field(..., gt=0, examples=[285.50])
    buy_currency: str = Field(default="TRY", examples=["TRY"])
    commission: float = Field(default=0.0, ge=0, examples=[15.0])
    buy_date: Optional[datetime] = None
    notes: Optional[str] = Field(None, examples=["Güçlü 2024 bilançosu"])
    emotion_tag: Optional[EmotionTag] = Field(None, examples=["research"])


class HoldingUpdate(BaseModel):
    """Partial update — only set fields are changed."""
    quantity: Optional[float] = Field(None, gt=0)
    buy_price: Optional[float] = Field(None, gt=0)
    commission: Optional[float] = Field(None, ge=0)
    notes: Optional[str] = None
    emotion_tag: Optional[EmotionTag] = None


class HoldingResponse(BaseModel):
    id: int
    ticker: str
    market: str
    quantity: float
    buy_price: float
    buy_currency: str
    commission: float
    buy_date: datetime
    notes: Optional[str]
    emotion_tag: Optional[str]
    created_at: datetime

    model_config = {"from_attributes": True}


class HoldingWithPrice(HoldingResponse):
    """Holding enriched with live market data."""
    current_price: Optional[float] = None
    current_value: Optional[float] = None
    total_cost: Optional[float] = None
    profit_loss: Optional[float] = None
    profit_loss_pct: Optional[float] = None


# ─── Journal ────────────────────────────────────────────

class JournalCreate(BaseModel):
    title: str = Field(..., min_length=1)
    content: str = Field(..., min_length=1)
    entry_type: str = "thought"
    related_ticker: Optional[str] = None


class JournalResponse(BaseModel):
    id: int
    title: str
    content: str
    entry_type: str
    related_ticker: Optional[str]
    created_at: datetime

    model_config = {"from_attributes": True}


# ─── Portfolio Summary ──────────────────────────────────

class PortfolioSummary(BaseModel):
    total_invested: float
    total_commission: float
    current_value: float
    total_profit_loss: float
    total_profit_loss_pct: float
    holdings_count: int
    holdings: list[HoldingWithPrice]


class AllocationItem(BaseModel):
    label: str
    value: float
    percentage: float


class PortfolioDistribution(BaseModel):
    by_market: list[AllocationItem]
    by_currency: list[AllocationItem]
    total_invested: float


# ─── Price Alerts ───────────────────────────────────────

class AlertDirection(str, Enum):
    ABOVE = "above"
    BELOW = "below"


class PriceAlertCreate(BaseModel):
    ticker: str = Field(..., min_length=1, max_length=20)
    market: MarketType
    target_price: float = Field(..., gt=0)
    direction: AlertDirection
    note: Optional[str] = None


class PriceAlertResponse(BaseModel):
    id: int
    ticker: str
    market: str
    target_price: float
    direction: str
    is_active: bool
    triggered_at: Optional[datetime]
    created_at: datetime
    note: Optional[str]

    model_config = {"from_attributes": True}


class PriceAlertCheck(PriceAlertResponse):
    """Alert response with current price info for check endpoint."""
    current_price: Optional[float] = None
    is_triggered: bool = False
