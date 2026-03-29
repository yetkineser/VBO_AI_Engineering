from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import Optional

from app.models.database import get_db
from app.models.portfolio import JournalEntry
from app.schemas.portfolio import JournalCreate, JournalResponse

router = APIRouter(prefix="/journal", tags=["journal"])


@router.post("/", response_model=JournalResponse, status_code=201)
def create_entry(entry: JournalCreate, db: Session = Depends(get_db)):
    """Write a new journal entry."""
    db_entry = JournalEntry(**entry.model_dump())
    db.add(db_entry)
    db.commit()
    db.refresh(db_entry)
    return db_entry


@router.get("/", response_model=list[JournalResponse])
def list_entries(
    entry_type: Optional[str] = None,
    ticker: Optional[str] = None,
    db: Session = Depends(get_db),
):
    """List journal entries, optionally filtered."""
    query = db.query(JournalEntry)
    if entry_type:
        query = query.filter(JournalEntry.entry_type == entry_type)
    if ticker:
        query = query.filter(JournalEntry.related_ticker == ticker.upper())
    return query.order_by(JournalEntry.created_at.desc()).all()


@router.get("/{entry_id}", response_model=JournalResponse)
def get_entry(entry_id: int, db: Session = Depends(get_db)):
    entry = db.query(JournalEntry).filter(JournalEntry.id == entry_id).first()
    if not entry:
        raise HTTPException(404, "Journal entry not found")
    return entry


@router.delete("/{entry_id}", status_code=204)
def delete_entry(entry_id: int, db: Session = Depends(get_db)):
    entry = db.query(JournalEntry).filter(JournalEntry.id == entry_id).first()
    if not entry:
        raise HTTPException(404, "Journal entry not found")
    db.delete(entry)
    db.commit()
