from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.models.database import init_db
from app.routers import portfolio, journal, market, alerts


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield


app = FastAPI(
    title="InvestMentor API",
    description="Portfolio tracker, investment journal, and AI mentor for BIST, US stocks, and crypto.",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(portfolio.router, prefix="/api/v1")
app.include_router(journal.router, prefix="/api/v1")
app.include_router(market.router, prefix="/api/v1")
app.include_router(alerts.router, prefix="/api/v1")


@app.get("/")
def root():
    return {
        "app": "InvestMentor",
        "version": "0.1.0",
        "docs": "/docs",
        "endpoints": {
            "portfolio": "/api/v1/portfolio",
            "journal": "/api/v1/journal",
            "market": "/api/v1/market",
            "alerts": "/api/v1/alerts",
        },
    }
