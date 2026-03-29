"""Tests for the portfolio and journal API."""


def test_root(client):
    resp = client.get("/")
    assert resp.status_code == 200
    data = resp.json()
    assert data["app"] == "InvestMentor"
    assert "portfolio" in data["endpoints"]


def test_add_holding(client):
    payload = {
        "ticker": "THYAO",
        "market": "BIST",
        "quantity": 100,
        "buy_price": 285.50,
        "buy_currency": "TRY",
        "commission": 15.0,
        "notes": "Strong Q4 results",
        "emotion_tag": "research",
    }
    resp = client.post("/api/v1/portfolio/holdings", json=payload)
    assert resp.status_code == 201
    data = resp.json()
    assert data["ticker"] == "THYAO"
    assert data["quantity"] == 100
    assert data["commission"] == 15.0


def test_list_holdings(client):
    for ticker, market, currency in [
        ("THYAO", "BIST", "TRY"),
        ("AAPL", "US", "USD"),
    ]:
        client.post("/api/v1/portfolio/holdings", json={
            "ticker": ticker,
            "market": market,
            "quantity": 10,
            "buy_price": 100.0,
            "buy_currency": currency,
        })

    resp = client.get("/api/v1/portfolio/holdings")
    assert resp.status_code == 200
    assert len(resp.json()) == 2

    resp = client.get("/api/v1/portfolio/holdings?market=BIST")
    assert len(resp.json()) == 1
    assert resp.json()[0]["ticker"] == "THYAO"


def test_update_holding(client):
    resp = client.post("/api/v1/portfolio/holdings", json={
        "ticker": "AAPL",
        "market": "US",
        "quantity": 5,
        "buy_price": 190.0,
        "buy_currency": "USD",
    })
    holding_id = resp.json()["id"]

    resp = client.put(f"/api/v1/portfolio/holdings/{holding_id}", json={
        "quantity": 10,
        "notes": "Doubled position after earnings",
        "emotion_tag": "conviction",
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["quantity"] == 10
    assert data["notes"] == "Doubled position after earnings"
    assert data["buy_price"] == 190.0  # unchanged


def test_delete_holding(client):
    resp = client.post("/api/v1/portfolio/holdings", json={
        "ticker": "AAPL",
        "market": "US",
        "quantity": 5,
        "buy_price": 190.0,
        "buy_currency": "USD",
    })
    holding_id = resp.json()["id"]

    resp = client.delete(f"/api/v1/portfolio/holdings/{holding_id}")
    assert resp.status_code == 204

    resp = client.get("/api/v1/portfolio/holdings")
    assert len(resp.json()) == 0


def test_delete_nonexistent_holding(client):
    resp = client.delete("/api/v1/portfolio/holdings/999")
    assert resp.status_code == 404


def test_invalid_market(client):
    resp = client.post("/api/v1/portfolio/holdings", json={
        "ticker": "INVALID",
        "market": "MOON",
        "quantity": 1,
        "buy_price": 1.0,
    })
    assert resp.status_code == 422


def test_journal_crud(client):
    payload = {
        "title": "İlk yatırım kararım",
        "content": "THYAO aldım çünkü bilanço güçlü",
        "entry_type": "thought",
        "related_ticker": "THYAO",
    }
    resp = client.post("/api/v1/journal/", json=payload)
    assert resp.status_code == 201
    entry_id = resp.json()["id"]
    assert resp.json()["title"] == "İlk yatırım kararım"

    resp = client.get("/api/v1/journal/")
    assert len(resp.json()) == 1

    resp = client.get(f"/api/v1/journal/{entry_id}")
    assert resp.status_code == 200
    assert resp.json()["content"] == "THYAO aldım çünkü bilanço güçlü"

    resp = client.delete(f"/api/v1/journal/{entry_id}")
    assert resp.status_code == 204


def test_journal_filter_by_type(client):
    client.post("/api/v1/journal/", json={
        "title": "Lesson 1", "content": "Sabırlı ol", "entry_type": "lesson",
    })
    client.post("/api/v1/journal/", json={
        "title": "Strategy", "content": "DCA", "entry_type": "strategy",
    })

    resp = client.get("/api/v1/journal/?entry_type=lesson")
    assert len(resp.json()) == 1
    assert resp.json()[0]["entry_type"] == "lesson"


def test_portfolio_distribution(client):
    for ticker, market, currency, qty, price in [
        ("THYAO", "BIST", "TRY", 100, 300.0),
        ("AAPL", "US", "USD", 5, 200.0),
        ("bitcoin", "CRYPTO", "USD", 0.1, 60000.0),
    ]:
        client.post("/api/v1/portfolio/holdings", json={
            "ticker": ticker,
            "market": market,
            "quantity": qty,
            "buy_price": price,
            "buy_currency": currency,
        })

    resp = client.get("/api/v1/portfolio/distribution")
    assert resp.status_code == 200
    data = resp.json()
    assert "by_market" in data
    assert "by_currency" in data
    assert len(data["by_market"]) == 3
