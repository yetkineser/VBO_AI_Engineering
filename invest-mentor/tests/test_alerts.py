"""Tests for the price alerts API."""


def test_create_alert(client):
    resp = client.post("/api/v1/alerts/", json={
        "ticker": "THYAO",
        "market": "BIST",
        "target_price": 300.0,
        "direction": "above",
        "note": "THYAO 300'ü geçerse sat",
    })
    assert resp.status_code == 201
    data = resp.json()
    assert data["ticker"] == "THYAO"
    assert data["target_price"] == 300.0
    assert data["direction"] == "above"
    assert data["is_active"] is True


def test_list_alerts(client):
    client.post("/api/v1/alerts/", json={
        "ticker": "AAPL",
        "market": "US",
        "target_price": 200.0,
        "direction": "below",
    })
    client.post("/api/v1/alerts/", json={
        "ticker": "THYAO",
        "market": "BIST",
        "target_price": 300.0,
        "direction": "above",
    })

    resp = client.get("/api/v1/alerts/")
    assert resp.status_code == 200
    assert len(resp.json()) == 2


def test_delete_alert(client):
    resp = client.post("/api/v1/alerts/", json={
        "ticker": "AAPL",
        "market": "US",
        "target_price": 200.0,
        "direction": "below",
    })
    alert_id = resp.json()["id"]

    resp = client.delete(f"/api/v1/alerts/{alert_id}")
    assert resp.status_code == 204

    resp = client.get("/api/v1/alerts/?active_only=false")
    assert len(resp.json()) == 0


def test_delete_nonexistent_alert(client):
    resp = client.delete("/api/v1/alerts/999")
    assert resp.status_code == 404
