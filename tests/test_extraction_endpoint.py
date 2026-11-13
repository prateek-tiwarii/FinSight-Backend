from fastapi.testclient import TestClient

from app.main import app
from app.models.portfolio import ExtractedHolding, PortfolioExtractionResponse
from tests.helpers import prime_csrf

client = TestClient(app)
prime_csrf(client)


def test_health():
    resp = client.get("/api/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_extract_rejects_unsupported_type():
    resp = client.post(
        "/api/extract/portfolio",
        files={"file": ("portfolio.txt", b"not an image", "text/plain")},
    )
    assert resp.status_code == 400


def test_extract_rejects_empty_file():
    resp = client.post(
        "/api/extract/portfolio",
        files={"file": ("portfolio.png", b"", "image/png")},
    )
    assert resp.status_code == 400


def test_extract_success_with_mock(monkeypatch):
    async def fake_extract(self, image_bytes, mime_type):
        return PortfolioExtractionResponse(
            holdings=[
                ExtractedHolding(
                    stock_symbol="INFY",
                    quantity=5,
                    average_purchase_price=1500.0,
                )
            ],
            warnings=[],
            extraction_notes="ok",
        )

    monkeypatch.setattr(
        "app.services.extraction.PortfolioExtractionService.extract",
        fake_extract,
    )

    resp = client.post(
        "/api/extract/portfolio",
        files={"file": ("portfolio.png", b"\x89PNG\r\n\x1a\nfake", "image/png")},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["requires_confirmation"] is True
    assert data["source"] == "screenshot"
    assert data["holdings"][0]["stock_symbol"] == "INFY"
    assert data["holdings"][0]["quantity"] == 5
    assert data["holdings"][0]["average_purchase_price"] == 1500.0
