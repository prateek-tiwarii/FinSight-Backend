import pytest
from pydantic import ValidationError

from app.models.portfolio import ExtractedHolding, PortfolioExtractionResponse


def test_extracted_holding_valid():
    holding = ExtractedHolding(
        stock_symbol="RELIANCE",
        quantity=10,
        average_purchase_price=2500.5,
    )
    assert holding.stock_symbol == "RELIANCE"
    assert holding.quantity == 10
    assert holding.average_purchase_price == 2500.5


def test_extracted_holding_rejects_invalid_values():
    with pytest.raises(ValidationError):
        ExtractedHolding(stock_symbol="", quantity=1, average_purchase_price=100)
    with pytest.raises(ValidationError):
        ExtractedHolding(stock_symbol="TCS", quantity=0, average_purchase_price=100)
    with pytest.raises(ValidationError):
        ExtractedHolding(stock_symbol="TCS", quantity=1, average_purchase_price=-1)


def test_portfolio_response_defaults():
    resp = PortfolioExtractionResponse(holdings=[])
    assert resp.requires_confirmation is True
    assert resp.source == "screenshot"
    assert resp.warnings == []
    assert resp.extraction_notes is None
