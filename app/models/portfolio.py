from pydantic import BaseModel, Field


class ExtractedHolding(BaseModel):
    stock_symbol: str = Field(..., min_length=1, description="Ticker or symbol as seen in screenshot")
    quantity: float = Field(..., gt=0, description="Number of shares/units held")
    average_purchase_price: float = Field(..., ge=0, description="Average buy price per unit")


class PortfolioExtractionResponse(BaseModel):
    holdings: list[ExtractedHolding]
    requires_confirmation: bool = True
    source: str = "screenshot"
    warnings: list[str] = Field(default_factory=list)
    extraction_notes: str | None = None
