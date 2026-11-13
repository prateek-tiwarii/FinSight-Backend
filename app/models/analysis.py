from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

from app.models.portfolio import ExtractedHolding


class ConfirmedHolding(BaseModel):
    stock_symbol: str = Field(..., min_length=1)
    quantity: float = Field(..., gt=0)
    average_purchase_price: float = Field(..., ge=0)
    exchange: str | None = Field(default=None, description="NSE/BSE if known")


class PortfolioAnalyzeRequest(BaseModel):
    holdings: list[ConfirmedHolding] = Field(..., min_length=1)
    history_period: str = "1y"
    history_interval: str = "1d"


class MarketPrice(BaseModel):
    symbol: str
    resolved_symbol: str | None = None
    latest_price: float | None = None
    currency: str | None = None
    market_timestamp: datetime | None = None
    source: str = "yfinance"
    status: Literal["ok", "unresolved", "missing_data", "error"]
    message: str | None = None


class OHLCVPoint(BaseModel):
    date: str
    open: float | None = None
    high: float | None = None
    low: float | None = None
    close: float | None = None
    volume: int | float | None = None


class HistoricalData(BaseModel):
    symbol: str
    resolved_symbol: str | None = None
    period: str
    interval: str
    points: list[OHLCVPoint] = Field(default_factory=list)
    source: str = "yfinance"
    status: Literal["ok", "unresolved", "missing_data", "error"]
    message: str | None = None


class CompanyInfo(BaseModel):
    symbol: str
    resolved_symbol: str | None = None
    company_name: str | None = None
    sector: str | None = None
    industry: str | None = None
    market_cap: float | None = None
    pe_ratio: float | None = None
    dividend_yield: float | None = None
    source: str = "yfinance"
    status: Literal["ok", "unresolved", "missing_data", "error"]
    message: str | None = None
    missing_fields: list[str] = Field(default_factory=list)


class HoldingAnalysis(BaseModel):
    input_symbol: str
    resolved_symbol: str | None = None
    quantity: float
    average_purchase_price: float
    invested_value: float | None = None
    latest_price: float | None = None
    current_value: float | None = None
    absolute_pnl: float | None = None
    percentage_return: float | None = None
    allocation_percent: float | None = None
    sector: str | None = None
    status: Literal["ok", "unresolved", "missing_price", "error"]
    message: str | None = None
    company: CompanyInfo | None = None
    historical_data: HistoricalData | None = None


class RiskMetrics(BaseModel):
    portfolio_volatility_annualized: float | None = None
    portfolio_volatility_note: str | None = None
    max_drawdown: float | None = None
    max_drawdown_note: str | None = None
    concentration_top_holding_percent: float | None = None
    concentration_hhi: float | None = None
    concentration_note: str = (
        "Concentration metrics (top holding %, HHI) are basic indicators, not a complete risk assessment."
    )
    assumptions: list[str] = Field(default_factory=list)


class PortfolioSummary(BaseModel):
    total_invested_amount: float
    total_current_value: float | None = None
    total_absolute_pnl: float | None = None
    overall_percentage_return: float | None = None
    analyzed_holdings_count: int
    unresolved_or_failed_holdings_count: int
    sector_allocation: dict[str, float] = Field(default_factory=dict)
    # Transparency for partial valuations (defaults keep older payloads valid).
    total_holdings_count: int = 0
    priced_holdings_count: int = 0
    missing_price_holdings_count: int = 0
    unresolved_holdings_count: int = 0
    failed_holdings_count: int = 0
    analyzed_invested_amount: float | None = None
    unvalued_invested_amount: float | None = None
    is_partial: bool = False
    valuation_note: str | None = None


class PortfolioAnalyzeResponse(BaseModel):
    holdings: list[HoldingAnalysis]
    summary: PortfolioSummary
    risk: RiskMetrics
    data_source: str = "yfinance"
    generated_at: datetime
    warnings: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
