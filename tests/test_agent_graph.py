import json
from datetime import datetime, timezone

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from app.graph.nodes import market_analyst_node, portfolio_analysis_node
from app.graph.state import PortfolioState
from app.graph.workflow import _route_after_analyst, build_agent_graph
from app.models.analysis import CompanyInfo, HistoricalData, MarketPrice, OHLCVPoint, PortfolioAnalyzeResponse


def _sample_metrics():
    return {
        "holdings": [
            {
                "input_symbol": "AAA",
                "resolved_symbol": "AAA.NS",
                "quantity": 10,
                "average_purchase_price": 100,
                "invested_value": 1000,
                "latest_price": 110,
                "current_value": 1100,
                "absolute_pnl": 100,
                "percentage_return": 10.0,
                "allocation_percent": 100.0,
                "sector": "Tech",
                "status": "ok",
                "message": None,
                "company": {
                    "company_name": "AAA Ltd",
                    "sector": "Tech",
                    "industry": "Software",
                    "market_cap": 1,
                    "pe_ratio": 10,
                    "dividend_yield": 0.01,
                    "missing_fields": [],
                    "status": "ok",
                },
                "historical_data": {
                    "status": "ok",
                    "period": "1y",
                    "interval": "1d",
                    "points": [
                        {"date": "2025-01-01", "open": 100, "high": 101, "low": 99, "close": 100, "volume": 1},
                        {"date": "2025-01-02", "open": 101, "high": 102, "low": 100, "close": 101, "volume": 1},
                        {"date": "2025-01-03", "open": 102, "high": 103, "low": 101, "close": 102, "volume": 1},
                    ],
                },
            }
        ],
        "summary": {
            "total_invested_amount": 1000.0,
            "total_current_value": 1100.0,
            "total_absolute_pnl": 100.0,
            "overall_percentage_return": 10.0,
            "analyzed_holdings_count": 1,
            "unresolved_or_failed_holdings_count": 0,
            "sector_allocation": {"Tech": 100.0},
        },
        "risk": {
            "portfolio_volatility_annualized": 0.1,
            "portfolio_volatility_note": None,
            "max_drawdown": -0.01,
            "max_drawdown_note": None,
            "concentration_top_holding_percent": 100.0,
            "concentration_hhi": 10000.0,
            "concentration_note": "note",
            "assumptions": [],
        },
        "data_source": "yfinance",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "warnings": [],
        "limitations": [],
    }


def test_portfolio_analysis_node_populates_state(monkeypatch):
    class DummyResp:
        def model_dump(self, mode="json"):
            return _sample_metrics()

    monkeypatch.setattr("app.graph.nodes.analyze_portfolio", lambda holdings, history_period, history_interval: DummyResp())
    state = {
        "confirmed_holdings": [{"stock_symbol": "AAA", "quantity": 10, "average_purchase_price": 100}],
        "history_period": "1y",
        "history_interval": "1d",
    }
    out = portfolio_analysis_node(state)
    assert out["analysis_status"] == "portfolio_ready"
    assert out["portfolio_metrics"]["summary"]["total_current_value"] == 1100.0
    assert "AAA.NS" in out["market_data"]
    assert out["data_sources"][0]["name"] == "yfinance"


def test_route_after_analyst_tool_calls_and_end():
    state_tools = {"agent_messages": [AIMessage(content="", tool_calls=[{"name": "get_stock_price", "args": {"symbol": "AAA"}, "id": "1"}])], "analysis_status": "analyzing"}
    assert _route_after_analyst(state_tools) == "analyst_tools"
    state_end = {"agent_messages": [AIMessage(content="done")], "analysis_status": "analyzing"}
    assert _route_after_analyst(state_end) == "research"


class FakeLLM:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0

    def bind_tools(self, tools):
        return self

    def invoke(self, msgs):
        r = self.responses[min(self.calls, len(self.responses) - 1)]
        self.calls += 1
        return r


def test_market_analyst_node_final_json(monkeypatch):
    final_json = {
        "portfolio_overview": "Overview",
        "key_findings": ["Finding"],
        "stock_level_insights": [{"symbol": "AAA.NS", "observation": "up", "interpretation": "positive"}],
        "sector_observations": ["Tech heavy"],
        "risk_observations": ["Concentrated"],
        "supporting_data": [{"description": "pnl", "value": 100, "source": "calculation"}],
        "sources": [{"name": "yfinance", "type": "market_data", "detail": "prices"}],
        "warnings": [],
    }
    fake = FakeLLM([AIMessage(content=json.dumps(final_json))])
    monkeypatch.setattr("app.graph.nodes._build_llm", lambda with_tools=True: fake)
    state = {
        "portfolio_metrics": _sample_metrics(),
        "market_data": {},
        "company_information": {},
        "agent_messages": [],
        "tool_iterations": 0,
        "max_tool_iterations": 3,
    }
    out = market_analyst_node(state)
    assert out["analysis_status"] == "completed"
    assert out["analyst_findings"]["portfolio_overview"] == "Overview"
    assert out["final_analysis"]["key_findings"] == ["Finding"]


def test_market_analyst_node_respects_max_tool_iterations(monkeypatch):
    # LLM keeps requesting tools; node should stop at limit and force finalize
    tool_call_msg = AIMessage(content="", tool_calls=[{"name": "get_stock_price", "args": {"symbol": "AAA"}, "id": "t1"}])
    final_json = {
        "portfolio_overview": "Partial",
        "key_findings": [],
        "stock_level_insights": [],
        "sector_observations": [],
        "risk_observations": [],
        "supporting_data": [],
        "sources": [],
        "warnings": ["limited"],
    }
    seq = [tool_call_msg, AIMessage(content=json.dumps(final_json))]

    def fake_build(with_tools=True):
        return FakeLLM(seq if with_tools else [AIMessage(content=json.dumps(final_json))])

    monkeypatch.setattr("app.graph.nodes._build_llm", fake_build)
    state = {
        "portfolio_metrics": _sample_metrics(),
        "agent_messages": [HumanMessage(content="ctx")],
        "tool_iterations": 3,
        "max_tool_iterations": 3,
        "analysis_status": "analyzing",
    }
    out = market_analyst_node(state)
    assert out["analysis_status"] in {"completed", "partial"}
    assert any("Maximum tool-call iterations" in w for w in (out.get("warnings") or []))


def test_build_agent_graph_compiles():
    g = build_agent_graph()
    assert g is not None
