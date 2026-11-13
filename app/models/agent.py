import uuid
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

from app.models.analysis import PortfolioAnalyzeRequest, PortfolioAnalyzeResponse


class MarketAnalystFindings(BaseModel):
    portfolio_overview: str
    key_findings: list[str] = Field(default_factory=list)
    stock_level_insights: list[dict[str, Any]] = Field(default_factory=list)
    sector_observations: list[str] = Field(default_factory=list)
    risk_observations: list[str] = Field(default_factory=list)
    supporting_data: list[dict[str, Any]] = Field(default_factory=list)
    sources: list[dict[str, Any]] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class ResearchDevelopment(BaseModel):
    related_company: str | None = None
    headline: str | None = None
    summary: str | None = None
    sentiment: Literal["positive", "negative", "neutral", "mixed", "unclear"] | str | None = None
    publication_date: str | None = None
    source_name: str | None = None
    source_url: str | None = None
    why_it_may_matter: str | None = None
    evidence_or_limitation_notes: str | None = None


class CompanyResearch(BaseModel):
    symbol: str | None = None
    company_name: str | None = None
    developments: list[ResearchDevelopment] = Field(default_factory=list)
    notes: str | None = None


class ResearchFindings(BaseModel):
    research_overview: str
    company_research: list[CompanyResearch] = Field(default_factory=list)
    important_developments: list[ResearchDevelopment] = Field(default_factory=list)
    positive_signals: list[str] = Field(default_factory=list)
    negative_signals: list[str] = Field(default_factory=list)
    neutral_observations: list[str] = Field(default_factory=list)
    potential_concerns: list[str] = Field(default_factory=list)
    sources: list[dict[str, Any]] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


# M5: Risk Critic
class CriticChallengedClaim(BaseModel):
    claim: str
    challenged_by: str | None = None
    reasoning: str | None = None
    evidence: list[dict[str, Any]] = Field(default_factory=list)


class CriticIdentifiedRisk(BaseModel):
    title: str
    description: str
    severity: Literal["low", "medium", "high", "critical"]
    evidence: list[dict[str, Any]] = Field(default_factory=list)
    related_stock_or_sector: str | None = None
    reasoning_summary: str | None = None


class RiskCriticFindings(BaseModel):
    critic_overview: str
    challenged_claims: list[CriticChallengedClaim] = Field(default_factory=list)
    identified_risks: list[CriticIdentifiedRisk] = Field(default_factory=list)
    contradictions: list[dict[str, Any]] = Field(default_factory=list)
    missing_evidence: list[str] = Field(default_factory=list)
    severity_classification: dict[str, Any] = Field(default_factory=dict)
    recommendations_for_report: list[str] = Field(default_factory=list)
    supporting_sources: list[dict[str, Any]] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


# M5: Behavioral Questionnaire
class BehavioralQuestionnaireOption(BaseModel):
    value: str
    label: str


class BehavioralQuestionnaireQuestion(BaseModel):
    key: str
    question: str
    type: Literal["single_choice"] = "single_choice"
    required: bool = True
    options: list[BehavioralQuestionnaireOption] = Field(default_factory=list)


class BehavioralQuestionnaireSchema(BaseModel):
    title: str
    description: str
    questions: list[BehavioralQuestionnaireQuestion] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


INVESTMENT_HORIZON_OPTIONS = [
    BehavioralQuestionnaireOption(value="less_than_1_year", label="Less than 1 year"),
    BehavioralQuestionnaireOption(value="1_3_years", label="1–3 years"),
    BehavioralQuestionnaireOption(value="3_5_years", label="3–5 years"),
    BehavioralQuestionnaireOption(value="more_than_5_years", label="More than 5 years"),
]
RISK_TOLERANCE_OPTIONS = [
    BehavioralQuestionnaireOption(value="conservative", label="Conservative"),
    BehavioralQuestionnaireOption(value="moderate", label="Moderate"),
    BehavioralQuestionnaireOption(value="aggressive", label="Aggressive"),
]
INVESTMENT_OBJECTIVE_OPTIONS = [
    BehavioralQuestionnaireOption(value="capital_preservation", label="Capital preservation"),
    BehavioralQuestionnaireOption(value="long_term_wealth_creation", label="Long-term wealth creation"),
    BehavioralQuestionnaireOption(value="regular_income", label="Regular income"),
    BehavioralQuestionnaireOption(value="capital_appreciation", label="Capital appreciation"),
]
REACTION_DECLINE_OPTIONS = [
    BehavioralQuestionnaireOption(value="sell_most", label="Sell most holdings"),
    BehavioralQuestionnaireOption(value="sell_some", label="Sell some holdings"),
    BehavioralQuestionnaireOption(value="hold", label="Hold investments"),
    BehavioralQuestionnaireOption(value="invest_more_if_fundamentals_strong", label="Invest more if fundamentals remain strong"),
]
EXPERIENCE_OPTIONS = [
    BehavioralQuestionnaireOption(value="beginner", label="Beginner"),
    BehavioralQuestionnaireOption(value="intermediate", label="Intermediate"),
    BehavioralQuestionnaireOption(value="experienced", label="Experienced"),
]


def get_behavioral_questionnaire_schema() -> BehavioralQuestionnaireSchema:
    return BehavioralQuestionnaireSchema(
        title="FinSight AI Behavioral Questionnaire",
        description="Please complete these questions. Responses are used to tailor the final report to your preferences. This is not a professional psychometric assessment.",
        questions=[
            BehavioralQuestionnaireQuestion(key="investment_horizon", question="Investment horizon:", options=INVESTMENT_HORIZON_OPTIONS, required=True),
            BehavioralQuestionnaireQuestion(key="risk_tolerance", question="Risk tolerance:", options=RISK_TOLERANCE_OPTIONS, required=True),
            BehavioralQuestionnaireQuestion(key="primary_investment_objective", question="Primary investment objective:", options=INVESTMENT_OBJECTIVE_OPTIONS, required=True),
            BehavioralQuestionnaireQuestion(key="reaction_to_15_20pct_decline", question="Reaction to a 15–20% portfolio decline:", options=REACTION_DECLINE_OPTIONS, required=True),
            BehavioralQuestionnaireQuestion(key="investment_experience", question="Investment experience:", options=EXPERIENCE_OPTIONS, required=True),
            BehavioralQuestionnaireQuestion(
                key="financial_goals",
                question="Optional additional context - Financial goals:",
                options=[BehavioralQuestionnaireOption(value="free_text", label="Free text (provide in 'additional_context.financial_goals')")],
                required=False,
            ),
            BehavioralQuestionnaireQuestion(
                key="liquidity_requirements",
                question="Optional additional context - Liquidity requirements:",
                options=[BehavioralQuestionnaireOption(value="free_text", label="Free text (provide in 'additional_context.liquidity_requirements')")],
                required=False,
            ),
            BehavioralQuestionnaireQuestion(
                key="existing_investment_constraints",
                question="Optional additional context - Existing investment constraints:",
                options=[BehavioralQuestionnaireOption(value="free_text", label="Free text (provide in 'additional_context.existing_investment_constraints')")],
                required=False,
            ),
        ],
        notes=[
            "Questionnaire must be completed by the user. Answers are not inferred from screenshots.",
            "Required answers must be provided to resume analysis.",
        ],
    )


class BehavioralQuestionnaireResponse(BaseModel):
    investment_horizon: Literal["less_than_1_year", "1_3_years", "3_5_years", "more_than_5_years"]
    risk_tolerance: Literal["conservative", "moderate", "aggressive"]
    primary_investment_objective: Literal["capital_preservation", "long_term_wealth_creation", "regular_income", "capital_appreciation"]
    reaction_to_15_20pct_decline: Literal["sell_most", "sell_some", "hold", "invest_more_if_fundamentals_strong"]
    investment_experience: Literal["beginner", "intermediate", "experienced"]
    additional_context: dict[str, Any] = Field(default_factory=dict)


class BehavioralProfile(BaseModel):
    stated_preferences: BehavioralQuestionnaireResponse
    portfolio_risk_profile: dict[str, Any] = Field(default_factory=dict)
    mismatch_indicators: list[dict[str, Any]] = Field(default_factory=list)
    behavioral_observations: list[str] = Field(default_factory=list)
    educational_notes: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class BehavioralFindings(BaseModel):
    behavioral_overview: str
    profile: BehavioralProfile
    comparisons_to_portfolio: list[dict[str, Any]] = Field(default_factory=list)
    potential_behavioral_risks: list[dict[str, Any]] = Field(default_factory=list)
    areas_of_conflict: list[dict[str, Any]] = Field(default_factory=list)
    recommendations_educational: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


# M5: Final Advisor
class FinalPortfolioReport(BaseModel):
    report_title: str
    executive_summary: str
    overall_assessment: str
    portfolio_strengths: list[str] = Field(default_factory=list)
    portfolio_weaknesses: list[str] = Field(default_factory=list)
    performance_insights: list[dict[str, Any]] = Field(default_factory=list)
    risk_assessment: dict[str, Any] = Field(default_factory=dict)
    research_summary: dict[str, Any] = Field(default_factory=dict)
    behavioral_assessment: dict[str, Any] = Field(default_factory=dict)
    preference_mismatches: list[dict[str, Any]] = Field(default_factory=list)
    key_observations: list[str] = Field(default_factory=list)
    areas_requiring_attention: list[str] = Field(default_factory=list)
    critic_findings_summary: dict[str, Any] = Field(default_factory=dict)
    supporting_sources: list[dict[str, Any]] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    generated_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")


class AgentAnalyzeRequest(PortfolioAnalyzeRequest):
    # Deprecated/ignored for ownership: ownership is always derived from the verified
    # server-side session or guest session.
    user_id: str | None = None
    max_tool_iterations: int = Field(default=3, ge=0, le=10)
    # Optional: allows non-interactive/programmatic runs (e.g. tests) when provided.
    # In interactive HITL flow, omit this and complete questionnaire via /api/agent/resume.
    questionnaire_responses: BehavioralQuestionnaireResponse | None = None
    portfolio_id: str | None = None
    portfolio_name: str | None = None


class AgentResumeRequest(BaseModel):
    thread_id: str
    questionnaire_responses: BehavioralQuestionnaireResponse


class AgentAnalyzeResponse(BaseModel):
    portfolio_metrics: PortfolioAnalyzeResponse | None = None
    analyst_findings: MarketAnalystFindings | None = None
    research_findings: ResearchFindings | None = None
    research_sources: list[dict[str, Any]] = Field(default_factory=list)
    research_status: str | None = None
    research_warnings: list[str] = Field(default_factory=list)
    tool_execution_summaries: list[dict[str, Any]] = Field(default_factory=list)
    sources: list[dict[str, Any]] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    analysis_status: str
    final_analysis: dict[str, Any] | None = None
    # M5
    thread_id: str | None = None
    workflow_stage: str | None = None
    human_review_status: str | None = None
    pending_questionnaire: bool | None = None
    questionnaire_schema: BehavioralQuestionnaireSchema | None = None
    critic_findings: RiskCriticFindings | None = None
    critic_status: str | None = None
    behavioral_profile: BehavioralProfile | None = None
    behavioral_findings: BehavioralFindings | None = None
    behavioral_status: str | None = None
    final_report: FinalPortfolioReport | None = None
    final_advisor_status: str | None = None
    # Persistence identifiers (M7)
    analysis_id: uuid.UUID | None = None
    portfolio_id: uuid.UUID | None = None
    report_id: uuid.UUID | None = None

