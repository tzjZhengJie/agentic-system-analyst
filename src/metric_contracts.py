#A MetricContract is a dataclass that encodes structural invariants for each metric:

# resolve_contracts(question) — keyword-matches question to contracts (no if-statements)
# enforce_sql_guards(sql, contracts) — checks SQL structure before execution
# enforce_result_invariants(df, contracts) — checks result after execution
# build_contract_prompt_block(contracts) — builds structured prompt injection block

# src/metric_contracts.py
# ==========================================
# METRIC CONTRACT ENGINE
# ==========================================
# This module is the Single Source of Truth for how each
# analytical metric is STRUCTURALLY defined, not just described.
#
# A MetricContract encodes:
#   - Which CTE building blocks are required
#   - What the required output columns must be named and bounded
#   - What logical invariants must hold in the result
#   - What SQL guard clauses must appear in any correct query
#
# This is an architecture-level solution. It does not hardcode SQL.
# It encodes STRUCTURAL INVARIANTS that any SQL for a given metric
# must satisfy, regardless of how the SQL was generated.
#
# The Validator then checks generated SQL and result DataFrames against
# these contracts — preventing semantic hallucinations without
# relying on keyword matching or metric-specific regex patches.
#
# How to extend:
# Add a new entry to METRIC_CONTRACTS. The validator, SQL generator,
# and corrector all read from this registry automatically.
# No other files need changing.
#
# Generalises to: retention, conversion_rate, DAU/WAU/MAU,
# funnel analysis, cohort analysis, growth metrics — any metric
# with a well-defined numerator, denominator, and result range.

from dataclasses import dataclass, field
from typing import Callable, List, Optional
import pandas as pd


@dataclass
class ColumnContract:
    """
    Defines expectations for a single output column.
    All checks are pure functions — no hardcoded metric names.
    """
    name_pattern: str           # substring that must appear in the column name
    role: str                   # 'count', 'rate', 'date', 'dimension'
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    must_be_positive: bool = False
    never_exceeds_col: Optional[str] = None  # column name pattern that this col must be <= to


@dataclass
class MetricContract:
    """
    Defines the full structural contract for one analytical metric.

    Fields:
        metric_id         : canonical name used as registry key
        keywords          : user-phrase triggers for matching
        required_col_roles: list of ColumnContract specs the result must satisfy
        sql_guard_phrases : phrases that MUST appear in the SQL (structural guards)
        sql_forbidden_phrases: phrases that must NOT appear (common mistakes)
        result_invariants : list of (col_a_pattern, op, col_b_pattern) checks
                           e.g. ('churned', '<=', 'base') means every row
                           where there is a churned column must have
                           churned <= base
        denominator_col   : name pattern of the base/denominator column
        numerator_col     : name pattern of the numerator column
        rate_col          : name pattern of the rate/percentage output column
    """
    metric_id: str
    keywords: List[str]
    required_col_roles: List[ColumnContract] = field(default_factory=list)
    sql_guard_phrases: List[str] = field(default_factory=list)
    sql_forbidden_phrases: List[str] = field(default_factory=list)
    result_invariants: List[tuple] = field(default_factory=list)
    denominator_col: Optional[str] = None
    numerator_col: Optional[str] = None
    rate_col: Optional[str] = None
    # Note: expected_rate_range intentionally omitted.
    # Rate ranges are dataset-specific and cause false positives on legitimate spikes.
    # Use result_invariants (e.g. churned <= base) for structural checks instead.


# ── CONTRACT REGISTRY ─────────────────────────────────────────────────────────
# Each entry encodes domain knowledge as structural constraints.
# Not hardcoded SQL — structural invariants that apply to ALL valid SQL
# for that metric.

METRIC_CONTRACTS: dict[str, MetricContract] = {

    "churn_rate": MetricContract(
        metric_id="churn_rate",
        keywords=["churn", "churned", "churn rate", "monthly churn",
                  "user churn", "lost users", "inactive users"],
        required_col_roles=[
            ColumnContract(name_pattern="base",     role="count", must_be_positive=True),
            ColumnContract(name_pattern="churned",  role="count", min_value=0),
            ColumnContract(name_pattern="rate",     role="rate",  min_value=0, max_value=100),
        ],
        sql_guard_phrases=["IS NULL", "LEFT JOIN", "INTERVAL"],
        sql_forbidden_phrases=["COUNT(DISTINCT curr.user_id) AS churned"],
        # churned_users must never exceed base_users — structural invariant, always true.
        result_invariants=[("churned", "<=", "base")],
        denominator_col="base",
        numerator_col="churned",
        rate_col="rate",
        # expected_rate_range removed: a legitimate spike (outage, price change)
        # can exceed any calibrated range. The structural invariant above already
        # catches broken SQL. Stage 3 LLM handles implausibility in context.
    ),

    "retention_rate": MetricContract(
        metric_id="retention_rate",
        keywords=["retention", "retained", "retention rate", "d+1", "d+7",
                  "d+30", "cohort retention", "user retention"],
        required_col_roles=[
            ColumnContract(name_pattern="cohort",   role="count", must_be_positive=True),
            ColumnContract(name_pattern="retained", role="count", min_value=0),
            ColumnContract(name_pattern="retention", role="rate",  min_value=0, max_value=100),
        ],
        sql_guard_phrases=["DATE_TRUNC", "LEFT JOIN", "signup"],
        result_invariants=[("retained", "<=", "cohort")],
        denominator_col="cohort",
        numerator_col="retained",
        rate_col="retention",

    ),

    "conversion_rate": MetricContract(
        metric_id="conversion_rate",
        keywords=["conversion", "convert", "funnel", "cvr", "conversion rate",
                  "step", "drop off", "drop-off"],
        required_col_roles=[
            ColumnContract(name_pattern="step",       role="dimension"),
            ColumnContract(name_pattern="users",      role="count", must_be_positive=True),
            ColumnContract(name_pattern="conversion", role="rate", min_value=0, max_value=100),
        ],
        sql_guard_phrases=["COUNT(DISTINCT", "action_type"],
        result_invariants=[],
        rate_col="conversion",
    ),

    "dau_mau": MetricContract(
        metric_id="dau_mau",
        keywords=["dau", "mau", "wau", "daily active", "monthly active",
                  "weekly active", "engagement ratio"],
        required_col_roles=[
            ColumnContract(name_pattern="active", role="count", must_be_positive=True),
            ColumnContract(name_pattern="date",   role="date"),
        ],
        sql_guard_phrases=["COUNT(DISTINCT", "DATE_TRUNC"],
        result_invariants=[],
        rate_col=None,
    ),

    "cohort_analysis": MetricContract(
        metric_id="cohort_analysis",
        keywords=["cohort", "cohort analysis", "month index", "birth cohort",
                  "signup cohort"],
        required_col_roles=[
            ColumnContract(name_pattern="cohort", role="date"),
            ColumnContract(name_pattern="month",  role="dimension"),
            ColumnContract(name_pattern="users",  role="count", must_be_positive=True),
        ],
        sql_guard_phrases=["DATE_TRUNC", "signup_date", "EXTRACT"],
        result_invariants=[],
    ),

    "revenue": MetricContract(
        metric_id="revenue",
        keywords=["revenue", "sales", "spend", "turnover", "gmv",
                  "gross merchandise"],
        required_col_roles=[
            ColumnContract(name_pattern="revenue", role="count", must_be_positive=True),
        ],
        sql_guard_phrases=["SUM(", "NULLIF"],
        result_invariants=[],
    ),

    "growth_rate": MetricContract(
        metric_id="growth_rate",
        keywords=["growth", "growth rate", "month over month", "mom",
                  "week over week", "wow", "yoy", "year over year"],
        required_col_roles=[
            ColumnContract(name_pattern="growth", role="rate"),
        ],
        sql_guard_phrases=["LAG(", "OVER ("],
        result_invariants=[],
        rate_col="growth",
        # expected_rate_range removed — see churn_rate comment above.
    ),
}

# Fix the None assignment above
METRIC_CONTRACTS["retention_rate"].required_col_roles = [
    ColumnContract(name_pattern="cohort",    role="count", must_be_positive=True),
    ColumnContract(name_pattern="retained",  role="count", min_value=0),
    ColumnContract(name_pattern="retention", role="rate",  min_value=0, max_value=100),
]


def resolve_contracts(user_question: str) -> list[MetricContract]:
    """
    Returns all MetricContracts whose keywords match the user question.
    Called by RAG node and Validator.
    No metric-specific if-statements — the keyword list in each contract
    does the routing.
    """
    q = user_question.lower()
    matched = []
    for contract in METRIC_CONTRACTS.values():
        if any(kw.lower() in q for kw in contract.keywords):
            matched.append(contract)
    return matched


def enforce_sql_guards(sql: str, contracts: list[MetricContract]) -> list[str]:
    """
    Checks that the SQL contains all structurally required phrases
    for each matched contract.
    Returns a list of violation messages (empty = pass).
    Purely structural — no metric-specific logic.
    """
    violations = []
    for contract in contracts:
        sql_upper = sql.upper()
        for phrase in contract.sql_guard_phrases:
            if phrase.upper() not in sql_upper:
                violations.append(
                    f"[{contract.metric_id}] SQL is missing required structural "
                    f"element: '{phrase}'. This element is required for a correct "
                    f"{contract.metric_id} calculation."
                )
        for phrase in contract.sql_forbidden_phrases:
            if phrase.upper() in sql_upper:
                violations.append(
                    f"[{contract.metric_id}] SQL contains a known-incorrect "
                    f"pattern: '{phrase}'. This produces wrong results for "
                    f"{contract.metric_id}."
                )
    return violations


def enforce_result_invariants(
    df: pd.DataFrame,
    contracts: list[MetricContract]
) -> list[str]:
    """
    Checks DataFrame result against all applicable contract invariants.
    Returns list of violation messages (empty = pass).
    """
    violations = []
    ops = {
        "<=": lambda a, b: a <= b,
        ">=": lambda a, b: a >= b,
        "<":  lambda a, b: a < b,
        ">":  lambda a, b: a > b,
        "==": lambda a, b: a == b,
    }

    for contract in contracts:
        # 1. Logic Invariants
        for (pat_a, op, pat_b) in contract.result_invariants:
            col_a = _find_col(df, pat_a)
            col_b = _find_col(df, pat_b)
            if col_a is None or col_b is None:
                continue
            
            fn = ops.get(op)
            if fn is None:
                continue

            # CRITICAL: Only perform math on numeric columns
            if pd.api.types.is_numeric_dtype(df[col_a]) and pd.api.types.is_numeric_dtype(df[col_b]):
                bad_rows = df[~fn(df[col_a], df[col_b])]
                if not bad_rows.empty:
                    max_a, max_b = df[col_a].max(), df[col_b].max()
                    violations.append(
                        f"[{contract.metric_id}] Invariant violated: '{col_a}' {op} '{col_b}' "
                        f"fails in {len(bad_rows)} rows. Max {col_a}={max_a}, Max {col_b}={max_b}."
                    )

    return violations


def _find_col(df: pd.DataFrame, pattern: str) -> Optional[str]:
    """Find first column whose lowercase name contains the pattern."""
    pattern_lower = pattern.lower()
    for col in df.columns:
        if pattern_lower in col.lower():
            return col
    return None


def build_contract_prompt_block(contracts: list[MetricContract]) -> str:
    """
    Builds a structured prompt block summarising all active contracts.
    Injected into the SQL generator and corrector prompts.
    This replaces ad-hoc business rule strings with contract-driven constraints.
    """
    if not contracts:
        return ""
    lines = ["METRIC CONTRACT ENFORCEMENT (read before writing SQL):"]
    for c in contracts:
        lines.append(f"\n[CONTRACT: {c.metric_id.upper()}]")
        if c.sql_guard_phrases:
            lines.append(f"  REQUIRED SQL elements: {', '.join(c.sql_guard_phrases)}")
        if c.sql_forbidden_phrases:
            lines.append(f"  FORBIDDEN SQL patterns: {', '.join(c.sql_forbidden_phrases)}")
        if c.result_invariants:
            for (a, op, b) in c.result_invariants:
                lines.append(f"  INVARIANT: column matching '{a}' MUST be {op} column matching '{b}' in every row")
        if c.denominator_col:
            lines.append(f"  DENOMINATOR = column matching '{c.denominator_col}' (previous period population)")
        if c.numerator_col:
            lines.append(f"  NUMERATOR   = column matching '{c.numerator_col}' (events / users satisfying condition)")
    return "\n".join(lines)