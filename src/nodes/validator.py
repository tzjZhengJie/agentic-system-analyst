# src/nodes/validator.py
# ==========================================
# AGENT NODE F — CONTRACT-DRIVEN VALIDATOR
# ==========================================
# Three-stage pipeline ordered by cost (cheapest first):
#
#   Stage 1 — MetricContract invariants (free, deterministic)
#              Checks DataFrame against structural rules in metric_contracts.py.
#              Zero metric-specific if-blocks — all logic lives in contracts.
#
#   Stage 2 — Rule-based heuristics (free, no LLM)
#              Generic sanity checks: all-zero counts, negative counts,
#              integer division zeros, uniform results.
#
#   Stage 3 — LLM semantic check (only if Stages 1 & 2 pass)
#              Final reasonableness audit for things structural checks miss.
#
# LLM is the last resort — not the first line of defence.
# This cost ordering is the same pattern as dbt's test suite.
#
# Input state fields read:  query_result, user_question, generated_sql, query_plan
# Output state fields set:  logic_check_passed, logic_feedback

import json
import pandas as pd

from src.infra import llm, FAST_MODEL
from src.state import AgentState
from src.metric_contracts import (
    resolve_contracts,
    enforce_result_invariants,
    MetricContract,
)

_COUNT_KEYWORDS = ("users", "count", "orders", "sessions", "events",
                   "churned", "active", "retained", "total", "signups",
                   "cohort", "base", "visits", "impressions")

_RATE_KEYWORDS  = ("rate", "ratio", "pct", "percent", "conversion",
                   "retention", "churn_rate", "ctr", "cvr", "roas",
                   "percentage", "share", "proportion")


def _is_count_column(col: str) -> bool:
    return any(kw in col.lower() for kw in _COUNT_KEYWORDS)


def _is_rate_column(col: str) -> bool:
    return any(kw in col.lower() for kw in _RATE_KEYWORDS)


# ── STAGE 1: CONTRACT ENFORCEMENT ─────────────────────────────────────────────

def _contract_checks(df: pd.DataFrame, contracts: list) -> tuple:
    if not contracts:
        return True, ""
    violations = enforce_result_invariants(df, contracts)
    if violations:
        return False, " | ".join(violations)
    return True, ""


# ── STAGE 2: RULE-BASED CHECKS ────────────────────────────────────────────────

def _rule_based_checks(df: pd.DataFrame) -> tuple:
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    if not numeric_cols:
        return True, ""

    for col in numeric_cols:
        series = df[col].dropna()
        if series.empty:
            continue

        if (series == 0).all() and _is_count_column(col):
            return False, (
                f"Column '{col}' (a COUNT column) contains only zeros. "
                f"This is a broken JOIN or anti-join pattern — do NOT add CAST or ::NUMERIC. "
                f"Verify the LEFT JOIN condition and DATE_TRUNC interval arithmetic."
            )

        if (series == 0).all() and _is_rate_column(col):
            count_has_data = any(
                df[c].dropna().sum() > 0
                for c in numeric_cols if _is_count_column(c)
            )
            if count_has_data:
                return False, (
                    f"Column '{col}' (a rate column) is all zeros but COUNT columns "
                    f"have non-zero values. This is integer division truncation. "
                    f"Fix: ROUND(numerator::NUMERIC / NULLIF(denominator, 0) * 100, 2)"
                )

        if _is_count_column(col) and series.min() < 0:
            return False, (
                f"Column '{col}' contains negative values (min: {series.min()}). "
                f"COUNT columns must be non-negative."
            )

    if len(df) > 5:
        all_uniform = all(
            df[c].nunique() <= 1
            for c in numeric_cols if df[c].dropna().nunique() > 0
        )
        if all_uniform:
            return False, (
                f"All numeric columns have only one distinct value across {len(df)} rows. "
                f"This signals a broken GROUP BY or cartesian join."
            )

    return True, ""


# ── STAGE 3: LLM SEMANTIC CHECK ───────────────────────────────────────────────

def _llm_semantic_check(state: AgentState, df: pd.DataFrame, contracts: list) -> tuple:
    plan            = state.get("query_plan") or {}
    analytical_goal = plan.get("analytical_goal", "Not available")

    contract_context = ""
    for c in contracts:
        contract_context += (
            f"\nActive Contract: {c.metric_id}\n"
            f"  Denominator col: {c.denominator_col}\n"
            f"  Numerator col:   {c.numerator_col}\n"
        )

    try:
        sample_str = df.head(5).to_markdown(index=False)
    except Exception:
        sample_str = df.head(5).to_string(index=False)

    prompt = f"""
You are a Data Quality Auditor reviewing an analytics SQL query result.

USER QUESTION: {state["user_question"]}
ANALYTICAL GOAL: {analytical_goal}
SQL EXECUTED: {state["generated_sql"]}

DATA SAMPLE (first 5 rows):
{sample_str}

Total rows: {len(df)} | Columns: {list(df.columns)}

METRIC CONTRACT CONTEXT:
{contract_context}

CRITICAL DATASET CONTEXT — READ BEFORE CHECKING:
- This is a SYNTHETIC e-commerce dataset generated in 2025-2026.
- Date values from 2025 or 2026 are CORRECT and EXPECTED. Do NOT flag them as future dates.
- Do NOT use your assumption of "current date" to evaluate date columns — the dataset has its own fixed range.

ALREADY VERIFIED (do NOT re-check these):
- Rate/percentage column value ranges (contract engine verified)
- Structural invariants e.g. churned <= base (checked row-by-row)
- Negative count values (checked above)
- Whether date values look like "the future" (irrelevant for synthetic data)

Your job: catch ONLY these specific blockers:
1. Column names are completely wrong for the question (e.g. revenue columns for a churn query).
2. Time granularity is totally mismatched (daily rows when monthly was explicitly asked).
3. output_format is 'trend' but only 1 row returned — this means GROUP BY on the period column is missing.
4. An obvious logical impossibility that structural checks cannot catch.

If none of the above apply, return valid=true. Be conservative — only flag genuine blockers, not style preferences.

Return ONLY a JSON object:
{{
  "valid": true or false,
  "issues_found": ["list each blocking issue, empty if valid"],
  "feedback": "if invalid: one precise sentence telling the SQL corrector what to fix. If valid: empty string.",
  "severity": "blocking" or "warning" or "ok"
}}
"""
    response = llm.chat.completions.create(
        model=FAST_MODEL,
        response_format={"type": "json_object"},
        messages=[{"role": "user", "content": prompt}]
    )

    result   = json.loads(response.choices[0].message.content)
    valid    = result.get("valid", True)
    severity = result.get("severity", "ok")
    feedback = result.get("feedback", "")
    issues   = result.get("issues_found", [])

    if issues:
        print(f"[Validator] Issues found: {issues}")

    if severity == "warning":
        print(f"[Validator] ⚠️  Warning (non-blocking): {feedback}")
        return True, ""

    if not valid and severity == "blocking":
        return False, feedback

    return True, ""


# ── NODE ENTRY POINT ──────────────────────────────────────────────────────────

def validate_result(state: AgentState) -> dict:
    """
    Three-stage validation. Cheap stages run first; LLM only runs
    when deterministic checks pass.
    """
    df = state.get("query_result")
    if not isinstance(df, pd.DataFrame):
        print("[Validator] No DataFrame to validate — skipping.")
        return {"logic_check_passed": False, "logic_feedback": "No data returned."}

    print(f"[Validator] Checking result — {len(df)} rows, {len(df.columns)} columns.")

    contracts = resolve_contracts(state["user_question"])
    if contracts:
        print(f"[Validator] Active contracts: {[c.metric_id for c in contracts]}")

    # Stage 1: Contract invariants
    passed, feedback = _contract_checks(df, contracts)
    if not passed:
        print(f"[Validator] ❌ Contract check failed: {feedback}")
        return {"logic_check_passed": False, "logic_feedback": feedback}

    # Stage 2: Rule-based heuristics
    passed, feedback = _rule_based_checks(df)
    if not passed:
        print(f"[Validator] ❌ Rule check failed: {feedback}")
        return {"logic_check_passed": False, "logic_feedback": feedback}

    print("[Validator] ✅ Deterministic checks passed.")

    # Stage 3: LLM semantic check (only reached if 1 & 2 pass)
    passed, feedback = _llm_semantic_check(state, df, contracts)
    if not passed:
        print(f"[Validator] ❌ Semantic check failed: {feedback}")
        return {"logic_check_passed": False, "logic_feedback": feedback}

    print("[Validator] ✅ All checks passed.")
    return {"logic_check_passed": True, "logic_feedback": ""}
