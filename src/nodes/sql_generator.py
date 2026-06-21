# src/nodes/sql_generator.py
# ==========================================
# AGENT NODE D — SQL GENERATOR + CORRECTOR
# ==========================================
# Reads from FOUR grounding sources before writing any SQL:
#   1. query_plan        — what to measure (from Query Planner)
#   2. business_rules    — how to measure it (from RAG)
#   3. column_profiles   — what values actually exist (from Data Profiler)
#   4. metric_contracts  — structural invariants the SQL MUST satisfy
#
# Two separate nodes:
#   sql_generator_node  — first-pass generation (fast model)
#   sql_corrector_node  — retry correction (strong model, only on failure)
#
# Input state fields read:  query_plan, business_rules, column_profiles,
#                           generated_sql, attempt_history, user_question
# Output state fields set:  generated_sql, should_export

import json
import re
from sqlalchemy import inspect as sql_inspect

from src.infra import llm, engine, FAST_MODEL, STRONG_MODEL
from src.state import AgentState
from src.metric_contracts import (
    resolve_contracts,
    enforce_sql_guards,
    build_contract_prompt_block,
)


# ── SCHEMA LOADER ─────────────────────────────────────────────────────────────
# Schema is loaded fresh per call here (not cached globally) so that the
# infra singleton's engine is used. In production: cache in Redis with TTL.

def _load_schema() -> str:
    try:
        inspector = sql_inspect(engine)
        tables = inspector.get_table_names(schema="public")
        if not tables:
            return "[Schema Warning] No tables found in public schema."
        lines = []
        for table in tables:
            lines.append(f"Table: {table}")
            for col in inspector.get_columns(table, schema="public"):
                lines.append(f"  - {col['name']} ({str(col['type'])})")
            lines.append("")
        return "\n".join(lines)
    except Exception as e:
        raise RuntimeError(
            f"[Schema] Cannot connect to database.\n"
            f"Check DATABASE_URL in .env\nError: {e}"
        )


# ── UTILITIES ──────────────────────────────────────────────────────────────────

def _clean_sql(raw: str) -> str:
    return re.sub(r"```[a-zA-Z]*", "", raw).replace("```", "").strip()


def _sanitise_sql(sql: str) -> str:
    """
    Deterministic post-generation fix: FLOAT → NUMERIC.
    PostgreSQL's ROUND() requires NUMERIC, not FLOAT.
    No LLM can override this — it runs unconditionally.
    """
    for bad in ("::FLOAT8", "::FLOAT", "::float8", "::float"):
        sql = sql.replace(bad, "::NUMERIC")
    sql = re.sub(
        r"CAST\s*\((.+?)\s+AS\s+(?:FLOAT8?|DOUBLE\s+PRECISION)\)",
        r"\1::NUMERIC",
        sql,
        flags=re.IGNORECASE
    )
    return sql


def _format_plan(plan: dict) -> str:
    return f"""
Analytical Goal:      {plan.get('analytical_goal')}
Metrics Required:     {', '.join(plan.get('metrics_required', []))}
Tables Needed:        {', '.join(plan.get('tables_needed', []))}
Time Dimension:       {plan.get('time_dimension')}
Output Format:        {plan.get('output_format')}
Filters/Conditions:   {'; '.join(plan.get('filters', []))}
Ambiguities Resolved:
{chr(10).join(f"  - {a}" for a in plan.get('ambiguities_resolved', []))}
Rewritten Question:   {plan.get('rewritten_question')}
""".strip()


# ── NODE: FIRST-PASS SQL GENERATOR ────────────────────────────────────────────

def sql_generator_node(state: AgentState) -> dict:
    """
    First-pass SQL generation with contract-aware prompting.
    Uses FAST_MODEL — only the corrector escalates to STRONG_MODEL.
    """
    schema_map      = _load_schema()
    plan            = state.get("query_plan") or {}
    plan_block      = _format_plan(plan) if plan else "No query plan available."
    column_profiles = state.get("column_profiles", "No column profiles available.")
    output_format   = plan.get("output_format", "aggregate")

    contracts      = resolve_contracts(state["user_question"])
    contract_block = build_contract_prompt_block(contracts)

    prompt = f"""
You are an expert data analyst writing PostgreSQL queries.

STRICT SCHEMA (ONLY these tables and columns exist — do not invent others):
{schema_map}

ACTUAL DATA VALUES — RETRIEVED FROM LIVE DATABASE:
{column_profiles}

CRITICAL: ONLY use the exact values shown above in WHERE / CASE / IN clauses.
If a value is not listed, do not guess it — write a SQL comment explaining why.

ANALYTICAL BRIEF FROM QUERY PLANNER:
{plan_block}

MANDATORY BUSINESS RULES:
{state["business_rules"]}

{contract_block}

HARD RULES:
1. PostgreSQL syntax only.
2. ONLY use column names from STRICT SCHEMA above.
3. ONLY use values confirmed in ACTUAL DATA VALUES above.
4. LIMIT to 1000 rows unless the question asks for a total or aggregate.
5. Use NULLIF(denominator, 0) to prevent division-by-zero.
6. DO NOT use CURRENT_DATE or NOW() — the dataset has a fixed date range.
7. NEVER use CAST(x AS FLOAT). Use x::NUMERIC for division.
8. Always use: ROUND((numerator::NUMERIC / NULLIF(denominator, 0)) * 100, 2)
9. If output_format is 'trend', return one row per period across FULL history.
10. If MANDATORY BUSINESS RULES includes a Validated SQL Pattern, use it EXACTLY.
11. If a METRIC CONTRACT says an element is REQUIRED, it must appear in your SQL.
12. If a METRIC CONTRACT says a pattern is FORBIDDEN, it must NOT appear.

Return a JSON object with exactly two keys:
- "sql": the raw SQL string (no markdown fences)
- "export_excel": true if output_format is 'trend', otherwise false

User question (for reference only — the brief and contracts take priority):
{state["user_question"]}
"""
    response = llm.chat.completions.create(
        model=FAST_MODEL,
        response_format={"type": "json_object"},
        messages=[{"role": "user", "content": prompt}]
    )
    result = json.loads(response.choices[0].message.content)

    raw_sql  = _clean_sql(result.get("sql", ""))
    safe_sql = _sanitise_sql(raw_sql)

    if raw_sql != safe_sql:
        print("[SQL Generator] ⚠️  Sanitiser corrected type cast (::FLOAT → ::NUMERIC).")

    # Pre-flight contract check — violations are logged, not blocking here.
    # The Validator will catch result-level issues and trigger correction.
    violations = enforce_sql_guards(safe_sql, resolve_contracts(state["user_question"]))
    for v in violations:
        print(f"[SQL Generator] ⚠️  Contract pre-flight: {v}")

    should_export = result.get("export_excel", output_format == "trend")
    print(f"[SQL Generator] Query written. Export to Excel: {should_export}")
    return {"generated_sql": safe_sql, "should_export": should_export}


# ── NODE: SQL CORRECTOR ────────────────────────────────────────────────────────

def sql_corrector_node(state: AgentState) -> dict:
    """
    Retry-pass SQL correction.
    Escalates to STRONG_MODEL — called only after cheap checks fail.
    Receives both execution errors AND validator logic_feedback in history.
    """
    schema_map      = _load_schema()
    plan            = state.get("query_plan") or {}
    plan_block      = _format_plan(plan) if plan else "No query plan available."
    column_profiles = state.get("column_profiles", "No column profiles available.")

    contracts      = resolve_contracts(state["user_question"])
    contract_block = build_contract_prompt_block(contracts)

    history_str = ""
    for idx, h in enumerate(state["attempt_history"]):
        history_str += (
            f"\n[Attempt {idx+1} — Failed SQL]:\n{h['sql']}\n"
            f"[Attempt {idx+1} — Error/Feedback]:\n{h['error']}\n"
        )

    last_error = (
        state["attempt_history"][-1]["error"]
        if state["attempt_history"] else "Unknown"
    )

    prompt = f"""
You generated a PostgreSQL query that failed validation. Fix it.

STRICT SCHEMA (ONLY these columns exist):
{schema_map}

ACTUAL DATA VALUES (ONLY use these in WHERE/CASE):
{column_profiles}

ORIGINAL ANALYTICAL BRIEF:
{plan_block}

{contract_block}

FAILED ATTEMPT HISTORY (do NOT repeat these mistakes):
{history_str}

Most recent broken SQL:
{state["generated_sql"]}

Most recent error/feedback:
{last_error}

FIX RULES:
1. Only use columns from STRICT SCHEMA.
2. Only use values confirmed in ACTUAL DATA VALUES.
3. Do not repeat mistakes from FAILED ATTEMPT HISTORY.
4. Stay true to the Analytical Brief — fix the error, not the goal.
5. If the error says a REQUIRED SQL element is missing, add it.
6. If the error says a FORBIDDEN SQL pattern exists, remove it.
7. If an INVARIANT is violated (e.g. churned > base), the anti-join direction
   is wrong. Use IS NULL on the right side of LEFT JOIN to identify absences.
8. If a COUNT column is all-zero, the JOIN is wrong — do NOT add CAST or ::NUMERIC.
   Re-examine the join keys and INTERVAL arithmetic between CTEs.
9. If output_format is 'trend' and only one row returned, group by the period
   column for all historical periods.
10. If MANDATORY BUSINESS RULES includes a Validated SQL Pattern, use it EXACTLY.

Return ONLY the corrected raw SQL string. No markdown, no explanation.
"""
    response = llm.chat.completions.create(
        model=STRONG_MODEL,
        messages=[{"role": "user", "content": prompt}]
    )

    raw_sql  = _clean_sql(response.choices[0].message.content)
    safe_sql = _sanitise_sql(raw_sql)

    if raw_sql != safe_sql:
        print("[SQL Corrector] ⚠️  Sanitiser corrected type cast (::FLOAT → ::NUMERIC).")

    print("[SQL Corrector] Rewritten SQL ready.")
    return {"generated_sql": safe_sql}
