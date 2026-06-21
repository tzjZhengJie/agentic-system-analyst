# src/nodes/query_planner.py
# ==========================================
# AGENT NODE B — QUERY PLANNER
# ==========================================
import json
from sqlalchemy import inspect as sql_inspect

from src.infra import llm, engine, FAST_MODEL
from src.state import AgentState


class IrrelevantQuestionError(Exception):
    """Raised when the question has nothing to do with the database."""
    pass


def _load_schema_summary() -> str:
    try:
        inspector = sql_inspect(engine)
        tables = inspector.get_table_names(schema="public")
        lines = []
        for table in tables:
            cols = [c["name"] for c in inspector.get_columns(table, schema="public")]
            lines.append(f"{table}: {', '.join(cols)}")
        return "\n".join(lines)
    except Exception as e:
        return f"Schema unavailable: {e}"


def plan_query(state: AgentState) -> dict:
    schema_summary = _load_schema_summary()

    prompt = f"""
You are a senior data product manager. Read the user's analytics question
and produce a structured analytical brief for a SQL-writing agent.

AVAILABLE DATABASE TABLES AND COLUMNS:
{schema_summary}

BUSINESS RULES ALREADY RETRIEVED:
{state["business_rules"]}

USER QUESTION:
{state["user_question"]}

FIRST — decide if this question is answerable from the database above.
A question is IRRELEVANT if it:
- Is a greeting, small talk, or social message (hi, hello, how are you, thanks)
- Asks about something completely unrelated to e-commerce, users, revenue, or activity
- Cannot be answered by any combination of the tables and columns above

Return ONLY a valid JSON object with these exact keys:
{{
  "is_relevant": true or false,
  "analytical_goal": "one sentence describing what we are measuring, or empty string if irrelevant",
  "metrics_required": ["list", "of", "exact", "metrics"],
  "tables_needed": ["only tables that exist in the schema above"],
  "time_dimension": "monthly | daily | weekly | total | none",
  "filters": ["any WHERE conditions implied"],
  "output_format": "trend | aggregate",
  "ambiguities_resolved": ["each ambiguous phrase and how you resolved it"],
  "rewritten_question": "a single clean, precise analytical question, or empty string if irrelevant"
}}

RULES:
- Only reference tables and columns from AVAILABLE DATABASE TABLES AND COLUMNS.
- output_format must be 'trend' if the question asks for time-series or month-by-month.
- output_format must be 'aggregate' if the question asks for a single summary number.
- Do not wrap the response in markdown fences.
"""

    response = llm.chat.completions.create(
        model=FAST_MODEL,
        response_format={"type": "json_object"},
        messages=[{"role": "user", "content": prompt}]
    )

    plan = json.loads(response.choices[0].message.content)

    if not plan.get("is_relevant", True):
        raise IrrelevantQuestionError("Question is not related to the database.")

    print("\n[Query Planner] Analytical brief:")
    print(f"  Goal:      {plan.get('analytical_goal')}")
    print(f"  Metrics:   {plan.get('metrics_required')}")
    print(f"  Tables:    {plan.get('tables_needed')}")
    print(f"  Format:    {plan.get('output_format')}")
    print(f"  Rewritten: {plan.get('rewritten_question')}\n")

    return {"query_plan": plan}