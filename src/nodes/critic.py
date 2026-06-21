# src/nodes/critic.py
# ==========================================
# AGENT NODE E — CRITIC (EXECUTE ONLY)
# AGENT NODE G — SUMMARISER
# ==========================================
# Critic:    Execute SQL → DataFrame or error. Nothing else.
#            Summarisation only happens AFTER the validator approves.
#
# Summariser: Called by orchestrator only after logic_check_passed = True.
#             Never summarises data the validator rejected.
#
# Input  (critic_node):    generated_sql, should_export, user_question
# Output (critic_node):    query_result, validation_passed, attempt_history
#
# Input  (summarise_node): query_result, user_question, should_export,
#                          generated_sql
# Output (summarise_node): final_summary

import re
import pandas as pd

from src.infra import llm, engine, FAST_MODEL
from src.state import AgentState

FORBIDDEN_KEYWORDS = ["drop", "delete", "truncate", "update", "insert", "alter"]


def _clean_sql(raw: str) -> str:
    return re.sub(r"```[a-zA-Z]*", "", raw).replace("```", "").strip()


# ── NODE E: CRITIC ────────────────────────────────────────────────────────────

def critic_node(state: AgentState) -> dict:
    """
    Executes SQL and returns raw DataFrame or error string.
    Does NOT summarise — waits for validator approval first.
    """
    sql = _clean_sql(state["generated_sql"])

    if any(kw in sql.lower() for kw in FORBIDDEN_KEYWORDS):
        error = "Blocked: query contains a forbidden write operation."
        print(f"[Critic] ❌ {error}")
        return {
            "query_result":      error,
            "validation_passed": False,
            "attempt_history":   state["attempt_history"] + [{"sql": sql, "error": error}]
        }

    try:
        df = pd.read_sql_query(sql, engine)
        print(f"[Critic] ✅ SQL executed. Shape: {df.shape}")
        return {
            "query_result":      df,
            "validation_passed": True,
        }
    except Exception as e:
        error = f"Error executing query: {e}"
        print(f"[Critic] ❌ {error}")
        return {
            "query_result":      error,
            "validation_passed": False,
            "attempt_history":   state["attempt_history"] + [{"sql": sql, "error": error}]
        }


# ── NODE G: SUMMARISER ────────────────────────────────────────────────────────

def summarise_node(state: AgentState) -> dict:
    """
    Generates natural language summary ONLY after validator approval.
    Also handles Excel export — only ever exports validated data.
    """
    df = state["query_result"]

    if state["should_export"]:
        _export_to_excel(df, state)
    else:
        print("[Summariser] 📊 Inline result — no Excel export.")

    summary = _call_summariser(state["user_question"], df, state["should_export"])
    return {"final_summary": summary}


def _export_to_excel(df: pd.DataFrame, state: AgentState):
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            if df[col].dt.tz is not None:
                df[col] = df[col].dt.strftime("%Y-%m-%d")

    path = "agent_dashboard_data.xlsx"
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Dashboard_Data")
        pd.DataFrame({
            "Audit Parameter": ["Question", "SQL"],
            "Value":           [state["user_question"], state["generated_sql"]]
        }).to_excel(writer, index=False, sheet_name="SQL_QA_Audit")
    print(f"[Summariser] 📁 Exported to {path}")


def _call_summariser(question: str, df: pd.DataFrame, exported: bool) -> str:
    export_note = (
        "Mention the full table has been exported to an Excel file."
        if exported else
        "Do NOT mention any Excel file — none was generated."
    )
    prompt = f"""
You are a data analyst. The user asked: "{question}"

The query returned {len(df)} rows. First 10 rows:
{df.head(10).to_string(index=False)}

Write a concise, professional summary (max 5 sentences).
Include the key numbers from the result.
{export_note}
"""
    response = llm.chat.completions.create(
        model=FAST_MODEL,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content
