# src/orchestrator.py
# ==========================================
# ORCHESTRATOR — AGENT LOOP CONTROLLER
# ==========================================
# Full pipeline:
#
#   Node A — RAG            fetch business rules from glossary
#   Node B — Query Planner  decompose + rewrite the question
#   Node C — Data Profiler  SELECT DISTINCT on categorical columns
#   Node D — SQL Generator  write SQL (or Corrector on retry)
#   Node E — Critic         execute SQL → DataFrame or error
#              │
#              ├── SQL execution error → SQL Corrector → back to Critic
#              │
#   Node F — Validator      check DataFrame for metric logic errors
#              │
#              ├── logic error → SQL Corrector (with logic_feedback) → back to Critic
#              │
#   Node G — Summariser     generate natural language answer
#
# Retry budget — two separate counters:
#
#   MAX_SQL_RETRIES    (default 3): retries for SQL syntax/execution errors.
#                      Postgres catches these fast, LLM not involved in check.
#
#   MAX_LOGIC_RETRIES  (default 2): retries for semantic/logic validation failures.
#                      Each retry calls STRONG_MODEL in the corrector — costs more.
#                      Kept low to prevent bill spikes.
#
#   Total worst-case LLM calls per question:
#     1 (generator) + MAX_SQL_RETRIES (corrector) + MAX_LOGIC_RETRIES (corrector)
#     + (MAX_SQL_RETRIES + MAX_LOGIC_RETRIES) × 1 (validator) + 1 (summariser)
#     = bounded, predictable, auditable.

from src.state import AgentState
from src.nodes.rag           import rag_node
from src.nodes.query_planner import plan_query, IrrelevantQuestionError
from src.nodes.data_profiler import profile_data
from src.nodes.sql_generator import sql_generator_node, sql_corrector_node
from src.nodes.critic        import critic_node, summarise_node
from src.nodes.validator     import validate_result


# ── RETRY BUDGETS ──────────────────────────────────────────────────────────────
MAX_SQL_RETRIES   = 3   # retries for SQL execution errors (cheap to retry)
MAX_LOGIC_RETRIES = 2   # retries for validator rejections (calls STRONG_MODEL)


def ask_agent(question: str) -> str:
    print(f"\n{'='*55}")
    print(f"Question: {question}")
    print(f"{'='*55}\n")

    state: AgentState = {
        "user_question":      question,
        "business_rules":     "",
        "query_plan":         None,
        "column_profiles":    "",
        "generated_sql":      "",
        "should_export":      False,
        "query_result":       None,
        "attempt_history":    [],
        "validation_passed":  False,
        "logic_check_passed": False,
        "logic_feedback":     "",
        "final_summary":      ""
    }

    # ── Node A: RAG s───────────────────────────────────────────────────────────
    state.update(rag_node(state))    

    # ── Node B: Query Planner ─────────────────────────────────────────────────
    try:
        state.update(plan_query(state))
    except IrrelevantQuestionError:
        print("\nSorry unknown request\n")
        return "Sorry unknown request"

    # ── Node C: Data Profiler ─────────────────────────────────────────────────
    state.update(profile_data(state))

    # ── Node D: Initial SQL Generation ────────────────────────────────────────
    state.update(sql_generator_node(state))

    # ── Retry loop: SQL errors + logic errors have separate budgets ────────────
    sql_retries   = 0
    logic_retries = 0

    while True:
        # ── Node E: Critic — execute SQL ──────────────────────────────────────
        state.update(critic_node(state))

        if not state["validation_passed"]:
            # SQL execution error
            sql_retries += 1
            if sql_retries > MAX_SQL_RETRIES:
                print(f"[Orchestrator] Max SQL retries ({MAX_SQL_RETRIES}) reached.")
                return "Sorry — the agent could not fix the SQL error after maximum attempts."

            print(f"\n--- SQL Retry {sql_retries}/{MAX_SQL_RETRIES} ---")
            state.update(sql_corrector_node(state))
            continue

        # ── Node F: Validator — check logic ───────────────────────────────────
        state.update(validate_result(state))

        if state["logic_check_passed"]:
            break  # Both execution and logic passed

        # Logic check failed — escalate to corrector with validator feedback
        logic_retries += 1
        if logic_retries > MAX_LOGIC_RETRIES:
            print(f"[Orchestrator] Max logic retries ({MAX_LOGIC_RETRIES}) reached.")
            return "Sorry — the agent produced a result that failed quality checks after maximum attempts."

        print(f"\n--- Logic Retry {logic_retries}/{MAX_LOGIC_RETRIES} ---")
        print(f"[Orchestrator] Validator rejected result. Feeding back to corrector.")

        # Inject logic_feedback into attempt_history so corrector sees it
        state["attempt_history"].append({
            "sql":   state["generated_sql"],
            "error": f"[Logic Validator Feedback]: {state['logic_feedback']}"
        })
        state["logic_feedback"]   = ""
        state["validation_passed"] = False  # reset so corrector runs next
        state.update(sql_corrector_node(state))

    # ── Node G: Summariser ────────────────────────────────────────────────────
    state.update(summarise_node(state))

    print(f"\n--- Final Answer ---\n{state['final_summary']}\n")
    return state["final_summary"]
