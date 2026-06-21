# src/state.py
# ==========================================
# SHARED AGENT STATE
# ==========================================
# AgentState is a TypedDict that flows through every node.
# Each node reads from it and returns a partial dict that the
# orchestrator merges back in — the same pattern LangGraph uses
# for StateGraph nodes.
#
# Fields are ordered by pipeline stage so the file reads top-to-bottom
# in the same order the agent executes.

from typing import Any, Dict, List, Optional
from typing_extensions import TypedDict


class QueryPlan(TypedDict):
    analytical_goal:      str
    metrics_required:     List[str]
    tables_needed:        List[str]
    time_dimension:       str          # monthly | daily | weekly | total | none
    filters:              List[str]
    output_format:        str          # trend | aggregate
    ambiguities_resolved: List[str]
    rewritten_question:   str


class AgentState(TypedDict):
    # ── Input ─────────────────────────────────────────────────────────────────
    user_question:      str

    # ── Node A: RAG ───────────────────────────────────────────────────────────
    business_rules:     str

    # ── Node B: Query Planner ─────────────────────────────────────────────────
    query_plan:         Optional[QueryPlan]

    # ── Node C: Data Profiler ─────────────────────────────────────────────────
    column_profiles:    str

    # ── Node D: SQL Generator ─────────────────────────────────────────────────
    generated_sql:      str
    should_export:      bool

    # ── Node E: Critic ────────────────────────────────────────────────────────
    query_result:       Any                   # DataFrame on success, str on error
    validation_passed:  bool                  # True = SQL executed without error
    attempt_history:    List[Dict[str, str]]  # {sql, error} pairs from failed runs

    # ── Node F: Validator ─────────────────────────────────────────────────────
    logic_check_passed: bool   # True = result passed all quality checks
    logic_feedback:     str    # validator's precise correction instruction

    # ── Node G: Summariser ────────────────────────────────────────────────────
    final_summary:      str
