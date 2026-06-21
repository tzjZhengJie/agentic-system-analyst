# src/nodes/data_profiler.py
# ==========================================
# AGENT NODE C — DATA PROFILER
# ==========================================
# Dynamically profiles categorical columns from the live database
# before any SQL is written. Injects REAL column values into the
# SQL Generator prompt so it cannot hallucinate values that don't exist.
#
# Three-layer filtering:
#   Layer 1 — Type filter (cheap, no DB query): TEXT/VARCHAR/CHAR only
#   Layer 2 — Name heuristics (cheap): skips IDs, dates, free-text fields
#   Layer 3 — Cardinality gate: skips high-cardinality columns (> 1% distinct)
#
# This is how Great Expectations and dbt's profiler work.
#
# Input state fields read:  query_plan
# Output state fields set:  column_profiles

from sqlalchemy import inspect as sql_inspect, text

from src.infra import engine
from src.state import AgentState

# ── TUNING CONSTANTS ──────────────────────────────────────────────────────────
MAX_DISTINCT_VALUES = 50
CARDINALITY_CEILING = 0.01   # skip column if distinct/total > 1%
MIN_ROW_THRESHOLD   = 10

_ID_SUFFIXES       = ("_id", "_key", "_uuid", "_pk", "_fk")
_DATE_PATTERNS     = ("date", "time", "at", "created", "updated", "modified",
                      "timestamp", "expiry", "expires", "dob")
_FREETEXT_PATTERNS = ("name", "email", "address", "description", "notes",
                      "comment", "url", "uri", "ip", "token", "hash",
                      "password", "secret", "code", "slug", "bio", "text",
                      "message", "subject", "title", "label", "tag")


def _is_profileable(col_name: str, col_type: str) -> bool:
    col_type_upper = col_type.upper()
    col_lower      = col_name.lower()
    if not any(t in col_type_upper for t in ("TEXT", "VARCHAR", "CHAR", "ENUM")):
        return False
    if col_lower == "id" or col_lower.endswith(_ID_SUFFIXES):
        return False
    if any(p in col_lower for p in _DATE_PATTERNS):
        return False
    if any(p in col_lower for p in _FREETEXT_PATTERNS):
        return False
    return True


def _get_row_count(table: str) -> int:
    try:
        with engine.connect() as conn:
            return conn.execute(text(f'SELECT COUNT(*) FROM "{table}"')).fetchone()[0]
    except Exception:
        return -1


def _get_cardinality_ratio(table: str, column: str, row_count: int) -> float:
    if row_count <= 0:
        return 1.0
    try:
        with engine.connect() as conn:
            count = conn.execute(
                text(f'SELECT COUNT(DISTINCT "{column}") FROM "{table}"')
            ).fetchone()[0]
        return count / row_count
    except Exception:
        return 1.0


def _fetch_distinct_values(table: str, column: str) -> list:
    try:
        q = text(
            f'SELECT DISTINCT "{column}" FROM "{table}" '
            f'WHERE "{column}" IS NOT NULL '
            f'ORDER BY "{column}" '
            f'LIMIT {MAX_DISTINCT_VALUES}'
        )
        with engine.connect() as conn:
            return [row[0] for row in conn.execute(q).fetchall()]
    except Exception as e:
        print(f"[Profiler] Could not fetch values for '{table}.{column}' — {e}")
        return []


def _discover_profileable_columns(tables: list) -> dict:
    inspector = sql_inspect(engine)
    result = {}

    for table in tables:
        try:
            schema_cols = inspector.get_columns(table, schema="public")
        except Exception as e:
            print(f"[Profiler] Could not inspect '{table}' — {e}")
            continue

        row_count = _get_row_count(table)
        if row_count < MIN_ROW_THRESHOLD:
            continue

        profileable = []
        for col in schema_cols:
            col_name = col["name"]
            col_type = str(col["type"])
            if not _is_profileable(col_name, col_type):
                continue
            ratio = _get_cardinality_ratio(table, col_name, row_count)
            if ratio > CARDINALITY_CEILING:
                continue
            profileable.append(col_name)

        if profileable:
            result[table] = profileable

    return result


def _run_profiling(profileable_cols: dict) -> str:
    profile_lines = []
    total_queries = 0

    for table, columns in profileable_cols.items():
        row_count = _get_row_count(table)
        profile_lines.append(f"Table: {table}  ({row_count:,} total rows)")
        for col in columns:
            values = _fetch_distinct_values(table, col)
            total_queries += 1
            if values:
                values_str = ", ".join(str(v) for v in values)
                profile_lines.append(f"  {col}: {values_str}")
        profile_lines.append("")

    print(f"[Profiler] Done — {total_queries} SELECT DISTINCT queries across {len(profileable_cols)} table(s).")
    return (
        "VERIFIED COLUMN VALUES (live database — "
        "use ONLY these exact strings in WHERE / CASE / IN clauses):\n\n"
        + "\n".join(profile_lines)
    )


def profile_data(state: AgentState) -> dict:
    plan          = state.get("query_plan") or {}
    tables_needed = plan.get("tables_needed", [])

    if not tables_needed:
        print("[Profiler] No tables in query plan — skipping.")
        return {"column_profiles": "No tables identified. Use values from the user question only."}

    print(f"[Profiler] Scanning {len(tables_needed)} table(s): {tables_needed}")
    profileable_cols = _discover_profileable_columns(tables_needed)

    if not profileable_cols:
        print("[Profiler] No categorical columns found after filtering.")
        return {"column_profiles": "No low-cardinality categorical columns found."}

    return {"column_profiles": _run_profiling(profileable_cols)}
