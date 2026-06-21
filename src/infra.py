# src/infra.py
# ==========================================
# SINGLETON INFRASTRUCTURE
# ==========================================
# Single source of truth for shared clients.
#
# Problem solved: previously each node created its own OpenAI() and
# create_engine() at import time — 5 separate connection pools per run,
# DB URL duplicated in 5 places, no way to swap models or DBs without
# touching multiple files.
#
# Pattern: module-level singletons (the Python-idiomatic Singleton).
# Every node imports `llm` and `engine` from here — one object, one pool.
#
# To change the DB URL or model: edit this file only.

import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from sqlalchemy import create_engine as _create_engine

# Load .env from project root (two levels up from src/)
_root = Path(__file__).resolve().parent.parent
load_dotenv(_root / ".env")

# ── LLM CLIENT ────────────────────────────────────────────────────────────────
# One client, one connection pool, one API key location.
llm = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ── DATABASE ENGINE ───────────────────────────────────────────────────────────
# Connection string read once from env; falls back to the default local URL.
# pool_pre_ping=True drops stale connections automatically.
_db_url = os.getenv(
    "DATABASE_URL"
)
engine = _create_engine(_db_url, pool_pre_ping=True)

# ── MODEL NAMES ───────────────────────────────────────────────────────────────
# Change model in one place; all nodes pick it up automatically.
FAST_MODEL    = os.getenv("FAST_MODEL",    "gpt-4o-mini")   # planning, generation, validation
STRONG_MODEL  = os.getenv("STRONG_MODEL",  "gpt-4.1")       # SQL correction only
