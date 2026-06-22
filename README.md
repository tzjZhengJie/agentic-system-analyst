# Multi-agentic Analytics 

A natural language → SQL analytics agent built to learn multi-agent orchestration, semantic rule retrieval, and self-healing LLM pipelines.

Ask a plain-English question about an e-commerce dataset. The agent writes the SQL, validates it against metric contracts, retries on failure, and returns a summary or Excel export.

---

## Architecture

```
          User Question
               │
               ▼
┌─────────────────────────────┐
│  Node A: RAG Lookup         │  Keyword-matches question → business_glossary.json
│                             │  Injects metric formulas + join rules into SQL prompt
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│  Node B: Query Planner      │  Decomposes question → structured QueryPlan
│                             │  Resolves ambiguity before SQL is written
|                             |  Return unknown if user's prompt is gibberish or unclear
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│  Node C: Data Profiler      │  SELECT DISTINCT on low-cardinality columns
│                             │  Injects REAL values → prevents hallucinated WHERE clauses
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│  Node D: SQL Generator      │  Writes PostgreSQL from plan + rules + profiles + contracts
│                             │  Uses gpt-4o-mini for first pass
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐      ┌─────────────────────────────┐
│  Node E: Critic             │─────▶│  SQL Corrector              │
│  (Execute SQL)              │ fail │  gpt-4.1, max 3 retries     │
│                             │◀─────│  Gets full error history    │
└──────────────┬──────────────┘      └─────────────────────────────┘
               │ success
               ▼
┌─────────────────────────────┐      ┌─────────────────────────────┐
│  Node F: Validator          │─────▶│  SQL Corrector              │
│  Stage 1: MetricContracts   │ fail │  gpt-4.1, max 2 retries     │
│  Stage 2: Rule-based checks │◀─────│  Gets validator feedback    │
│  Stage 3: LLM semantic      │      └─────────────────────────────┘
└──────────────┬──────────────┘
               │ approved
               ▼
┌─────────────────────────────┐
│  Node G: Summariser         │  Natural language answer (never summarises bad data)
└─────────────────────────────┘
```

**Shared state** (`AgentState` TypedDict) flows through every node — the same pattern as LangGraph's `StateGraph`. Each node returns a partial dict; the orchestrator merges it.

---

## Key Design Decisions

### 1. MetricContract Engine (`src/metric_contracts.py`)

Structural invariants for each metric are encoded as **data**, not `if/elif` chains:

```python
MetricContract(
    metric_id="churn_rate",
    sql_guard_phrases=["IS NULL", "LEFT JOIN", "INTERVAL"],   # must appear in SQL
    sql_forbidden_phrases=["COUNT(DISTINCT curr.user_id) AS churned"],  # must not
    result_invariants=[("churned", "<=", "base")],            # checked row-by-row on DataFrame
    expected_rate_range=(30.0, 80.0) #Removed from the latest commit
)
```

This is what dbt Semantic Layer and Cube.js do at scale. To add a new metric: add one entry to `METRIC_CONTRACTS`. No other files change.

### 2. Three-Stage Validator (cost-ordered)

```
Stage 1 — Contract invariants   (free, deterministic)
Stage 2 — Rule-based heuristics (free, no LLM)
Stage 3 — LLM semantic check    (only if 1 & 2 pass)
```

LLM calls are the most expensive check. They run last.

### 3. Separate Retry Budgets

```python
MAX_SQL_RETRIES   = 3   # Postgres execution errors — cheap to retry
MAX_LOGIC_RETRIES = 2   # Validator rejections — each calls STRONG_MODEL
```

The old architecture had `MAX_ATTEMPTS = 11` with one counter for everything. At 11 strong-model calls per failed question, 100 concurrent users could cause a significant bill spike. Separating the counters caps cost without reducing correctness.

### 4. Singleton Infrastructure (`src/infra.py`)

One `OpenAI()` client, one `create_engine()`, one place to change the DB URL or model:

```python
from src.infra import llm, engine, FAST_MODEL, STRONG_MODEL
```

Previously: 6 `create_engine()` + `OpenAI()` calls across 5 files = 5 separate connection pools per run.

### 5. RAG for Business Rules, Caching for Schema

Business rules (CAC join strategy, cohort denominators) cannot be inferred from column names — they encode domain decisions that need retrieval.

Schema is deterministic and cached in-memory via SQLAlchemy.

---

## Dataset

Synthetically generated Southeast Asian e-commerce dataset.

| Table | Rows | Description |
|---|---|---|
| `users` | 5,000 | Signups across SG, MY, TH, ID, VN — device + acquisition channel |
| `orders` | 50,000 | Revenue, discount, COGS by product category |
| `marketing_campaigns` | 1,000 | Spend, impressions, clicks, conversions by channel |
| `user_activity` | 100,000 | Session events: login, view, add_to_cart, purchase |

---

## Project Structure

```
shopsphere_agent/
├── main.py                        ← Entry point (REPL + single-question mode)
├── requirements.txt
├── .env.example                   ← Copy to .env and add OPENAI_API_KEY
├── data/
│   ├── users.csv
│   ├── orders.csv
│   ├── marketing.csv
│   └── activity.csv
└── src/
    ├── infra.py                   ← Singleton: shared LLM client + DB engine
    ├── state.py                   ← AgentState TypedDict (shared across all nodes)
    ├── metric_contracts.py        ← MetricContract engine (structural invariants)
    ├── orchestrator.py            ← Pipeline controller with separate retry budgets
    ├── business_glossary.json     ← Domain metric definitions + SQL guardrails
    ├── generate_data.py           ← Generates all 4 CSV files
    ├── setup_db.py                ← Loads CSVs into PostgreSQL or SQLite
    └── nodes/
        ├── rag.py                 ← Node A: business rule retrieval
        ├── query_planner.py       ← Node B: question → QueryPlan
        ├── data_profiler.py       ← Node C: live column value profiling
        ├── sql_generator.py       ← Node D: SQL generation + correction
        ├── critic.py              ← Node E: SQL execution + Node G: summariser
        └── validator.py           ← Node F: 3-stage validation pipeline
```

---

## Setup

**1. Clone and install**
```bash
git clone https://github.com/your-username/shopsphere-analytics-agent
cd shopsphere-analytics-agent
pip install -r requirements.txt
```

**2. Configure environment**
```bash
cp .env.example .env
# Add your OPENAI_API_KEY to .env
```

**3. Generate the dataset**
```bash
python src/generate_data.py
```

**4. Load into database**

SQLite (no server required):
```bash
python src/setup_db.py --sqlite
```

PostgreSQL (if you have a local instance):
```bash
# Set DATABASE_URL in .env first, then:
python src/setup_db.py
```

**5. Build the ChromaDB vector index**
```bash
python src/build_index.py
```
This embeds all metric definitions from `business_glossary.json` into a local ChromaDB store (`.chromadb/`). Run this once, and again whenever you update the glossary. The index persists on disk — no re-embedding on every agent start.

**6. Run**
```bash
# Interactive REPL
python main.py

# Single question
python main.py "What is our monthly churn rate?"
```

---

## Example Questions

**Inline answers (aggregate):**
```
What is our overall CAC payback period?
Which country generates the highest average revenue per order?
Which acquisition channel has the best D+1 retention rate?
What is the overall conversion rate across the purchase funnel?
```

**Recommendation steps (Optional):**
```
Talk to LLM first, ask for a 1-liner prompt for the metrics you want.
Example: Calculate the monthly retention rate based solely on user activity in year 2025. A user is considered retained if they were active in the previous month and are also active in the current month. Do not use signup date, user age, minimum activity thresholds, cohort definitions, or any other constraints. Retention should be based purely on month-over-month activity presence.
LLM Output: "Calculate month-over-month active user retention for 2025, where a user is counted as retained only if they were active in both the current month and the immediately preceding month, expressed as a percentage of the prior month's active base."
```

**Excel export (trend/dashboard):**
```
Give me the monthly churn rate trend for a dashboard
Monthly revenue by product category over time
Day-by-day DAU/MAU engagement ratio for the last 90 days
```

Excel exports include two tabs: `Dashboard_Data` (query results) and `SQL_QA_Audit` (question + executed SQL for traceability).

---

## What I Built and Why

| Component | What it does | Why it matters |
|---|---|---|
| `MetricContract` | Encodes metric invariants as data | Same pattern as dbt Semantic Layer — no metric-specific if/elif |
| 3-stage Validator | Contracts → rules → LLM | Cost-ordering: expensive LLM only runs when cheap checks pass |
| Separate retry budgets | `MAX_SQL_RETRIES=3`, `MAX_LOGIC_RETRIES=2` | Prevents bill spikes; makes cost per question predictable |
| `src/infra.py` singleton | One engine, one client | Eliminates duplicate connection pools; one place to change config |
| `AgentState` TypedDict | Shared state across nodes | Maps directly to LangGraph's `StateGraph` — concrete next step |
| Data Profiler | SELECT DISTINCT before SQL gen | Prevents hallucinated WHERE clause values |
| RAG for business rules | Glossary keyword-match | CAC join logic, cohort denominators can't be inferred from column names |

---

## Next Steps

- [ ] Migrate nodes into a LangGraph `StateGraph` with explicit conditional edges
- [ ] Replace JSON glossary with ChromaDB vector search (true semantic RAG) ✅ (Done)
- [ ] Add Redis schema cache with TTL-based invalidation on migrations
- [ ] Multi-question session state for follow-up questions

---

## Tech Stack

Python · OpenAI API (gpt-4o-mini + gpt-4.1) · PostgreSQL / SQLite · SQLAlchemy · pandas · openpyxl

---

*Self-study project — built to learn multi-agent orchestration, LLM-driven SQL pipelines, and production data quality patterns.*
