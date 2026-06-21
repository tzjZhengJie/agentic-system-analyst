# src/nodes/rag.py
# ==========================================
# AGENT NODE A — CHROMADB RAG LOOKUP
# ==========================================
# Replaces keyword matching with semantic vector search.
#
# OLD approach (keyword):
#   "rolling retention" → scans every glossary entry for exact keyword overlap
#   → misses "cohort_retention" because "rolling" and "cohort" don't share words
#   → matches "retention_metrics" because "retention" is a keyword
#   → injects wrong rules (signup-based logic for an activity-based question)
#
# NEW approach (vector search):
#   "rolling retention" → embedded into a 1536-dim vector
#   → ChromaDB finds the N nearest metric documents by cosine similarity
#   → similarity is computed on MEANING, not word overlap
#   → "users active last week still active this week" retrieves
#      "rolling_retention" even with zero keyword overlap
#
# How cosine similarity works (plain English):
#   Each document and query is a point in high-dimensional space.
#   Cosine similarity measures the angle between two vectors — 1.0 means
#   identical direction (same meaning), 0.0 means orthogonal (unrelated).
#   We retrieve documents whose vectors point in the same direction as the query.
#
# Two-stage retrieval (the key addition over raw similarity search):
#   Stage 1 — Vector search: retrieve top-K candidates by cosine similarity
#   Stage 2 — Threshold filter: only keep results above MIN_SIMILARITY
#   This prevents injecting weakly-related rules when nothing relevant exists.
#
# Input state fields read:  user_question
# Output state fields set:  business_rules

import json
import os
from pathlib import Path
from typing import Optional

import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv

from src.state import AgentState

_root = Path(__file__).resolve().parent.parent.parent
load_dotenv(_root / ".env")

CHROMA_PATH     = _root / ".chromadb"
COLLECTION_NAME = "metric_definitions"

# How many candidate documents to retrieve before threshold filtering.
# Higher = more recall, more noise. 3 is right for a 10-metric glossary.
# At 500+ metrics you'd raise this to 5-10.
TOP_K = 3

# Minimum cosine similarity to include a result.
# 0.0 = identical, 1.0 = completely unrelated (cosine distance).
# ChromaDB returns distances, not similarities: distance = 1 - similarity.
# So MIN_DISTANCE = 0.45
# Tuning guide:
# Run a question and check the printed similarity scores.
# If a correct metric is filtered out → raise MIN_DISTANCE closer to 0.5
# If wrong metrics are sneaking in    → lower MIN_DISTANCE closer to 0.3
# 0.45 is calibrated for this 11-metric glossary with text-embedding-3-small. means we require similarity >= 0.65.
# Tune this by running queries and inspecting the distances printed below.

MIN_DISTANCE = 0.45


# ── CHROMADB CLIENT (module-level singleton) ───────────────────────────────────
# The client is initialised once when the module is first imported.
# Subsequent calls reuse the same client — no reconnection overhead.
# This is the same pattern as src/infra.py for the DB engine.

_client: Optional[chromadb.PersistentClient] = None
_collection: Optional[chromadb.Collection] = None


def _get_collection():
    """
    Lazy initialisation — connects to ChromaDB on first use.
    Raises a clear error if the index hasn't been built yet.
    """
    global _client, _collection

    if _collection is not None:
        return _collection

    if not CHROMA_PATH.exists():
        raise RuntimeError(
            "[RAG] ChromaDB index not found. "
            "Run:  python src/build_index.py  first."
        )

    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key=os.getenv("OPENAI_API_KEY"),
        model_name="text-embedding-3-small"
    )

    _client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    _collection = _client.get_collection(
        name=COLLECTION_NAME,
        embedding_function=openai_ef
    )

    print(f"[RAG] Connected to ChromaDB — {_collection.count()} metric(s) indexed.")
    return _collection


# ── DOCUMENT FORMATTER ────────────────────────────────────────────────────────
# Same formatter as before — takes the raw_json stored in ChromaDB metadata
# and produces the structured prompt block for the SQL Generator.

def _format_metric_block(metric_id: str, metadata: dict) -> str:
    lines = [f"\n--- METRIC: {metric_id.upper()} ---"]

    if "description" in metadata:
        lines.append(f"Description: {metadata['description']}")

    if "logic" in metadata:
        logic = metadata["logic"]
        lines.append("Formula Logic:")
        lines.append(f"  Numerator:   {logic.get('numerator', 'N/A')}")
        lines.append(f"  Denominator: {logic.get('denominator', 'N/A')}")
        lines.append(f"  Granularity: {logic.get('time_granularity', 'N/A')}")

    if "business_constraints" in metadata:
        lines.append("Business Constraints (MUST follow exactly):")
        for rule in metadata["business_constraints"]:
            lines.append(f"  - {rule}")

    if "formula" in metadata:
        formula = metadata["formula"]
        if formula and formula != "N/A":
            lines.append(f"Formula: {formula}")

    if "rules" in metadata:
        rules = metadata["rules"]
        if rules:
            lines.append("Rules:")
            for rule in rules:
                lines.append(f"  - {rule}")

    if "pattern" in metadata:
        lines.append("Validated SQL Pattern (use this structure exactly):")
        lines.append(metadata["pattern"])

    return "\n".join(lines)


# ── RETRIEVAL ─────────────────────────────────────────────────────────────────

def retrieve_business_rules(question: str) -> str:
    """
    Two-stage semantic retrieval:
      1. Embed the question and find top-K nearest metric documents
      2. Filter by distance threshold to drop weakly-related results

    ChromaDB query() returns:
      results["documents"]  — the text of each retrieved document
      results["metadatas"]  — the metadata dict for each document
      results["distances"]  — cosine distance for each result (lower = more similar)
      results["ids"]        — the metric_id for each result
    """
    try:
        collection = _get_collection()
    except RuntimeError as e:
        print(e)
        return "No glossary available. Use standard aggregation logic."

    # ── Stage 1: Vector search ────────────────────────────────────────────────
    # ChromaDB embeds `question` using the same model used at index time,
    # then returns the TOP_K most similar documents.
    results = collection.query(
        query_texts=[question],
        n_results=min(TOP_K, collection.count()),
        include=["documents", "metadatas", "distances"]
    )

    # Results are nested lists because query() supports multiple queries at once.
    # We sent one query, so we index [0].
    distances = results["distances"][0]
    metadatas = results["metadatas"][0]
    ids       = results["ids"][0]

    # Print distances so you can tune MIN_DISTANCE during development
    print("[RAG] Retrieval results:")
    for metric_id, dist in zip(ids, distances):
        similarity = 1 - dist
        marker = "✅" if dist < MIN_DISTANCE else "❌"
        print(f"  {marker} {metric_id:<35} similarity={similarity:.3f}  distance={dist:.3f}")

    # ── Stage 2: Threshold filter ─────────────────────────────────────────────
    # Drop any result where the distance exceeds MIN_DISTANCE.
    # This prevents injecting CAC rules into a churn question just because
    # both mention "users" and "monthly".
    collected = []
    for metric_id, dist, meta in zip(ids, distances, metadatas):
        if dist >= MIN_DISTANCE:
            continue  # too dissimilar — skip

        # Reconstruct the full glossary entry from the raw_json we stored
        raw_metadata = json.loads(meta["raw_json"])
        block = _format_metric_block(metric_id, raw_metadata)
        collected.append(block)

    if collected:
        print(f"[RAG] Injecting {len(collected)} rule block(s) above similarity threshold.")
        return "\n".join(collected)

    print("[RAG] No results above similarity threshold. Using standard aggregation logic.")
    return "No specific metric matched. Use standard transactional aggregation."


# ── NODE ENTRY POINT ──────────────────────────────────────────────────────────

def rag_node(state: AgentState) -> dict:
    rules = retrieve_business_rules(state["user_question"])
    return {"business_rules": rules}
