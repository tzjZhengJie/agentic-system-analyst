# src/build_index.py
# ==========================================
# CHROMADB INDEX BUILDER
# ==========================================
# Run this ONCE (or whenever business_glossary.json changes) to embed
# all metric definitions into a persistent ChromaDB vector store.
#
# Usage:
#   python src/build_index.py
#
# What it produces:
#   .chromadb/          ← persisted vector store (add to .gitignore)
#
# How ChromaDB works (the 30-second version):
#   1. You give it text documents + optional metadata
#   2. It embeds each document into a high-dimensional vector using an
#      embedding model (we use OpenAI text-embedding-3-small)
#   3. Those vectors are stored on disk in .chromadb/
#   4. At query time, your question is embedded the same way, and
#      ChromaDB finds the N closest vectors by cosine similarity
#   5. "Close in vector space" ≈ "similar meaning" — this is why
#      "how much did we lose" can match "churn_rate" without the
#      word "churn" appearing anywhere in the question
#
# Why this beats keyword matching:
#   Keyword: "rolling retention" → misses "cohort_retention" (no overlap)
#   Vector:  "rolling retention" → finds "cohort_retention" because the
#            *meaning* of both phrases is close in embedding space

import json
import os
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv

_root = Path(__file__).resolve().parent.parent
load_dotenv(_root / ".env")

GLOSSARY_PATH  = _root / "src" / "business_glossary.json"
CHROMA_PATH    = _root / ".chromadb"
COLLECTION_NAME = "metric_definitions"


def _build_document(metric_id: str, metadata: dict) -> str:
    """
    Converts one glossary entry into a rich text document for embedding.

    The document is intentionally verbose — more semantic signal = better
    retrieval. We include the metric name, description, formula, rules,
    and business constraints all in one string.

    This is called "document preparation" or "chunking strategy".
    For a glossary this size, one document per metric is correct.
    At thousands of metrics you'd split by section.
    """
    parts = [f"Metric: {metric_id.replace('_', ' ').title()}"]

    if "description" in metadata:
        parts.append(f"Description: {metadata['description']}")

    if "logic" in metadata:
        logic = metadata["logic"]
        parts.append(f"Formula: {logic.get('numerator', '')} divided by {logic.get('denominator', '')}")
        parts.append(f"Time granularity: {logic.get('time_granularity', '')}")

    if "formula" in metadata:
        parts.append(f"Formula: {metadata['formula']}")

    if "rules" in metadata:
        parts.append("Rules: " + " | ".join(metadata["rules"]))

    if "business_constraints" in metadata:
        parts.append("Constraints: " + " | ".join(metadata["business_constraints"]))

    # Keywords are included in the document so they boost retrieval
    # when the user happens to use them — belt AND braces
    if "keywords" in metadata:
        parts.append(f"Also known as: {', '.join(metadata['keywords'])}")

    return "\n".join(parts)


def build_index() -> None:
    print("=" * 55)
    print("  ShopSphere — Building ChromaDB Index")
    print("=" * 55)

    # Load glossary
    with open(GLOSSARY_PATH, "r") as f:
        glossary = json.load(f)

    # ── ChromaDB client (persistent = survives restarts) ──────────────────────
    # PersistentClient writes to disk at CHROMA_PATH.
    # Every time your agent starts, it reads from here — no re-embedding needed.
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))

    # ── Embedding function ─────────────────────────────────────────────────────
    # OpenAIEmbeddingFunction calls text-embedding-3-small via the API.
    # This is the same model OpenAI recommends for RAG — good quality/cost ratio.
    # Alternative: chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction
    # for a fully local option (no API cost, slightly lower quality).
    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key=os.getenv("OPENAI_API_KEY"),
        model_name="text-embedding-3-small"
    )

    # ── Collection ─────────────────────────────────────────────────────────────
    # A collection is like a table in a DB — it holds documents + their vectors.
    # get_or_create: safe to re-run; won't duplicate if collection already exists.
    # delete_collection first if you want a clean rebuild.
    try:
        client.delete_collection(COLLECTION_NAME)
        print(f"  Deleted existing collection '{COLLECTION_NAME}' for clean rebuild.")
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=openai_ef,
        # cosine distance = standard for semantic similarity
        # other options: "l2" (euclidean), "ip" (inner product)
        metadata={"hnsw:space": "cosine"}
    )

    # ── Index each metric ──────────────────────────────────────────────────────
    documents = []
    metadatas = []
    ids       = []

    for metric_id, metadata in glossary.items():
        if metric_id == "_comment" or not isinstance(metadata, dict):
            continue

        doc = _build_document(metric_id, metadata)

        # Metadata stored alongside the vector — retrieved with the document.
        # We store the full raw metadata so the RAG node can format it properly.
        # ChromaDB metadata values must be str/int/float/bool — no nested dicts.
        chroma_meta = {
            "metric_id":  metric_id,
            "has_pattern": "pattern" in metadata,
            "keywords":   ", ".join(metadata.get("keywords", [])),
            # Store the raw JSON so we can reconstruct the full entry on retrieval
            "raw_json":   json.dumps(metadata),
        }

        documents.append(doc)
        metadatas.append(chroma_meta)
        ids.append(metric_id)

        print(f"  Indexed: {metric_id}")

    # ── Upsert all documents in one batch ─────────────────────────────────────
    # ChromaDB calls the embedding function once per batch — efficient.
    # For large glossaries (1000+ metrics), batch in chunks of 100.
    collection.add(
        documents=documents,
        metadatas=metadatas,
        ids=ids,
    )

    print(f"\n✅ Index built — {len(ids)} metric(s) embedded into '{COLLECTION_NAME}'")
    print(f"   Stored at: {CHROMA_PATH}")
    print(f"\nNext step: python main.py")


if __name__ == "__main__":
    build_index()
