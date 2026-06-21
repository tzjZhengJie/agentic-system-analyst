# src/setup_db.py
# ==========================================
# DATABASE SETUP
# ==========================================
# Loads all 4 CSV files into PostgreSQL (or SQLite for local dev).
#
# Usage:
#   python src/setup_db.py                  # PostgreSQL (uses DATABASE_URL from .env)
#   python src/setup_db.py --sqlite         # SQLite (no server needed)
#
# Run AFTER: python src/generate_data.py

import sys
import argparse
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
import os

_root = Path(__file__).resolve().parent.parent
load_dotenv(_root / ".env")

DATA_DIR = _root / "data"

TABLES = {
    "users":                "users.csv",
    "orders":               "orders.csv",
    "marketing_campaigns":  "marketing.csv",
    "user_activity":        "activity.csv",
}


def load(db_url: str) -> None:
    print(f"\nConnecting to: {db_url.split('@')[-1] if '@' in db_url else db_url}")
    engine = create_engine(db_url, echo=False)

    for table_name, csv_file in TABLES.items():
        path = DATA_DIR / csv_file
        if not path.exists():
            print(f"  ❌ {csv_file} not found — run python src/generate_data.py first")
            continue
        df = pd.read_csv(path)
        df.to_sql(table_name, engine, if_exists="replace", index=False)
        print(f"  ✅ {table_name:<25} {len(df):>7,} rows loaded")

    print("\nVerifying row counts:")
    with engine.connect() as conn:
        for table_name in TABLES:
            try:
                count = conn.execute(text(f'SELECT COUNT(*) FROM "{table_name}"')).fetchone()[0]
                print(f"  {table_name:<25} {count:>7,} rows ✓")
            except Exception as e:
                print(f"  {table_name:<25} ❌ {e}")

    print("\n✅ Database ready. Run: python main.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sqlite", action="store_true", help="Use SQLite instead of PostgreSQL")
    args = parser.parse_args()

    if args.sqlite:
        db_url = f"sqlite:///{_root / 'shopsphere.db'}"
    else:
        db_url = os.getenv(
            "DATABASE_URL",
            "postgresql+psycopg2://postgres:postgres@localhost:5432/shopsphere"
        )

    load(db_url)
