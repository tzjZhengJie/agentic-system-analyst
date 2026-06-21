# main.py
# ==========================================
# ENTRY POINT — SHOPSPHERE ANALYTICS AGENT
# ==========================================
# Usage:
#   python main.py                  # interactive REPL
#   python main.py "your question"  # single question, then exit

import sys
from src.orchestrator import ask_agent


def print_banner():
    print("=" * 55)
    print("  ShopSphere Analytics Agent")
    print("  Natural language → SQL → validated answer")
    print("=" * 55)
    print("  Type 'exit' to quit.\n")


def run():
    print_banner()

    # Support single-question mode: python main.py "what is our churn rate?"
    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
        ask_agent(question)
        return

    while True:
        try:
            q = input("Ask Agent: ").strip()
            if not q:
                continue
            if q.lower() in ("exit", "quit", "q"):
                print("Goodbye!")
                break
            ask_agent(q)
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break


if __name__ == "__main__":
    run()
