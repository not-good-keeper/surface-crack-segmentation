"""Drop and rebuild the database and generated demo assets.

Usage: python -m scripts.reset_db [--anchor YYYY-MM-DD]
"""

from __future__ import annotations

import argparse
from datetime import datetime

from app.database.seed import seed


def main() -> None:
    parser = argparse.ArgumentParser(description="Reset and reseed the inspection database.")
    parser.add_argument("--anchor", help="anchor date (YYYY-MM-DD) for reproducible timestamps")
    args = parser.parse_args()
    anchor = datetime.fromisoformat(args.anchor) if args.anchor else None
    summary = seed(reset=True, anchor=anchor)
    print("Database reset and reseeded.")
    print(f"  {summary['total_inspections']} inspections, {summary['total_regions']} regions")
    print(f"  batch runs: {summary['batch_runs']}")


if __name__ == "__main__":
    main()
