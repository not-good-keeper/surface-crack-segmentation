"""Build the compact demo bundle that gets deployed.

    python -m scripts.build_demo_bundle

The full local seed writes about 280 images at up to 960 px and takes minutes; that is
right for a demo on a laptop and wrong for a serverless upload. This script seeds the
same sequence with fewer inspections and a smaller image cap, then reports the bundle
size so it can be checked against the platform's limit before deploying.

Run this before `vercel deploy`. The resulting data/ folder is committed and shipped;
the deployed application never seeds itself, because a cold start has seconds, not
minutes.
"""

from __future__ import annotations

import argparse
import shutil
from datetime import datetime
from pathlib import Path

from app.config import get_settings

#: Deliberately small. Every state is still represented - the forced sequence at the
#: front of the seed guarantees clean, crack, scratch, mixed and both failure kinds.
DEMO_LIVE_COUNT = 60  # keeps the deployed bundle above the 75-inspection floor in the spec
DEMO_IMAGE_CAP = 480
DEMO_BATCH_FOLDERS = [
    ("2026-08-12-steel-lineB", "steel", 7),
    ("2026-08-11-ceramic-lineA", "ceramic", 5),
    ("2026-08-10-plastic-lineB", "plastic", 4),
]


def folder_size_mb(path: Path) -> float:
    if not path.exists():
        return 0.0
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1024**2


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the deployable demo bundle.")
    parser.add_argument("--anchor", default="2026-08-06", help="anchor date for reproducible timestamps")
    parser.add_argument("--live", type=int, default=DEMO_LIVE_COUNT)
    parser.add_argument("--cap", type=int, default=DEMO_IMAGE_CAP, help="longest image edge in pixels")
    args = parser.parse_args()

    settings = get_settings()
    if settings.is_serverless:
        raise SystemExit("Run this locally, not on the serverless host.")

    # The cap is read by the mock provider through settings.
    object.__setattr__(settings, "mock_image_max_width", args.cap)

    from app.database import seed as seed_module

    seed_module.BATCH_FOLDERS = DEMO_BATCH_FOLDERS

    print(f"Building demo bundle: {args.live} live inspections, images capped at {args.cap} px")
    summary = seed_module.seed(
        settings, anchor=datetime.fromisoformat(args.anchor), reset=True, live_count=args.live
    )

    # Records log and exports are runtime artefacts; they are not part of the bundle.
    (settings.bundled_data_dir / "records.jsonl").unlink(missing_ok=True)
    shutil.rmtree(settings.export_dir, ignore_errors=True)
    for suffix in ("-wal", "-shm"):
        Path(str(settings.db_file) + suffix).unlink(missing_ok=True)

    # Only the folders that are actually deployed. `data/` in this repository also
    # holds the training corpus and the benchmark checkpoints - about 93 GB - so
    # measuring the whole directory reported a bundle 3,000x its real size and buried
    # the check that matters under a warning that always fires.
    bundle_parts = {
        "database": settings.db_file.stat().st_size / 1024**2,
        "sources": folder_size_mb(settings.source_dir),
        "overlays": folder_size_mb(settings.overlay_dir),
        "batches": folder_size_mb(settings.batch_dir),
        "metrics": folder_size_mb(settings.metrics_file.parent),
    }
    print()
    print(f"  inspections      {summary['total_inspections']}")
    print(f"  regions          {summary['total_regions']}")
    print(f"  batch runs       {summary['batch_runs']}")
    for status, count in sorted(summary["status_totals"].items()):
        print(f"    {status:<22} {count}")
    print()
    for name, size in bundle_parts.items():
        print(f"  {name:<16} {size:.1f} MB")
    total = sum(bundle_parts.values())
    print(f"  {'bundle total':<16} {total:.1f} MB")
    print()

    # The function also carries the Python dependencies - roughly 60 MB for numpy,
    # Pillow, FastAPI and Pydantic - so the ceiling checked here is the bundle's share
    # of the 250 MB limit set in vercel.json, not the whole limit.
    if total > 150:
        print(f"  WARNING: {total:.1f} MB of data plus ~60 MB of dependencies is close to")
        print("  the 250 MB lambda limit. Lower --live or --cap (see docs/DEPLOYMENT.md).")
    else:
        print(f"  {total:.1f} MB of data plus ~60 MB of dependencies fits the 250 MB")
        print("  lambda limit configured in vercel.json.")


if __name__ == "__main__":
    main()
