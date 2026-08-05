"""Unattended download runner with a hard wall-clock deadline and a disk reserve.

Ordering is deliberate: cheap/high-value sources first, so that if anything runs
long the thing we lose is the marginal wood shard, never the backbone.

  1. Severstal (steel surfaces + steel distractors)
  2. wood yield scan, if it hasn't produced data/wood_yield.csv yet
  3. wood image shards, best crack-yield-per-GB first, taken only while BOTH
     the projected finish time and the disk reserve still allow it

Two budgets are enforced before each shard, never mid-flight:
  * time  -- projected finish must land before --deadline
  * disk  -- must leave --reserve-gb free for the patch memmap built later

Usage:
    python dataset/overnight.py --deadline 08:00 --reserve-gb 75
"""
import argparse
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
PY = str(ROOT / ".venv/Scripts/python.exe")
CFG = ROOT / "dataset/sources.yaml"
YIELD = ROOT / "data/wood_yield.csv"
LOG = ROOT / "data/overnight.log"


def log(msg):
    line = f"[{datetime.now():%H:%M:%S}] {msg}"
    print(line, flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def free_gb():
    return shutil.disk_usage(ROOT).free / 1e9


def run(cmd, tag):
    log(f"$ {' '.join(cmd[1:])}")
    p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    for ln in (p.stdout or "").splitlines():
        if not re.match(r"^\[.*\] \d+%", ln):  # drop progress spam from the log
            log(f"  {tag}| {ln}")
    if p.returncode != 0:
        for ln in (p.stderr or "").splitlines()[-6:]:
            log(f"  {tag}! {ln}")
    return p.returncode == 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--deadline", default="08:00", help="HH:MM local, hard stop")
    ap.add_argument("--reserve-gb", type=float, default=75.0,
                    help="disk to leave free for the patch memmap")
    ap.add_argument("--rate-mbs", type=float, default=7.0,
                    help="conservative throughput estimate for planning")
    ap.add_argument("--buffer-min", type=float, default=25.0)
    ap.add_argument("--max-shards", type=int, default=10)
    args = ap.parse_args()

    now = datetime.now()
    hh, mm = map(int, args.deadline.split(":"))
    deadline = now.replace(hour=hh, minute=mm, second=0, microsecond=0)
    if deadline <= now:
        deadline += timedelta(days=1)
    deadline -= timedelta(minutes=args.buffer_min)

    log("=" * 68)
    log(f"overnight runner | now {now:%H:%M} -> effective deadline {deadline:%H:%M} "
        f"({(deadline-now).total_seconds()/3600:.1f} h)")
    log(f"disk free {free_gb():.0f} GB, reserving {args.reserve_gb:.0f} GB")

    # ---------------------------------------------------------------- 1. Severstal
    run([PY, "dataset/kaggle_fetch.py", "--only", "severstal"], "sev")

    # ---------------------------------------------------------------- 2. yield scan
    if not YIELD.exists():
        log("wood yield scan not finished; waiting up to 40 min for it")
        for _ in range(80):
            if YIELD.exists():
                break
            time.sleep(30)
    if not YIELD.exists():
        log("no wood_yield.csv -> running the scan now")
        run([PY, "dataset/wood_yield.py", "--workers", "8"], "wy")

    if not YIELD.exists():
        log("STILL no wood yield data; skipping wood entirely. "
            "Everything else is downloaded, so this is degraded, not fatal.")
        return 1

    # ---------------------------------------------------------------- 3. rank shards
    df = pd.read_csv(YIELD, dtype={"image_id": str})
    crack_ids = set(df.loc[df.any_crack, "image_id"])
    log(f"wood: {len(df)} masks, {len(crack_ids)} carry crack pixels")

    cfg = yaml.safe_load(CFG.read_text(encoding="utf-8"))
    shards = {n: s for n, s in cfg["sources"].items() if n.startswith("wood_images")}

    # Shard membership was computed by wood_yield.py via remote central-directory
    # reads; recompute here so the runner is self-contained if that file is stale.
    sys.path.insert(0, str(ROOT / "dataset"))
    from wood_yield import remote_namelist  # noqa: E402

    rank = []
    for name, spec in sorted(shards.items(), key=lambda kv: kv[1]["shard"]):
        try:
            ids = {Path(n).stem for n in remote_namelist(spec["url"])
                   if n.lower().endswith((".bmp", ".png", ".jpg"))}
            hits = len(ids & crack_ids)
            gb = spec["bytes"] / 1e9
            rank.append(dict(name=name, shard=spec["shard"], gb=gb,
                             hits=hits, per_gb=hits / gb if gb else 0))
            log(f"  shard {spec['shard']:>2}: {hits:>5} crack imgs / {gb:.1f} GB "
                f"= {hits/gb:.0f} per GB")
        except Exception as e:  # noqa: BLE001
            log(f"  shard {spec['shard']:>2}: listing failed ({e})")

    rank.sort(key=lambda r: -r["per_gb"])

    # ---------------------------------------------------------------- 4. download
    taken, got = [], 0
    for r in rank[:args.max_shards]:
        remain_s = (deadline - datetime.now()).total_seconds()
        need_s = r["gb"] * 1e9 / (args.rate_mbs * 1e6)
        if need_s > remain_s:
            log(f"STOP: shard {r['shard']} needs {need_s/60:.0f} min, "
                f"only {remain_s/60:.0f} min left before deadline")
            break
        if free_gb() - r["gb"] < args.reserve_gb:
            log(f"STOP: shard {r['shard']} would leave "
                f"{free_gb()-r['gb']:.0f} GB, under the {args.reserve_gb:.0f} GB reserve")
            break
        log(f"--> shard {r['shard']} ({r['gb']:.1f} GB, {r['hits']} crack imgs, "
            f"~{need_s/60:.0f} min)")
        if run([PY, "dataset/fetch.py", "--only", r["name"]], f"s{r['shard']}"):
            taken.append(r["shard"])
            got += r["hits"]
        else:
            log(f"shard {r['shard']} failed; continuing with the rest")

    log("=" * 68)
    log(f"DONE. wood shards fetched: {taken or 'none'} -> ~{got} wood crack images")
    log(f"disk free {free_gb():.0f} GB | finished {datetime.now():%H:%M}")
    log("next: python dataset/adapters.py  (wood adapter reads shards in place)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
