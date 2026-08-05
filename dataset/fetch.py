"""Resumable, verified, idempotent downloader driven by sources.yaml.

Design notes:
  * Resume via HTTP Range so a dropped 15 GB shard doesn't restart from zero.
  * sha256 recorded on first success into data/raw/.lock.json, verified on every
    later run. Unknown-hash sources are therefore self-pinning after run one.
  * Idempotent: a file already present with a matching hash is skipped entirely,
    so re-running the pipeline costs nothing.
  * Parallel across HOSTS, serial within a host, so we don't hammer one server.

Usage:
    python dataset/fetch.py                # everything enabled
    python dataset/fetch.py --only crackseg9k_vol1 wood_semantic_maps
    python dataset/fetch.py --list         # show plan, download nothing
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urlparse

import requests
import yaml

ROOT = Path(__file__).resolve().parent.parent
CFG = Path(__file__).resolve().parent / "sources.yaml"
CHUNK = 1 << 20  # 1 MiB
_print_lock = threading.Lock()


def log(msg):
    with _print_lock:
        print(msg, flush=True)


def human(n):
    if not n:
        return "?"
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}TB"


def sha256_file(path, bufsize=1 << 22):
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(bufsize):
            h.update(chunk)
    return h.hexdigest()


_lock_write = threading.Lock()


def lock_path(raw_dir):
    return raw_dir / ".lock.json"


def lock_load(raw_dir):
    """What previous runs recorded downloading, so this one can verify."""
    p = lock_path(raw_dir)
    return json.loads(p.read_text()) if p.exists() else {}


def lock_record(raw_dir, lock, name, **kw):
    """Merge one entry and rewrite via a temp file -- workers write concurrently, and a
    half-written lock file would fail every later verification."""
    with _lock_write:
        lock[name] = {**lock.get(name, {}), **kw}
        p = lock_path(raw_dir)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(lock, indent=2, sort_keys=True))
        tmp.replace(p)


def download(name, spec, raw_dir, lock, retries=4):
    dest = raw_dir / spec["filename"]
    expected_hash = spec.get("sha256") or lock.get(name, {}).get("sha256")
    expected_size = spec.get("bytes") or lock.get(name, {}).get("bytes")

    # --- already have it? -------------------------------------------------
    if dest.exists() and expected_hash:
        log(f"[{name}] verifying existing file ...")
        if sha256_file(dest) == expected_hash:
            log(f"[{name}] OK (cached, hash matches) -> {dest.name}")
            return True
        log(f"[{name}] hash MISMATCH on existing file, re-downloading")
        dest.unlink()
    elif dest.exists() and expected_size and dest.stat().st_size == expected_size:
        log(f"[{name}] present with expected size, hashing to pin it ...")
        h = sha256_file(dest)
        lock_record(raw_dir, lock, name, sha256=h, bytes=dest.stat().st_size,
                    url=spec["url"])
        log(f"[{name}] OK (cached) sha256={h[:16]}...")
        return True

    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")

    for attempt in range(1, retries + 1):
        have = part.stat().st_size if part.exists() else 0
        headers = {"Range": f"bytes={have}-"} if have else {}
        try:
            with requests.get(spec["url"], headers=headers, stream=True, timeout=60,
                              allow_redirects=True) as r:
                if have and r.status_code == 200:
                    # server ignored Range; start clean
                    have = 0
                    part.unlink(missing_ok=True)
                elif have and r.status_code == 416:
                    break  # already complete
                r.raise_for_status()

                total = int(r.headers.get("Content-Length", 0)) + have
                mode = "ab" if have else "wb"
                t0, last = time.time(), have
                with part.open(mode) as fh:
                    for chunk in r.iter_content(CHUNK):
                        fh.write(chunk)
                        have += len(chunk)
                        now = time.time()
                        if now - t0 > 20:
                            rate = (have - last) / (now - t0)
                            pct = f"{100*have/total:.0f}%" if total else "?"
                            eta = f"{(total-have)/rate/60:.0f}m" if rate and total else "?"
                            log(f"[{name}] {pct} {human(have)}/{human(total)} "
                                f"{human(rate)}/s eta {eta}")
                            t0, last = now, have
            break
        except Exception as e:  # noqa: BLE001 - network layer, retry everything
            if attempt == retries:
                log(f"[{name}] FAILED after {retries} attempts: {e}")
                return False
            wait = 2 ** attempt
            log(f"[{name}] attempt {attempt} failed ({e}); retrying in {wait}s")
            time.sleep(wait)

    if not part.exists():
        log(f"[{name}] FAILED: nothing downloaded")
        return False

    size = part.stat().st_size
    if expected_size and abs(size - expected_size) > max(4096, 0.01 * expected_size):
        log(f"[{name}] SIZE MISMATCH: got {human(size)}, expected {human(expected_size)}")
        log(f"[{name}] -> source may have moved or changed. Not accepting this file.")
        return False

    h = sha256_file(part)
    if expected_hash and h != expected_hash:
        log(f"[{name}] HASH MISMATCH: {h[:16]}... != {expected_hash[:16]}...")
        return False

    part.replace(dest)
    lock_record(raw_dir, lock, name, sha256=h, bytes=size, url=spec["url"],
                license=spec.get("license"), tier=spec.get("tier"),
                fetched_at=time.strftime("%Y-%m-%dT%H:%M:%S"))
    log(f"[{name}] DONE {human(size)} sha256={h[:16]}... -> {dest.name}")
    if not spec.get("verified", False):
        log(f"[{name}] NOTE: source was unverified; hash now pinned in .lock.json")
    return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", help="subset of source names")
    ap.add_argument("--list", action="store_true", help="show plan and exit")
    ap.add_argument("--workers", type=int, default=3)
    args = ap.parse_args()

    cfg = yaml.safe_load(CFG.read_text(encoding="utf-8"))
    raw_dir = ROOT / cfg.get("defaults", {}).get("raw_dir", "data/raw")
    lock = lock_load(raw_dir)

    todo = {n: s for n, s in cfg["sources"].items()
            if (s.get("enabled", True) or (args.only and n in args.only))
            and (not args.only or n in args.only)}

    if args.list or not todo:
        print(f"{'source':<24}{'tier':<6}{'size':>10}  {'license':<18} url")
        for n, s in cfg["sources"].items():
            mark = "*" if n in todo else " "
            print(f"{mark}{n:<23}{s.get('tier',''):<6}{human(s.get('bytes')):>10}  "
                  f"{str(s.get('license','')):<18} {s['url'][:60]}")
        total = sum(s.get("bytes") or 0 for s in todo.values())
        print(f"\nenabled: {len(todo)} source(s), ~{human(total)} known size")
        if args.list:
            return 0

    # group by host: parallel across hosts, serial within one
    by_host: dict[str, list] = defaultdict(list)
    for n, s in todo.items():
        by_host[urlparse(s["url"]).netloc].append((n, s))

    results: dict[str, bool] = {}

    def run_host(items):
        for n, s in items:
            results[n] = download(n, s, raw_dir, lock)

    log(f"fetching {len(todo)} source(s) across {len(by_host)} host(s) -> {raw_dir}")
    with ThreadPoolExecutor(max_workers=min(args.workers, len(by_host))) as ex:
        list(ex.map(run_host, by_host.values()))

    ok = [n for n, v in results.items() if v]
    bad = [n for n, v in results.items() if not v]
    log(f"\n=== {len(ok)} ok, {len(bad)} failed ===")
    for n in bad:
        log(f"  FAILED: {n}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
