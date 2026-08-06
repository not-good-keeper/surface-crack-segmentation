"""Record an exported ONNX model in the database and make it the active one.

    python -m scripts.register_model --model data/export/<file>.onnx
    python -m scripts.register_model --model <file>.onnx --params 1430000 --latency-ms 26

This is the step `docs/ARCHITECTURE.md` §11.3 calls "swap the .onnx and record the
hash". Until it is run, the active `model_version` row still describes whatever was
seeded, the Status screen finds a hash mismatch and inspection is stopped — which is
the intended behaviour, not a bug. A system that quietly accepted an unrecognised
model file would be producing results it could not attribute to anything.

What this script will and will not fill in
------------------------------------------
Read from the file itself: SHA-256, size, and the contract the graph declares —
input size, class count and the input transform, all via `Inspector.describe()`, so the
registration cannot disagree with what the pipeline will actually do.

Not invented: parameter count and latency. Latency in particular is a measurement, and
one taken on a loaded machine is worse than none — `--latency-ms` exists so the number
comes from a deliberate benchmark run rather than from whatever this process happened
to observe. Left unset, both stay NULL and the Materials screen shows an em dash.
"""

from __future__ import annotations

import argparse
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import get_settings  # noqa: E402
from app.database.connection import connect  # noqa: E402
from app.providers.real_provider import sha256_of_file  # noqa: E402


def describe_graph(path: Path) -> dict:
    """Ask the inference core what the graph declares, rather than guessing from a name.

    More than one export can sit in data/export at once, and a 256-input model and a
    512-input one differ in nothing visible in a file listing while accepting the same
    tensor shape. Picking the wrong one costs accuracy silently.
    """
    from app.inference import Inspector

    inspector = Inspector(str(path), station_id="registration")
    return inspector.describe()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", help="path to the .onnx file (default: MODEL_PATH)")
    parser.add_argument("--version", help="version label, e.g. v13 (default: the file stem)")
    parser.add_argument("--params", type=int, help="parameter count, from the training run record")
    parser.add_argument("--latency-ms", type=float, help="measured median latency, one CPU thread")
    parser.add_argument("--precision", default="float32", choices=["float32", "int8"])
    parser.add_argument("--db", help="database file (default: DATABASE_PATH)")
    args = parser.parse_args()

    settings = get_settings()
    path = Path(args.model) if args.model else settings.model_file
    if not path.is_absolute():
        path = (ROOT / path).resolve()
    if not path.exists():
        print(f"no model file at {path}", file=sys.stderr)
        return 1

    digest = sha256_of_file(path)
    size_mb = round(path.stat().st_size / (1024 * 1024), 2)
    version = args.version or path.stem

    try:
        spec = describe_graph(path)
    except Exception as exc:
        print(f"could not read the graph: {exc}", file=sys.stderr)
        print("The file exists but the inference core cannot open it. It is not "
              "registered, so inspection stays stopped.", file=sys.stderr)
        return 1

    conn = connect(Path(args.db) if args.db else None)
    try:
        now = datetime.now(UTC).isoformat(timespec="seconds")
        existing = conn.execute(
            "SELECT model_version_id FROM model_version WHERE artefact_sha256 = ?", [digest]
        ).fetchone()

        # Every row is deactivated first: is_active is a single-winner flag, and two
        # active models would make "which model produced this inspection" unanswerable
        # for every row written afterwards.
        conn.execute("UPDATE model_version SET is_active = 0")

        if existing:
            conn.execute(
                """
                UPDATE model_version
                   SET file_name = ?, version = ?, parameter_count = COALESCE(?, parameter_count),
                       size_mb = ?, precision = ?, latency_ms = COALESCE(?, latency_ms),
                       is_active = 1
                 WHERE model_version_id = ?
                """,
                [path.name, version, args.params, size_mb, args.precision,
                 args.latency_ms, existing["model_version_id"]],
            )
            model_version_id = int(existing["model_version_id"])
            action = "updated"
        else:
            cursor = conn.execute(
                """
                INSERT INTO model_version
                    (file_name, version, artefact_sha256, parameter_count, size_mb,
                     precision, latency_ms, created_at, is_active)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1)
                """,
                [path.name, version, digest, args.params, size_mb, args.precision,
                 args.latency_ms, now],
            )
            model_version_id = int(cursor.lastrowid)
            action = "registered"
        conn.commit()
    finally:
        conn.close()

    print(f"{action} model_version {model_version_id}")
    print(f"  file       {path.name}  ({size_mb} MB, {args.precision})")
    print(f"  sha256     {digest}")
    print(f"  contract   {spec['input_size']} px input | prep {spec['prep']} | "
          f"{spec['input_mode']} | {spec['classes']}-class")
    # ASCII only: this prints to a Windows console under cp1252, where an em dash
    # comes out as a replacement character.
    print(f"  params     {args.params if args.params is not None else 'not recorded'}")
    print(f"  latency    {f'{args.latency_ms} ms' if args.latency_ms is not None else 'not measured'}")
    print()
    print("Set MODEL_SHA256 in .env to this hash to pin it, or leave it blank to let the")
    print("database row be the reference. Either way the Status screen now agrees.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
