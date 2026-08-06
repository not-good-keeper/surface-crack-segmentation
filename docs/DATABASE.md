# Database

SQLite, plain SQL, no ORM. One file, no server to install.

The point of the schema is that **a stored inspection can always be explained later**:
which model file, which thresholds, which station, which image.

---

## Design rules

1. **Nothing is updated or deleted.** Re-running an image writes a new `inspection` row;
   re-tuning thresholds writes a new `profile` version. An old inspection therefore
   still resolves to the thresholds that actually produced it.
2. **`clean` and `acquisition_failure` are different statuses.** A defect-free product
   and an unreadable image are not the same outcome, and collapsing them would hide the
   difference between a good part and a broken camera.
3. **Regions are rows, not JSON.** Storing a list inside `inspection` would break 1NF
   and make the class filter impossible.
4. **Thresholds live in `profile` only**, mirrored from `app/postprocess.py` at seed
   time. The frontend never holds a copy.

## Tables

| Table | Purpose | Key points |
|---|---|---|
| `material` | Materials in scope and their support status | unique `material_code`; status constrained to six values |
| `defect_class` | `crack` and `scratch` | `class_id` matches the model output channel: 1 = crack, 2 = scratch |
| `model_version` | One exported ONNX file | unique `artefact_sha256`; records params, size, precision, latency |
| `station` | Camera station and calibration | `mm_per_pixel` is NULL until the station is calibrated |
| `profile` | A versioned threshold set | unique `(material_id, version_no)`; mirrors `Profile` |
| `batch_run` | One batch job | source folder, image and failure counts |
| `inspection` | One inspected image | FKs to station, profile, model, material, optional batch |
| `defect_region` | One detected region | unique `(inspection_id, region_index)`, cascades on delete |

### `inspection.status`

Constrained by `CHECK` to exactly four values:

```
regions_found        the model found regions; the operator decides what to do
clean                defect-free, recorded with an empty region list
acquisition_failure  the image could not be read
processing_failure   the image was read but the pipeline failed after that
```

The database refuses anything else — including a verdict.
`tests/test_database.py::test_status_column_rejects_an_unknown_state` proves it.

### `defect_region` geometry

| Column | Meaning |
|---|---|
| `area_px` | Pixel count of the connected component |
| `length_px` | Medial-axis arc length, diagonal steps counted as √2 |
| `max_width_px` | Twice the maximum distance-transform value along the skeleton — the widest **inscribable** point, not the widest visual extent |
| `bbox_*`, `centroid_*` | Bounding box and centroid in source-image pixels |

## Relationships

```
material ──1:N──> profile ──1:N──┐
material ──1:N──> batch_run ──1:N──┤
model_version ──1:N────────────────┼──> inspection ──1:N──> defect_region
station ──1:N──────────────────────┘                            │
defect_class ──1:N──────────────────────────────────────────────┘
```

Foreign keys are enabled on **every connection** — SQLite disables them per connection,
not per database, so `PRAGMA foreign_keys = ON` runs in `connect()`.

## Indexes

History filters on six fields (FR-18) and batch reporting aggregates by run (FR-19):

```
idx_inspection_captured_at    (captured_at DESC)   history ordering
idx_inspection_product_id                          product filter
idx_inspection_status                              status filter
idx_inspection_material                            material filter
idx_inspection_station                             station filter
idx_inspection_batch                               batch report
idx_inspection_station_cap    (station_id, captured_at DESC)
idx_region_inspection                              region joins
idx_region_class                                   class filter
idx_profile_active, idx_model_active               active lookups
```

## Normalisation

| Form | How the schema meets it |
|---|---|
| 1NF | Every column holds a single value; regions are a separate table |
| 2NF | Every table has a single-column primary key, so a partial dependency cannot arise |
| 3NF | Material name is not on `inspection`; model hash lives only in `model_version`; thresholds live only in `profile` |

**One deliberate exception:** `inspection.region_count` is stored even though it could be
counted. The history and batch screens list thousands of rows and counting on every
query is wasteful, and the value can never change once the inspection is written.
`tests/test_database.py::test_region_count_matches_the_stored_regions` guards the
denormalisation.

## Seeding

```bash
python -m scripts.seed_db                        # ~140 inspections, 3 batch runs
python -m scripts.seed_db --anchor 2026-08-06    # reproducible timestamps
python -m scripts.reset_db                       # drop and rebuild
```

The seed guarantees coverage: the first twelve inspections force one of each state, so a
freshly seeded database always contains clean, crack-only, scratch-only, mixed,
acquisition-failure and processing-failure cases across several materials, stations and
days. Batch folders are written first so the Live screen shows a station capture rather
than the last file of a batch.

**Determinism.** Statuses, geometry and image pixels are a pure function of `MOCK_SEED`.
Timestamps are laid out backwards from an anchor that defaults to today, so history
looks current; pass `--anchor` to pin it.

The three unreadable files planted in each batch folder are **genuinely broken** — a
zero-byte file, a truncated PNG and a text file with a `.png` extension — so the failure
path is exercised by real decode failures, not by a simulated flag (T-08).

## Migrations

The schema is applied with `CREATE TABLE IF NOT EXISTS`, so start-up is idempotent.
There is no versioned migration chain because nothing has shipped yet. For the first
real migration:

1. add a numbered SQL file under `app/database/`;
2. record the applied version in a `schema_version` table;
3. apply pending files in order at start-up;
4. never rewrite history — add a column, backfill, and leave old rows resolvable.

Given rule 1 (nothing is updated or deleted), most schema changes are additive.

## Backup and retention

The database is a single file. Back it up by copying it while the application is stopped,
or with `sqlite3 data/inspection.db ".backup data/backup.db"` while it is running.

From the Phase 2 report:

- roughly **1 KB per inspection** including regions — about 30 MB a day at one part per
  second. Older months are detached into archive files.
- **images**: overlays and source images are kept 48 hours when the result is clean and
  90 days when regions were found — around 60 GB in steady state, which fits a 1 TB SSD.
- **a separate machine** can copy each unit's database file and record log on a
  schedule. The line never waits for it.

Retention is not automated in this build. It is a scheduled job over
`data/sources`, `data/overlays` and the `captured_at` column, and it must delete images
only — never inspection rows, which are the audit trail.

## Reading the data outside the application

```bash
sqlite3 data/inspection.db
```

```sql
-- everything needed to explain one inspection
SELECT i.inspection_id, i.captured_at, i.product_id, i.status,
       m.material_code, s.station_code,
       mv.file_name, mv.artefact_sha256,
       p.version_no, p.crack_threshold, p.scratch_threshold
FROM inspection i
LEFT JOIN material m       ON m.material_id = i.material_id
LEFT JOIN station s        ON s.station_id = i.station_id
LEFT JOIN model_version mv ON mv.model_version_id = i.model_version_id
LEFT JOIN profile p        ON p.profile_id = i.profile_id
WHERE i.inspection_id = 42;

-- totals that must reconcile with the batch screen
SELECT status, COUNT(*) FROM inspection WHERE batch_run_id = 1 GROUP BY status;
```

## Image paths

`source_image_path` and `overlay_image_path` are stored **relative to the data root**
(`overlays/mock/live-00001_ovl.png`), not as absolute paths. An absolute path would pin
the database to the machine that built it, and a database restored into a different
folder would lose every image.

At read time the path is resolved against the writable root first, then the read-only
bundled root, and the result is checked to be inside a configured media root before
anything is opened — the database is trusted, but not blindly.
