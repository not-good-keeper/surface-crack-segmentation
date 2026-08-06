-- Industrial Surface-Defect Inspection System - SQLite schema
-- Phase 2 report section 3.7 / UI_IMPLEMENTATION.md section 10.
--
-- Design rule: a stored inspection can always be explained later - which model file,
-- which thresholds, which station and which image.  Nothing is ever UPDATEd or
-- DELETEd; re-running an image writes a new inspection row and re-tuning thresholds
-- writes a new profile version.

PRAGMA foreign_keys = ON;

-- 1. material ---------------------------------------------------------------
CREATE TABLE IF NOT EXISTS material (
    material_id    INTEGER PRIMARY KEY AUTOINCREMENT,
    material_code  TEXT NOT NULL UNIQUE,
    material_name  TEXT NOT NULL,
    support_status TEXT NOT NULL CHECK (support_status IN (
                       'supported', 'one_product_only', 'thin_coverage',
                       'typing_unsupported', 'not_supported', 'under_evaluation')),
    notes          TEXT
);

-- 2. defect_class -----------------------------------------------------------
-- class_id matches the model output channel: 1 = crack, 2 = scratch.
-- Channel 0 (background) is not a defect class and is not stored here.
CREATE TABLE IF NOT EXISTS defect_class (
    class_id     INTEGER PRIMARY KEY,
    class_code   TEXT NOT NULL UNIQUE,
    display_name TEXT NOT NULL
);

-- 3. model_version ----------------------------------------------------------
CREATE TABLE IF NOT EXISTS model_version (
    model_version_id INTEGER PRIMARY KEY AUTOINCREMENT,
    file_name        TEXT NOT NULL,
    version          TEXT NOT NULL,
    artefact_sha256  TEXT NOT NULL UNIQUE,
    parameter_count  INTEGER,
    size_mb          REAL,
    precision        TEXT,
    latency_ms       REAL,
    created_at       TEXT NOT NULL,
    is_active        INTEGER NOT NULL DEFAULT 0 CHECK (is_active IN (0, 1))
);

-- 4. station ----------------------------------------------------------------
CREATE TABLE IF NOT EXISTS station (
    station_id    INTEGER PRIMARY KEY AUTOINCREMENT,
    station_code  TEXT NOT NULL UNIQUE,
    line_code     TEXT NOT NULL,
    mm_per_pixel  REAL,               -- NULL until the station is calibrated
    camera_status TEXT NOT NULL DEFAULT 'ok',
    created_at    TEXT NOT NULL
);

-- 5. profile ----------------------------------------------------------------
-- Mirrors app/postprocess.py::Profile.  Values are written here at seed time by
-- reading the module; they are never typed in by hand and never read by the frontend
-- from anywhere else.
CREATE TABLE IF NOT EXISTS profile (
    profile_id          INTEGER PRIMARY KEY AUTOINCREMENT,
    material_id         INTEGER REFERENCES material (material_id),
    version_no          INTEGER NOT NULL,
    crack_threshold     REAL NOT NULL,
    scratch_threshold   REAL NOT NULL,
    minimum_area_px     INTEGER NOT NULL,
    minimum_skeleton_px INTEGER NOT NULL,
    created_at          TEXT NOT NULL,
    is_active           INTEGER NOT NULL DEFAULT 0 CHECK (is_active IN (0, 1)),
    UNIQUE (material_id, version_no)
);

-- 6. batch_run --------------------------------------------------------------
CREATE TABLE IF NOT EXISTS batch_run (
    batch_run_id        INTEGER PRIMARY KEY AUTOINCREMENT,
    source_folder       TEXT NOT NULL,
    material_id         INTEGER REFERENCES material (material_id),
    started_at          TEXT NOT NULL,
    finished_at         TEXT,
    image_count         INTEGER NOT NULL DEFAULT 0,
    clean_count         INTEGER NOT NULL DEFAULT 0,
    regions_found_count INTEGER NOT NULL DEFAULT 0,
    failure_count       INTEGER NOT NULL DEFAULT 0,
    status              TEXT NOT NULL DEFAULT 'running'
                        CHECK (status IN ('running', 'completed', 'failed', 'dry_run'))
);

-- 7. inspection -------------------------------------------------------------
-- status carries four values.  clean and acquisition_failure / processing_failure are
-- deliberately different: a defect-free product and an unreadable image are not the
-- same outcome and must never collapse into one "no defects" state.
CREATE TABLE IF NOT EXISTS inspection (
    inspection_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    station_id         INTEGER NOT NULL REFERENCES station (station_id),
    profile_id         INTEGER NOT NULL REFERENCES profile (profile_id),
    model_version_id   INTEGER NOT NULL REFERENCES model_version (model_version_id),
    material_id        INTEGER REFERENCES material (material_id),
    batch_run_id       INTEGER REFERENCES batch_run (batch_run_id),
    image_sha256       TEXT,
    product_id         TEXT,
    captured_at        TEXT NOT NULL,
    processed_at       TEXT NOT NULL,
    status             TEXT NOT NULL CHECK (status IN (
                           'regions_found', 'clean',
                           'acquisition_failure', 'processing_failure')),
    region_count       INTEGER NOT NULL DEFAULT 0,
    latency_ms         REAL,
    source_image_path  TEXT,
    overlay_image_path TEXT,
    error_code         TEXT,
    error_message      TEXT
);

-- 8. defect_region ----------------------------------------------------------
CREATE TABLE IF NOT EXISTS defect_region (
    region_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    inspection_id INTEGER NOT NULL REFERENCES inspection (inspection_id) ON DELETE CASCADE,
    region_index  INTEGER NOT NULL,
    class_id      INTEGER NOT NULL REFERENCES defect_class (class_id),
    area_px       INTEGER NOT NULL,
    length_px     REAL NOT NULL,
    max_width_px  REAL NOT NULL,
    bbox_x        INTEGER NOT NULL,
    bbox_y        INTEGER NOT NULL,
    bbox_width    INTEGER NOT NULL,
    bbox_height   INTEGER NOT NULL,
    centroid_x    REAL,
    centroid_y    REAL,
    UNIQUE (inspection_id, region_index)
);

-- Indexes -------------------------------------------------------------------
-- History filtering (FR-18) is on six fields; batch reporting (FR-19) aggregates by
-- batch_run_id; the region tables always join on inspection_id.
CREATE INDEX IF NOT EXISTS idx_inspection_captured_at ON inspection (captured_at DESC);
CREATE INDEX IF NOT EXISTS idx_inspection_product_id  ON inspection (product_id);
CREATE INDEX IF NOT EXISTS idx_inspection_status      ON inspection (status);
CREATE INDEX IF NOT EXISTS idx_inspection_material    ON inspection (material_id);
CREATE INDEX IF NOT EXISTS idx_inspection_station     ON inspection (station_id);
CREATE INDEX IF NOT EXISTS idx_inspection_batch       ON inspection (batch_run_id);
CREATE INDEX IF NOT EXISTS idx_inspection_station_cap ON inspection (station_id, captured_at DESC);
CREATE INDEX IF NOT EXISTS idx_region_inspection      ON defect_region (inspection_id);
CREATE INDEX IF NOT EXISTS idx_region_class           ON defect_region (class_id);
CREATE INDEX IF NOT EXISTS idx_profile_active         ON profile (is_active);
CREATE INDEX IF NOT EXISTS idx_model_active           ON model_version (is_active);
