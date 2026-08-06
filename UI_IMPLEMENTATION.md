# Industrial Surface-Defect Inspection System — Complete UI Implementation Specification

**Team:** VISION 404  
**Project:** Industrial Surface-Defect Inspection System  
**Purpose:** Build the complete local web user interface and supporting application around the existing/future ONNX inspection model.

---

# 1. Project Authority

You are the Lead Full-Stack Engineer, UI/UX Engineer, Backend Engineer, Database Engineer, Test Engineer, and long-term maintainer of this repository.

Your responsibility is to build the complete user-interface application for the Industrial Surface-Defect Inspection System.

The following materials are authoritative:

1. `CV_Hackathon - Phase 2.pdf`
2. `INTEGRATION.md`
3. This file: `UI_IMPLEMENTATION.md`
4. The existing repository and its current folder structure

Read all of them completely before modifying the repository.

The Phase 2 PDF is the source of truth for screen layouts, navigation, visual hierarchy, operator-facing information, result states, database intent, and testing expectations.

`INTEGRATION.md` is the source of truth for model invocation, ONNX input/output contract, preprocessing, threshold behaviour, region geometry, output records, and known model limitations.

Do not rewrite the architecture. Do not create another design proposal. Do not stop after creating plans, wireframes, documentation, or mockups. Implement the complete working application.

---

# 2. Main Goal

Build a complete responsive local web application that visualises, stores, reviews, filters, and exports surface-defect inspection results.

The application must work before the final ONNX model is available by using realistic deterministic mock data. It must also contain a clean integration boundary so the mock provider can later be replaced with the existing `app/inference.py::Inspector` implementation without redesigning the application.

The finished application must:

- Run locally
- Work without internet access
- Require no cloud service
- Require no frontend build process
- Use responsive layouts suitable for laptop and phone
- Display realistic mock inspections
- Store inspection history in SQLite
- Support mock batch processing
- Export CSV and JSON
- Display material coverage, thresholds, and model metadata
- Display camera, model, database, and disk status
- Be ready for direct integration with `app/inference.py`
- Preserve the exact preprocessing and post-processing path used during evaluation

---

# 3. Frozen Technology Stack

## 3.1 Backend

Use:

- Python 3.11
- FastAPI
- Uvicorn
- Jinja2
- SQLite through Python `sqlite3`
- Pydantic

## 3.2 Frontend

Use:

- Server-rendered HTML
- Plain CSS
- Plain JavaScript

Do not use React, Vue, Angular, a Tailwind build process, npm packages, external CDNs, external web fonts, or cloud-hosted assets. Store all CSS, JavaScript, icons, images, and templates locally.

## 3.3 Testing

Use:

- pytest
- FastAPI TestClient
- Playwright when browser automation is available
- Ruff
- mypy or equivalent type checking

The application should run through one of these commands:

```bash
python -m app.main
```

or:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

---

# 4. Frozen Product Decisions

## 4.1 Model Output

The segmentation model produces:

```text
0 = background
1 = crack
2 = scratch
```

Do not confuse the class channels with RGB channels.

## 4.2 Primary UI Evidence

The interface must show:

- Original or overlay image
- Pixel-level defect overlay
- Region count
- Region class
- Area
- Centreline length
- Maximum width
- Bounding box
- Centroid where available
- Product ID
- Station ID
- Material
- Capture timestamp
- Model version
- Model SHA-256
- Active threshold/profile version
- Inspection status
- Processing latency

## 4.3 Do Not Display Confidence Percentages

Model scores are not calibrated probabilities. Do not display phrases such as:

```text
87% confidence
Confidence: 0.87
Probability of crack
```

Internal scores may be stored or used by the pipeline, but they must not be represented to the operator as probabilities.

## 4.4 Current Result States

The current Phase 2 interface uses:

```text
REGIONS FOUND
NO DEFECTS FOUND
COULD NOT PROCESS
CHECK STATION
```

Do not add automatic Accept or Reject verdicts unless the existing repository already implements and requires them. The application reports what was found and provides evidence for human review.

## 4.5 Clean and Failed-to-Process Must Remain Different

A clean result must resemble:

```json
{
  "status": "clean",
  "empty": true,
  "regions": []
}
```

An unreadable image or processing error must use a different state:

```json
{
  "status": "acquisition_failure",
  "empty": false,
  "regions": [],
  "error_code": "image_decode_failed"
}
```

Never display a failed capture as `NO DEFECTS FOUND`.

## 4.6 Defect Type Is Secondary to the Mask

Current localisation performance is stronger than crack-versus-scratch class typing. The interface should treat the mask as the primary evidence.

Display a small notice where appropriate:

```text
Class label is provisional — inspect the highlighted region.
```

---

# 5. Model Integration Boundary

The frontend and application services must never call ONNX Runtime directly.

Create an abstraction similar to:

```python
from typing import Protocol

class InspectionProvider(Protocol):
    def inspect(
        self,
        image_bgr,
        image_bytes: bytes,
        product_id: str,
        material: str,
    ):
        ...
```

Create two implementations:

```text
MockInspectionProvider
RealInspectionProvider
```

---

# 6. Mock Inspection Provider

Implement `MockInspectionProvider` now.

It must:

- Return realistic inspection records
- Return source-resolution overlays
- Return crack and scratch regions
- Sometimes return clean results
- Sometimes return acquisition failures
- Sometimes return processing failures
- Be deterministic for the same seed/input
- Generate records for steel, ceramic, plastic, glass, epoxy, and non-steel metal
- Mark unsupported and limited materials clearly
- Support demo mode and batch mode
- Create or load local placeholder source images and overlays
- Never download copyrighted images automatically

Use generated industrial-style surfaces and procedural defect overlays.

---

# 7. Real Inspection Provider

Create an integration-ready `RealInspectionProvider`, but do not require the final model file to exist in mock mode.

It will later call:

```python
from app.inference import Inspector

insp = Inspector(
    "data/export/model.onnx",
    station_id="line-1-cam-A",
)

record, overlay_bgr, class_map = insp.inspect(
    image_bgr,
    image_bytes,
    product_id="batch-77/item-12",
    material="steel",
)
```

The real provider must:

- Import `Inspector` only when real mode is enabled
- Fail with a clear startup message if the model is missing
- Preserve the source-resolution overlay
- Convert the returned record into the application schema
- Never reimplement preprocessing
- Never reimplement thresholding
- Never call ONNX Runtime directly
- Never duplicate thresholds inside the frontend
- Verify the configured model SHA-256
- Keep the UI independent of future weight changes

Use environment variables:

```env
INSPECTION_PROVIDER=mock
MODEL_PATH=data/export/model.onnx
STATION_ID=line-1-cam-A
```

Default to:

```env
INSPECTION_PROVIDER=mock
```

---

# 8. Fixed ONNX Contract

Do not implement a second preprocessing pipeline in the UI.

The real adapter must preserve this contract:

| Property | Value |
|---|---|
| Input name | `input` |
| Input shape | `(1, 3, 256, 256)` float32 |
| Batch axis | Dynamic |
| Channel order | RGB |
| Resize | Plain resize to 256×256 with `INTER_LINEAR` |
| Scaling | Divide by 255 |
| Mean | `[0.485, 0.456, 0.406]` |
| Standard deviation | `[0.229, 0.224, 0.225]` |
| Layout | CHW |
| Output name | `logits` |
| Output shape | `(1, 3, 256, 256)` float32 |
| Output type | Raw logits |
| Activation | Softmax outside the graph |

Foreground classes use separate thresholds. Do not replace the existing rule with plain `argmax`.

Threshold values belong to `app/postprocess.py::Profile`, not HTML, JavaScript, or duplicated UI configuration.

---

# 9. Recommended Repository Structure

Respect and reuse the existing repository. Do not duplicate existing working modules.

```text
app/
├── __init__.py
├── main.py
├── config.py
├── dependencies.py
│
├── providers/
│   ├── __init__.py
│   ├── base.py
│   ├── mock_provider.py
│   └── real_provider.py
│
├── services/
│   ├── inspection_service.py
│   ├── batch_service.py
│   ├── history_service.py
│   ├── material_service.py
│   ├── status_service.py
│   └── export_service.py
│
├── repositories/
│   ├── inspection_repository.py
│   ├── material_repository.py
│   ├── model_repository.py
│   └── batch_repository.py
│
├── database/
│   ├── connection.py
│   ├── schema.sql
│   ├── migrations.py
│   └── seed.py
│
├── schemas/
│   ├── inspection.py
│   ├── region.py
│   ├── batch.py
│   ├── material.py
│   └── health.py
│
├── routes/
│   ├── pages.py
│   ├── api_inspections.py
│   ├── api_batches.py
│   ├── api_materials.py
│   └── api_status.py
│
├── templates/
│   ├── base.html
│   ├── live.html
│   ├── region_detail.html
│   ├── batch.html
│   ├── history.html
│   ├── materials.html
│   ├── status.html
│   ├── inspection_detail.html
│   └── partials/
│       ├── navigation.html
│       ├── health_strip.html
│       ├── result_banner.html
│       ├── region_table.html
│       └── pagination.html
│
├── static/
│   ├── css/
│   │   ├── base.css
│   │   ├── layout.css
│   │   ├── components.css
│   │   └── responsive.css
│   ├── js/
│   │   ├── live.js
│   │   ├── region-detail.js
│   │   ├── batch.js
│   │   └── history.js
│   ├── icons/
│   └── demo/
│
└── exports/

tests/
├── test_routes.py
├── test_mock_provider.py
├── test_inspection_repository.py
├── test_batch_service.py
├── test_exports.py
└── test_ui.py
```

---

# 10. SQLite Database

Use SQLite and enable foreign keys.

Create these tables.

## 10.1 `material`

```text
material_id
material_code
material_name
support_status
notes
```

Support-status values:

```text
supported
one_product_only
thin_coverage
typing_unsupported
not_supported
under_evaluation
```

## 10.2 `defect_class`

```text
class_id
class_code
display_name
```

Seed `crack` and `scratch`.

## 10.3 `model_version`

```text
model_version_id
file_name
version
artefact_sha256
parameter_count
size_mb
precision
latency_ms
created_at
is_active
```

## 10.4 `station`

```text
station_id
station_code
line_code
mm_per_pixel
camera_status
created_at
```

## 10.5 `profile`

```text
profile_id
material_id
version_no
crack_threshold
scratch_threshold
minimum_area_px
minimum_skeleton_px
created_at
is_active
```

The frontend must not contain a duplicate set of threshold values.

## 10.6 `batch_run`

```text
batch_run_id
source_folder
material_id
started_at
finished_at
image_count
clean_count
regions_found_count
failure_count
status
```

## 10.7 `inspection`

```text
inspection_id
station_id
profile_id
model_version_id
material_id
batch_run_id
image_sha256
product_id
captured_at
processed_at
status
region_count
latency_ms
source_image_path
overlay_image_path
error_code
error_message
```

Allowed statuses:

```text
regions_found
clean
acquisition_failure
processing_failure
```

## 10.8 `defect_region`

```text
region_id
inspection_id
region_index
class_id
area_px
length_px
max_width_px
bbox_x
bbox_y
bbox_width
bbox_height
centroid_x
centroid_y
```

Use indexes for capture time, product ID, status, material, station, batch, and inspection-region relationships. Do not store all regions as a JSON array inside `inspection`. Create deterministic seed data for demonstration.

---

# 11. Global Visual Style

Follow the wireframes in the Phase 2 PDF.

The interface must look like an industrial quality-control application.

Use:

- Neutral white and grey palette
- Strong, clear borders
- Compact but readable tables
- Large result headings
- Minimal decoration
- Consistent alignment
- Clear information hierarchy
- Fixed left navigation on desktop
- Responsive top navigation or drawer on mobile
- Persistent station label
- Persistent health strip
- Accessible contrast
- Large touch targets
- Local system fonts
- Minimal animation

Avoid glassmorphism, generic startup-dashboard styling, decorative gradients, unnecessary charts, excessive cards, external imagery, tiny grey text, and hover-only actions.

Create reusable components for navigation, health strip, page header, result banner, summary cards, data tables, filter bars, status badges, empty states, error states, pagination, image viewer, and region navigation.

---

# 12. Persistent Navigation

Desktop navigation:

```text
Live
Regions
Batch
History
Materials
Status
```

Requirements:

- Highlight the current screen
- Display the station code
- Keyboard accessible
- Responsive on narrow screens
- No hover-only actions
- Minimum touch target around 44 px
- Each main screen reachable in one interaction

---

# 13. Persistent Health Strip

Display on primary screens:

```text
Camera status
Model version and abbreviated hash
Active material/profile
Database status
Disk usage
Last inspection latency
```

Example:

```text
Camera OK | Model v8a · SHA 9f2c… | Profile: steel | DB OK | Disk 71% | 26 ms
```

If a check fails, display the specific failure, link to Status, stop presenting the station as healthy, and do not silently show a normal inspection state.

---

# 14. Screen 1 — Live Inspection

Match the Phase 2 wireframe.

## Desktop Layout

- Large source-resolution overlay on the left
- Result summary on the right
- Region table below the summary
- Product metadata below the image
- Buttons for region detail and history

## Result States

### Regions Found

```text
3 REGIONS FOUND
2 crack, 1 scratch
```

Do not call this a rejection.

### Clean

```text
NO DEFECTS FOUND
```

### Could Not Process

```text
COULD NOT PROCESS
```

Show the reason.

### Station Problem

```text
CHECK STATION
```

Show the failing check.

## Region Table

Columns:

```text
#
Class
Length
Maximum width
Area
```

Requirements:

- Open the selected region
- Show the provisional-class notice
- Never show confidence percentages

## Demo Updating

Mock mode should refresh every 2–3 seconds. When `DEMO_MODE=true`, provide a developer control to advance to the next mock inspection.

---

# 15. Screen 2 — Region Detail

Use the three-column layout.

## Left

- Region list
- Selected region highlight
- Previous/Next
- Region number, class, and length

## Centre

- Enlarged crop
- Overlay/original/side-by-side toggle
- Zoom controls
- Correct aspect ratio

## Right

Show:

```text
Class
Area
Length
Maximum width
Bounding box
Centroid
```

Also show image SHA-256, model file and version, model hash, profile version, crack threshold, scratch threshold, minimum area, minimum skeleton length, capture time, product ID, and station ID.

Include:

```text
Maximum width is the widest inscribable point measured along the region skeleton. It may differ from the widest visual extent.
```

Thresholds are read-only.

---

# 16. Screen 3 — Batch Run and Report

Implement a working mock batch workflow.

## Configuration

Inputs:

- Source directory or safe configured demo directory
- Material
- Active model/profile display
- Optional product/batch prefix
- Dry run
- Start

Do not permit arbitrary path traversal.

Safe options:

- Select configured server directories
- Upload ZIP
- Select a subdirectory within a configured batch root

## Progress

Show processed/total, percentage, estimated time remaining, current file, and failure count.

## Summary

Show:

```text
Processed
With regions
Clean
Failed to read
Regions found
```

## Results

Columns:

```text
Image
Product ID
Material
Status
Regions
Classes
Maximum length
Processing time
```

## Export

Implement CSV, JSON, images-with-regions filter, clean filter, failure filter, and inspection-detail links. Displayed totals and exported totals must use the same query.

---

# 17. Screen 4 — Inspection History

Implement filtering and pagination.

Desktop filters:

```text
Date from
Date to
Material
Status
Class
Station
Product ID
Apply
Clear
```

Mobile filters may collapse vertically.

Columns:

```text
Timestamp
Product ID
Material
Status
Region count
Maximum length
Model version
Station
```

Requirements:

- Pagination
- Empty state
- Export filtered CSV
- Open inspection detail
- Preserve filters when navigating back
- Store filters in query parameters
- Support combined filters

Inspection detail must include original image, overlay, regions, geometry, model, profile, status, and errors.

---

# 18. Screen 5 — Materials and Thresholds

Read-only screen.

## Material Coverage Table

Columns:

```text
Material
Status
Training masks
Class typing
Notes
```

Load values from a backend JSON file or database table. Do not hardcode them in templates.

Include:

```text
Steel
Plastic
Ceramic
Epoxy
Glass
Non-steel metal
```

Clearly communicate supported, one product only, thin coverage, typing unsupported, and not supported. Painted-metal/non-steel-metal status must remain clearly limited unless actual evidence is supplied.

## Threshold Panel

Display:

```text
crack_thresh
scratch_thresh
min_area_px
min_skeleton_px
```

Include:

```text
Read from the active model profile — not copied into the UI.
```

## Model Panel

Show model filename, version, SHA-256, parameter count, file size, precision, measured latency, input size, and output taxonomy.

Include:

```text
No ARM or phone latency measurement is currently claimed.
```

---

# 19. Screen 6 — System Status

Primary checks:

```text
Camera
Model file/hash
Database
Disk space
```

Optional:

```text
Inference provider
Last inspection
Overlay folder
Batch folder
```

Each status card must show state, last checked, detail, and recommended action.

Provide:

```text
Run checks again
```

Mock mode must be able to simulate success and failure. Real mode must block inspection if the model hash does not match.

---

# 20. Responsive Design

Test at:

```text
1440 px
1024 px
768 px
390 px
360 px
```

On small screens:

- Replace fixed sidebar with top navigation or drawer
- Stack image and summary
- Keep result status visible near the top
- Allow tables to scroll horizontally
- Maintain readable font sizes
- Use 44 px touch targets
- Avoid hover
- Keep region navigation easy to reach

---

# 21. Accessibility

Implement:

- Semantic HTML
- Labels for controls
- Visible keyboard focus
- Logical tab order
- High contrast
- `aria-live` for changing live results
- Text labels in addition to colour
- Alt text
- `<th>` table headers
- Skip-navigation link
- Proper buttons
- Reduced-motion support
- No colour-only meaning

---

# 22. Mock Data

Seed at least:

- 75 inspections
- Multiple dates
- Multiple stations
- Multiple materials
- Clean results
- Single-region results
- Multiple-region results
- Crack-only results
- Scratch-only results
- Mixed crack/scratch results
- Acquisition failures
- Processing failures
- At least three batch runs

Create local placeholder surfaces and overlays. Mock output must be deterministic.

---

# 23. API Endpoints

Implement endpoints similar to:

```text
GET  /api/live
POST /api/demo/next
GET  /api/inspections
GET  /api/inspections/{inspection_id}
GET  /api/inspections/{inspection_id}/regions/{region_index}
POST /api/batches
GET  /api/batches/{batch_run_id}
GET  /api/batches/{batch_run_id}/export.csv
GET  /api/batches/{batch_run_id}/export.json
GET  /api/materials
GET  /api/model
GET  /api/status
POST /api/status/check
```

Use typed schemas. Return structured errors. Keep page routes and API routes in separate modules.

---

# 24. Environment Configuration

Create a typed configuration class.

Provide `.env.example`:

```env
APP_HOST=0.0.0.0
APP_PORT=8000
APP_ENV=development
DEMO_MODE=true
INSPECTION_PROVIDER=mock
DATABASE_PATH=data/inspection.db
MODEL_PATH=data/export/model.onnx
STATION_ID=line-1-cam-A
BATCH_ROOT=data/batches
OVERLAY_ROOT=data/overlays
EXPORT_ROOT=data/exports
MIN_FREE_DISK_GB=5
```

Do not commit secrets.

---

# 25. Testing Requirements

## 25.1 Backend Tests

Test:

- every page loads
- schema creation
- database seeding
- mock-provider determinism
- history filtering
- combined filtering
- pagination
- inspection detail
- region detail
- clean/failure distinction
- batch totals
- CSV export
- JSON export
- model metadata
- material metadata
- status checks
- model-hash validation
- path traversal protection
- invalid image handling

## 25.2 UI Tests

With Playwright where available, test:

- navigation
- current-screen highlighting
- live updates
- region previous/next
- image-mode toggle
- batch progress and summary
- history filtering
- mobile layout
- empty state
- failure state
- persistent status strip
- no confidence percentages
- no operator-facing Accept/Reject verdicts
- no hardcoded thresholds

## 25.3 Static Checks

Search the rendered UI for forbidden phrases:

```text
% confident
confidence:
probability:
```

`accept` and `reject` may appear in technical documentation or tests, but must not appear as present operator-facing model verdicts.

---

# 26. Documentation to Create

## `README.md`

Include purpose, current mock-mode status, installation, run commands, tests, database seed/reset, environment configuration, provider switching, screenshots, known limitations, and model integration.

## `UI_IMPLEMENTATION_NOTES.md`

Include screens, routes, templates, responsive behaviour, accessibility, mock-data generation, and reusable UI components.

## `MODEL_INTEGRATION.md`

Include `InspectionProvider`, mock and real providers, `Inspector.inspect()` contract, model path, switching providers, output mapping, overlay/class-map handling, and the warning against reimplementing preprocessing and thresholds.

## `DATABASE.md`

Include schema, relationships, indexes, seed data, migrations, backup/reset, and retention.

---

# 27. Implementation Order

Follow this order:

1. Inspect the entire repository.
2. Read the Phase 2 PDF.
3. Read `INTEGRATION.md`.
4. Read this implementation specification.
5. Identify reusable code.
6. Create configuration.
7. Create database and seed logic.
8. Create Pydantic schemas.
9. Create provider abstraction.
10. Create deterministic mock provider.
11. Create real-provider adapter.
12. Create repositories.
13. Create services.
14. Create API routes.
15. Create page routes.
16. Create base template.
17. Create navigation and health strip.
18. Build Live Inspection.
19. Build Region Detail.
20. Build Batch Run and Report.
21. Build Inspection History.
22. Build Materials and Thresholds.
23. Build System Status.
24. Add responsive design.
25. Add demo assets.
26. Add exports.
27. Add tests.
28. Run the application.
29. Run tests and fix failures.
30. Capture screenshots.
31. Complete documentation.
32. Provide a final implementation summary.

Do not pause after planning. Do not ask for approval after every stage. Make reasonable implementation decisions while preserving the frozen requirements.

---

# 28. Definition of Done

The implementation is complete only when:

- The application starts without the real model.
- All six screens are implemented.
- The layout follows the Phase 2 wireframes.
- Laptop and phone layouts work.
- Mock live inspections update.
- Region navigation works.
- Batch processing works in mock mode.
- History filtering and pagination work.
- CSV export works.
- JSON export works.
- Materials and thresholds come from backend data.
- Status checks work.
- SQLite persistence works.
- Clean and failed-to-process states remain distinct.
- No confidence percentage is displayed.
- No current Accept/Reject verdict is displayed.
- The real provider adapter exists.
- The UI does not duplicate thresholds.
- Runtime has no internet dependency.
- Tests pass.
- Documentation is complete.

---

# 29. Final Response Required from Claude

After implementation, provide:

1. Concise implementation summary
2. Final folder structure
3. Installation commands
4. Run commands
5. Database seed/reset commands
6. Test commands
7. Implemented screen list
8. Mock/real provider switching explanation
9. Files to update when the final model is ready
10. Known limitations
11. Confirmation of which commands were actually run
12. Confirmation of test results
13. Screenshot locations

Do not respond with another architecture plan. Implement the complete application.
