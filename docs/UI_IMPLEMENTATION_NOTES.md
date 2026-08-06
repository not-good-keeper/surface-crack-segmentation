# UI Implementation Notes

How the interface is put together: screens, routes, templates, responsive behaviour,
accessibility, and how the mock data is generated.

---

## Design position

The visual direction is fixed by the Phase 2 wireframes: an industrial quality-control
application, not a dashboard. Neutral white and grey, strong borders, compact readable
tables, large result headings, minimal decoration, local system fonts. No glassmorphism,
no gradients, no charts that do not carry information, no hover-only actions.

Most of the design follows from three rules about what must **not** appear on screen:

| Rule | Reason |
|---|---|
| No confidence percentage | The scores are uncalibrated. `0.87` rendered as "87 % confident" states a probability the model has never been validated to produce. |
| `clean` ≠ `could not process` | A defect-free product and an unreadable image are different outcomes. One "no defects" state would hide the difference between a good part and a broken camera. |
| The class label is provisional | Typing is about 49 % correct on crack pixels. The mask is the primary evidence; the label is secondary and marked everywhere it appears. |

Beyond those: the interface is read-only, station state is visible on every screen, and
the result has to be readable in about two seconds.

## Routes

### Pages (`app/routes/pages.py`)

| Route | Screen |
|---|---|
| `/` | redirects to `/live` |
| `/live` | Live Inspection |
| `/capture` | Capture — device camera or photo upload |
| `/regions` | Region Detail — `?inspection_id=&region=&mode=&zoom=` |
| `/batch` | Batch Run and Report — `?batch_run_id=&only_with_regions=&status=` |
| `/history` | Inspection History — all filters in the query string |
| `/inspections/{id}` | Inspection detail (opened from history) |
| `/materials` | Materials and Thresholds |
| `/status` | System Status |

### API

| Route | Purpose |
|---|---|
| `GET /api/live` | Current result + health strip |
| `POST /api/demo/next` | Advance the mock station (`?force=` a scenario) |
| `POST /api/inspections/capture` | Inspect one submitted frame — multipart `file`, `material`, `source`, optional `product_id` |
| `GET /api/inspections` | Filtered, paginated listing |
| `GET /api/inspections/{id}` | One inspection with regions and provenance |
| `GET /api/inspections/{id}/regions/{index}` | Region detail with prev/next |
| `GET /api/inspections/export.{csv,json}` | Export the current filter |
| `POST /api/batches` | Start a run (or dry run) |
| `GET /api/batches/{id}` | Report, totals and progress |
| `GET /api/batches/{id}/export.{csv,json}` | Export a run |
| `GET /api/materials`, `/api/thresholds`, `/api/model` | Coverage, thresholds, model metadata |
| `GET /api/status`, `POST /api/status/check` | Checks, with `?simulate=` in mock mode |
| `GET /media/inspection/{id}/{source\|overlay}` | Stored images |
| `GET /media/inspection/{id}/region/{index}` | Cropped region — `?mode=&zoom=` |

**No media route accepts a filesystem path.** A request names an inspection and a role;
the path comes from the database and is checked against the configured media roots before
anything is read. There is no traversal surface to defend.

## Templates

```
base.html                 app bar, sidebar, page header, health strip, skip link
partials/navigation.html  six links, current one marked
partials/health_strip.html camera, model+hash, profile, database, disk, last latency
partials/result_banner.html four states
partials/region_table.html  class, length, width, area — and no score column
partials/pagination.html    prev/next with page numbers, 0 renders as an ellipsis
live.html  region_detail.html  batch.html  history.html  materials.html  status.html
inspection_detail.html  error.html
```

Templates receive assembled view models from `app/dependencies.py`, never raw rows, so no
template decides what a status means. Filters: `px`, `number`, `short_time`,
`status_label`, `breakdown`, `split_last`.

### Reusable components

Navigation, health strip, page header, result banner, summary cards, data tables, filter
bars, status badges, empty states, error states, pagination, image viewer, region
navigation, chips, tags, notes. All in `components.css` against CSS custom properties.

## Screens

**Live Inspection.** Overlay left at source resolution, result right. Banner, region
table, provisional note, buttons to region detail and history. Polls `/api/live` every
2.5 s; with "Advance automatically" ticked it calls `/api/demo/next` instead. A failing
station check replaces the result with **CHECK STATION** and blocks the demo control.

**Capture.** *Not a Phase 2 wireframe screen.* The six wireframes assume a fixed camera
above a conveyor; Capture covers the case with no station at all — a phone or laptop
camera pointed at a part, or a photograph from disk. It is a separate screen precisely so
the Live wireframe stays exactly as drawn.

Both inputs end at the same place: a Blob POSTed to `/api/inspections/capture` as
multipart form data, then the configured provider, the same record mapping and the same
database writer as a station inspection. A capture therefore appears in History, in the
exports and in the batch totals like anything else — there is no second class of result.

Three details are load-bearing:

* **The frame is captured at the sensor's own size**, not the CSS size of the preview.
  Every geometry on the result is in source pixels, so sending a frame scaled to the
  width of a page column would quietly change the units of every measurement.
* **The camera is not started on load.** `getUserMedia` raises a permission prompt, and
  a page that demands the camera before the operator asks for it gets denied once and
  then stays denied for the origin. It also needs HTTPS or localhost; where the API is
  absent the screen says so and points at the file picker rather than failing silently.
* **In mock mode the screen says the regions are generated before showing any.** This is
  the easiest mistake the application could invite: an operator photographs a genuinely
  cracked part, sees regions drawn on their own photograph, and concludes the model found
  them. In mock mode nothing found anything.

**Region Detail.** Three columns: region list with prev/next, enlarged crop with
overlay/original/side-by-side and 1×–6× zoom, and measurements + source panel. The
measurements panel states that maximum width is the widest *inscribable* point. Left and
right arrow keys move between regions; every control is a real link that works without
JavaScript.

**Batch Run and Report.** Configuration (folder, material, prefix, read-only model),
progress bar, five summary cards, results table, exports and filters. Cards, table and
both exports come from the same query, so totals always reconcile. Folders come from a
configured root; anything outside it is refused.

**Inspection History.** All six FR-18 filters on one row, no hidden advanced search.
Filters live in the query string so a view can be bookmarked, shared and returned to with
the back button. Pagination, empty state, filtered CSV/JSON export, row opens the
original evidence.

**Materials and Thresholds.** Read-only. Coverage from the metrics file; thresholds from
`app/postprocess.py` shown read-only with the note that they are not copied into the UI;
model panel including the note that no ARM figure exists.

Thresholds come from `app/profiles.py::Profile`, which `app/postprocess.py` re-exports;
the split exists so the interface can read them where cv2 is not installed.

**System Status.** Seven checks (four primary: camera, model hash, database, disk; plus
provider, last inspection, folders), each with state, detail, recommended action and check
time. "Run checks again" re-runs them. In mock mode a fault-simulation control shows how a
real failure propagates to every screen.

## Responsive behaviour

Verified at 1440, 1024, 768, 390 and 360 px by
`tests/test_ui.py::test_layout_has_no_horizontal_overflow`.

| Breakpoint | Change |
|---|---|
| ≤ 1200 px | Region detail drops to two columns, measurements span the width |
| ≤ 1000 px | Live stacks — **and the result summary moves above the image**, because the operator needs the state before the picture |
| ≤ 900 px | Sidebar becomes a scrollable top bar; region detail stacks viewer → list → measurements |
| ≤ 700 px | Filter bars go vertical, side-by-side becomes stacked, data lists become single-column |
| ≤ 420 px | Tighter spacing, smaller card values, page header stacks |

Tables scroll horizontally rather than shrinking below a readable size. Nothing depends
on hover at any width.

## Accessibility

- Semantic HTML: `<nav>`, `<main>`, `<section aria-label>`, `<table>` with `<th scope>`,
  `<dl>` for measurements, `<fieldset>/<legend>` for control groups.
- Skip-navigation link; one `<h1>` per page; logical tab order.
- Visible focus (`:focus-visible`, 3 px outline) — this is used at a station, often by
  keyboard.
- `aria-live="polite"` on the result banner; `aria-current` on navigation and pagination.
- 44 px minimum touch targets (`--touch`).
- **Nothing is carried by colour alone.** Each result state differs in wording, fill *and*
  border: regions found is a grey banner, clean is white with a dash marker, could-not-
  process is hatched, check-station is dark. Status badges pair colour with a word and a
  border style. Overlay regions are numbered and labelled in text, not just tinted.
- `prefers-reduced-motion` respected; every label has a `<label for>`; every image has alt
  text describing what it shows.

## JavaScript

Five small files, no framework, no build step, no external script.

| File | Job |
|---|---|
| `live.js` | Poll `/api/live`, render the banner and region rows, advance the demo station |
| `region-detail.js` | Arrow-key navigation |
| `batch.js` | Start a run, poll progress, update the cards |
| `history.js` | Drop empty fields so the query string stays readable |
| `status.js` | Re-run checks, drive the fault simulation |

**No threshold, class rule or status wording is computed in the browser.** The scripts
render what the API returns. `tests/test_content_rules.py` fails the build if a threshold
literal appears in a template or a script.

Every screen renders complete on the server: with scripting disabled the interface still
reads, and only live refresh, progress polling and arrow keys are lost.

## Mock data

`app/providers/mock_assets.py` synthesises surfaces from a seeded NumPy generator —
brushed steel with directional machining marks, speckled ceramic, matte plastic and epoxy,
smooth glass — with uneven illumination, because a real station never lights a part
evenly. Defects are polylines: cracks wander with a jagged step, scratches are straighter
and thinner. Geometry is read off the drawn shapes; region floors come from `Profile`, so
a re-tuned floor changes the mock too.

**No image is ever downloaded.** Everything is generated offline, and the same key always
produces the same pixels.

Overlays are drawn at source resolution with numbered, labelled boxes. Crops for region
detail are generated on request from the stored bounding box.

The mock is not a model and its geometry is not a measurement of one. Nothing produced in
mock mode should be quoted as accuracy — the Status screen says so, and `describe()`
repeats it.
