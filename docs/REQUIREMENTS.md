# Factory Surface-Defect Inspection — Requirements

## Purpose

Factory end-of-line inspection is repetitive, inconsistent, and can both miss defects and cause costly over-rejection. This project is an offline visual-assistance tool: it identifies surface defects, draws a pixel-level mask, and reports geometry for an operator to review. It is not an automatic accept/reject authority; customer-specific tolerances and process context are required for that decision.

## Users and scope

The primary users are Indian MSMEs in sheet-metal fabrication, machining/casting, plastic moulding and ceramic/tile production — the Rajkot and Coimbatore foundry clusters, Morbi's 459 tile units, Ludhiana and Firozabad. Commercial vision lines are often unaffordable and these factories frequently lack reliable connectivity or ML staff. Larger factories can use the same system as a low-cost second-check station.

A fixed camera over a conveyor is an *assumption this document now makes*, and it is worth being explicit about the cost: it excludes factories that cannot mount a station, and it means the tool inspects what passes under the camera rather than whatever an operator points at. That was judged the right trade — controlled geometry is what makes 1–3 px defects measurable at all — but it narrows who can use this.

**Capture scope narrowed (see `ARCHITECTURE.md` §5).** This document originally covered three capture devices — borescope, webcam and phone. It now covers **one**: a fixed industrial camera above a conveyor at constant working distance under controlled illumination. The change is a narrowing of claims, not a loss of capability; handheld capture was never validated on real handheld captures, and a requirement that is asserted rather than measured is worse than an absent one. Handheld modes return only after a device-validation study on real images from those devices.

In scope: still images or single video frames from a **fixed conveyor-mounted camera**; pixel masks; crack/scratch type; per-region area, length and maximum width; overlay; batch CSV; and local audit records. Out of scope: accept/reject verdicts, severity grading, root-cause diagnosis, 3-D measurement, video tracking, and **handheld/borescope/phone capture**.

## Functional requirements

| ID | Requirement | Priority |
|---|---|---|
| FR-01 | Accept PNG/JPEG still images or decoded video frames. | Must |
| FR-02 | Produce a pixel mask for each surface defect. | Must |
| FR-03 | Label each region as `crack` or `scratch`. | Must |
| FR-04 | Report area, centreline length and maximum width per region. | Must |
| FR-05 | Write an overlay on the source image. | Must |
| FR-06 | Return an empty result rather than inventing a defect for a clean surface. | Must |
| FR-07 | Process a directory and emit a CSV report. | Should |
| FR-08 | Log model version, input hash and output for traceability. | Should |
| FR-09 | Require human review before pseudo-labelled images enter training. | Must |

## Non-functional requirements

| ID | Target | Rationale |
|---|---|---|
| NFR-01 | ≤6 MB float32; target ≤2 MB int8 | Suitable for app distribution and low-end phones. |
| NFR-02 | ≤60 ms laptop at 256×256 | Keeps pace with a line. The phone/ARM target is withdrawn with NFR-04; no ARM measurement was ever taken, so no ARM claim is made. |
| NFR-03 | False-positive area ≤0.5% on clean evaluation images | Over-flagging causes the factory losses this project addresses. |
| ~~NFR-04~~ | ~~One model for borescope, webcam and phone~~ **Withdrawn** | Superseded by the fixed-camera architecture. It required generalisation across three optical systems the corpus contains almost no real examples of, so it could only ever have been claimed, not demonstrated. One model still covers all **materials** on the fixed station — that part of the intent survives as NFR-08. |
| NFR-08 | One model across all supported materials on the fixed station | Replaces the surviving half of NFR-04: MSMEs cannot maintain per-material models. |
| NFR-05 | Fully offline inference | Protects factory IP and works without reliable connectivity. |
| NFR-06 | Deterministic fixed-model/config output | Enables auditing and regression tests. |
| NFR-07 | Frozen, real-image, leakage-tested evaluation | Prevents duplicate/synthetic data from overstating performance. |

## Input assumptions and constraints

- The surface fills most of the frame, is reasonably focused, and has one dominant material.
- Inputs normalise to 256×256; image/mask alignment is mandatory.
- The camera is **fixed, rectilinear and calibrated**: a machine-vision lens on an inspection station. Barrel distortion is therefore *not* modelled — simulating fisheye would train the model to undo a distortion the deployment optics do not have.
- The station's realistic artefacts are **belt-axis motion blur** (the part moves along one fixed axis, so smear direction is a property of the installation, not a random variable), **specular highlights** from the LED bar off polished steel or glazed ceramic, ring-light falloff, sensor noise and JPEG. These are what `bench/camera_aug.py --camera-profile conveyor` models.
- **Non-steel metal performance must be marked unsupported**: the corpus contains zero real non-steel-metal crack masks.
- **Plastic rests on one product.** All 936 real plastic masks are PVC pipe from a single source; plastic generalisation beyond that product is unevidenced.
- **Glass is not supported.** 12 real masks is not a measurement at any mixing ratio.
- Scratch evidence is steel-dominated (624 of 724 masks). Scratch claims on ceramic, plastic or glass rest on transfer and must say so.
- Confidence must not be displayed as a probability before calibration is implemented and validated.
