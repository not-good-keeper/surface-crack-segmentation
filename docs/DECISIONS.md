# Engineering Decision Log

## ADR-001 — Segmentation first

Use segmentation as the primary output and defer a detector gate. The operator needs boundaries and geometry; a detector alone cannot provide them. Add detection only if measured line throughput requires it.

## ADR-002 — Factory-first claim, civil auxiliary evidence

Factory surface inspection is the target; civil materials are auxiliary pre-training and transfer evidence. This avoids claiming factory generalisation from predominantly civil crack data.

## ADR-003 — Pretrained mobile U-Net candidate

The binary deployment baseline is MobileNetV3-Small plus slim U-Net: 1.43 M parameters, 5.8 MB float32 ONNX and 26.1 ms desktop single-core ONNX. Smaller from-scratch models had 10–15× worse FP area, unacceptable for QC.

**Three-class performance is now measured** on the same topology (splits v7, `c0fde17c96749567`): `test_factory` clDice 0.665, IoU 0.456, detection 0.962; clean-surface FP area 0.447 %. The encoder/decoder was not re-selected for the three-class head — only the final 1×1 convolution changed — so ADR-003's original comparison stands.

## ADR-004 — Metrics include false-positive area

Use clDice, tolerant F1, IoU and FP area. Pixel accuracy can be gamed by background; IoU alone is harsh for 1–3 px structures; FP area represents over-rejection risk.

## ADR-005 — Frozen, leakage-tested evaluation

Use parent-group/perceptual-hash splits and QA gating. A prior 1,492-row unseen-validation leak passed QA, so v4 explicitly asserts that the unseen-validation material is absent from train, val and test_seen.

## ADR-006 — Masonry validation, wood final transfer test

Masonry is held out for unseen validation and wood for final transfer. Holding plaster out removed 1,492 positives and reduced unseen-wood clDice to 0.444 ± 0.055; v4 costs 192 positives and measured 0.562 ± 0.055.

## ADR-007 — Reject invalid masks

Reject `cracktree` zero-width masks, wood knot outlines and wide MVTec anomaly blobs as segmentation targets. Width audit and contact sheets catch semantic errors that structural checks miss.

## ADR-008 — Mask-safe online camera augmentation

Use online augmentation: geometric changes apply to image and mask, photometric changes to image only.

**Superseded in part by ADR-011.** `camera_aug.py` now carries two profiles. `conveyor` is the default and deliberately *excludes* barrel distortion and rolling-shutter shear; `handheld` retains them so pre-v5 benchmarks stay reproducible. Neither profile has been validated against real device captures, which is why the handheld modes are not claimed as supported.

## ADR-011 — Model the deployment station, not the generic camera

The fixed-camera reframe narrows what augmentation should simulate, and narrowing is the point. Barrel distortion is **removed** from the default profile: a machine-vision lens on an inspection station is selected and calibrated to be rectilinear, so simulating fisheye would train the model to undo a distortion the deployment optics do not have.

The budget saved is spent on two artefacts that actually dominate a line. **Belt-axis motion blur**: a part on a conveyor moves along one fixed axis under a fixed shutter, so smear direction is a property of the installation, not a random variable — randomising it would teach the model that defect orientation and blur orientation are independent, which on a real line they are not. **Specular highlights**: a blown-out reflection of the LED bar off polished steel or glazed ceramic is bright, thin and directional, which is also the description of a scratch. It is the single most important negative artefact for our priority material, and it is photometric — it adds light, not a defect, so the mask must not follow it.

Consequence for requirements: NFR-04 (one model across borescope, webcam and phone) is **withdrawn**. See `REQUIREMENTS.md`.

## ADR-012 — Rebalance the scratch class at the sampler, never in the loss

At its natural ~6 % frequency the scratch class was never predicted on a single pixel — recall 0.000 — while cracks scored 0.53 clDice. Quota-sampling scratch to 35 % of defect draws took scratch clDice to 0.88.

The alternative fix, raising the scratch weight in the loss, was rejected rather than untried. Loss weighting buys minority-class recall by making the model paint more of that class, and over-painting a clean surface is the over-rejection failure this system exists to prevent — the same failure arrived at from the opposite direction. Class weights are still used, but **capped** (`--class-weight-cap`, default 6.0) for exactly this reason, and the cap is a CLI argument because the right value is an empirical question rather than a constant.

## ADR-013 — Synthetic kinds are selected per source, not globally

Our compositor produces cracks and crack-like distractors; the DefectForge engine is the only synthetic **scratch** source. Mixing the two synthetic *crack* domains was measured five ways under the binary head and consistently reduced cross-material clDice.

`--synth-kinds` could not express that: it filtered the concatenated frame by kind, so requesting `crack,scratch,negative` across both directories silently admitted 27,295 DefectForge cracks — 30 % of every synthetic crack the model saw, in direct contradiction of the recipe's own documented intent. Selection is now per source (`--synth data/synth:crack|negative,data/df_patches2:scratch`).

The lesson generalises past this bug: a configuration flag whose granularity does not match the decision it encodes will silently encode a different decision.

## ADR-014 — Real:synthetic mixing ratio is a transfer decision, not a headline decision

`--synth-frac` moved from 0.45 to 0.25 once v7 gave plastic 936 and steel 685 real crack masks; at 0.45 the batch was 55 % rendered when real data existed.

Measured against a same-splits control, the effect on the headline is **within seed noise** (+0.016 clDice, +0.010 IoU). The unambiguous gain is unseen-material transfer: wood clDice 0.737 vs 0.559 and detection 0.952 vs 0.801.

Recorded because the tempting summary — "more real data raised the headline" — is not what the control shows.

**Correction to an earlier reading of this run.** Watching epochs 1–3 arrive live, the control looked like it peaked at epoch 2 and decayed while the real-heavy arm kept climbing, and that story was written here. The completed histories do not support it: the control dipped at epoch 3 and then reached its best value at epoch 4, exactly like both arm-A seeds.

    V8A_11  0.6519  0.6565  0.6646  0.6651   best = ep4
    V8A_22  0.6341  0.6521  0.6395  0.6887   best = ep4
    V8B_11  0.6197  0.6488  0.6036  0.6493   best = ep4

The real finding is ADR-017's, not a difference between arms: **every run is still improving when it runs out of epochs.**

## ADR-017 — The 4-epoch cap is an artefact of an older recipe; the v7 runs are under-trained

The short schedule was inherited from the v5/v6 rounds, where the factory headline peaked at epoch 1–2 and then decayed by ~0.06 clDice while `val` kept climbing. Capping the run was the correct response *to that behaviour*.

Under splits v7 the behaviour is gone. All three completed runs — both recipes — reached their best headline at the **final** epoch, with no decay in between beyond one seed's mid-run dip. Best-epoch restoration was therefore never selecting anything but the last epoch, which means the cap was not protecting against overfitting; it was just stopping early.

The most likely reason is that v7 roughly doubled real factory-material coverage (plastic 0 → 936, steel 300 → 685) while the v8 recipe cut the synthetic share of a batch from 45 % to 25 %. Both changes push the effective training distribution toward real images, and the early peak was a symptom of fitting the rendered domain quickly and then drifting off real surfaces.

This was ruled in from the stored per-epoch histories at no compute cost. It is recorded because the alternative — leaving the cap in place because it was once measured — is how a setting outlives the evidence for it.

Roboflow imports pass three gates: per-class trust, a **15 % foreground** ceiling, and a **20 px** area-to-skeleton ceiling. Contact-sheet review of the first import found annotators who had outlined the entire phone rather than the crack across it, and a source whose "photographs" were line-art schematics. Roughly 15 % of candidate rows and two entire sources failed.

A dataset being good as a whole does not make every class in it usable: `pipe-crack-detection` pairs a real `PVC pipe crack` class with `Paper crack` and `Dummy crack`, and importing all three would teach the model that paper is a factory defect.

## ADR-018 — The input transform is a sensitivity knob, and NFR-03 picks the setting

`bench/prep_sweep.py` swept 22 configurations — 13 single transforms and 9 deliberately
ordered pairs — on `val`, then applied the winners to the frozen splits. Every denoiser
in the family moves the same four numbers together: detection, crack class recall,
unseen-material response and false-positive area all rise as one. The choice is not
"which transform is best" but which point on that curve stays inside the requirement.

`median3` scored the best headline (clDice +0.016, crack recall 0.429 → 0.460) and was
**rejected**: it pushes `fp_area` on `test_negatives` to 0.0061, through NFR-03's 0.005
ceiling that v9 had only just met. `bilateral` (d=5, σ=50/50) takes clDice +0.007,
detection 0.949 → 0.962, and unseen-material clDice +0.088 with detection 0.436 → 0.584,
at `fp_area` 0.0040 — inside the ceiling. That is the shipped setting.

Two findings worth keeping. **CLAHE, the reflex choice for surface inspection, is the
second-worst option measured** (−0.057 at clip 2, −0.108 at clip 4): it amplifies local
contrast on clean surface texture into crack-like evidence the model then believes.
And **no complementary pair exists** — every pair landed at or below its weaker
component, `clahe2+blackhat` at −0.21 against −0.057 and −0.051 alone. These ops all
manipulate local contrast, so stacking them compounds one distortion rather than
addressing independent nuisances.

Selection is on `val` and never on a test split, and this round shows why: two of the
four val winners lost on test, `flatten` by 0.084. The transform is stamped into the
ONNX graph's `metadata_props` so `app/inference.py` applies it without being told, and a
checkpoint trained with a different one cannot silently mismatch the deployment path.

## ADR-019 — A long run needs a schedule, or best-epoch selection becomes a lottery

At a constant 3e-4 the headline swung ~0.06 between adjacent epochs while `val` stayed
flat — the optimiser orbiting a minimum rather than converging. That is tolerable over
20 epochs and corrosive over 80: picking the best of 80 draws from a noisy series is
selecting on the test split, and the reported headline would be biased upward by
whichever epoch got lucky.

`--cosine` decays to 2 % of the initial rate across the run, so the tail is stable and
the final epoch is a defensible choice rather than a lottery win. It is opt-in; every
stored run predates it and stays reproducible.

## ADR-016 — The headline split must be reported per material

`test_factory` is not one population. Image-weighted clDice is 0.651 while material-weighted is 0.525, and the gap is plastic and epoxy carrying steel — the priority material, and the worst performer at 0.529.

Crack **class** recall tracks real crack training data per material almost monotonically: steel 0.702 (1,047 masks), plastic 0.657 (749), ceramic 0.315 (120), epoxy 0.279 (**0**). Since headline clDice and IoU are class-agnostic, none of this is visible in the headline. `bench/per_material.py` is therefore a reporting requirement (TC-13), not a diagnostic convenience.

## ADR-009 — Synthetic/pseudo labels are train-only

DefectForge and approved SAM2 pseudo-labels never enter evaluation. This preserves real-image evidence.

## ADR-010 — Do not overclaim readiness

Desktop latency is only a proxy; confidence is uncalibrated. These limits belong in the demo and paper.

Current honest position, superseding the original coverage note: **plastic** has 936 real masks but all from one PVC-pipe source, so the material is not covered, one product is. **Non-steel metal** remains at zero real crack masks. **Glass** has 12, which is not a measurement. **Crack/scratch typing is only partially working** — 0.46 crack class recall on the headline split — so FR-03 must be reported as partially met rather than delivered.
