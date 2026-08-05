# Results index — which run supports which claim

Every claim in `DECISIONS.md`, `DATASET.md` and `ARCHITECTURE.md` traces to a stored run
in `data/bench/*.json`. This file is the map. Without it the 70-odd result files are
clutter; with it they are the evidence base, and any number in the documentation can be
checked against the run that produced it.

Each JSON records the full argument set, the per-epoch history, the split version, the
cost measurement and every frozen-split metric. Weights are **not** in git (23
checkpoints is 139 MB) — the JSON beside each one records what it scored and how.

```bash
./.venv/Scripts/python.exe bench/summarize.py --tag V8A      # mean ± std over seeds
./.venv/Scripts/python.exe bench/per_material.py --tag V8A_11 # headline by material
```

Read `wood` as `test_unseen_material` clDice — the transfer metric, and by far the
noisiest number in the project. It swung **0.087 / 0.204 / 0.334** across three seeds of
one identical configuration (`mseed*`), which is why nothing here is quoted from a single
seed.

## Current model

| Tag | Script | Seeds | What it is |
|---|---|---|---|
| `V10_22` | `run_v10.sh` | 22 | **Shipped.** Adds the `bilateral` input transform (§5.1) and a cosine schedule over 80 epochs |
| `V9_22` | `run_v9.sh` | 22 | 20 epochs, no input transform. The comparison anchor for what the transform and the schedule bought |
| `V8A_*` | `run_v8.sh` | 11, 22, 33 | The three-seed spread. Still the only multi-seed evidence at this recipe, so seed variance is quoted from here |
| `V8B_11` | `run_v8.sh` | 11 | **Control**: the previous recipe on the *same* splits. Exists so V8A's numbers can be attributed to the recipe rather than to the split change |

The control is the load-bearing part. v7's test splits differ from v5 and v6, so the
stored `C3B`/`C3V6` results are **not** a valid baseline for V8A, and comparing against
them would credit the recipe with gains that belong to the data.

V9 and V10 are single-seed by choice: with a fixed deadline the budget bought epochs on
the best-performing seed rather than three under-trained runs. Reading rule 1 still
applies to them — their spread is unmeasured, and V8A's ±0.010 on `test_factory` is the
best available estimate of it.

## Three-class rounds, in order

| Tag | Splits | Seeds | Result and what changed |
|---|---|---|---|
| `C3_V5_*` | v5 | 11, 33 | First 3-class run. Established that the head works |
| `C3B_*` | v5 | 11, 22, 33 | `test_factory` clDice 0.592 ±0.050 — the first factory headline |
| `C3V6_*` | v6 | 11, 22, 33 | 0.611 ±0.020. Confirmed the early-peak behaviour that set the 2-epoch recipe |
| `V7_*` | v7 | 11 (killed) | Killed at epoch 5 of 8 in favour of the v8 recipe. Its epoch-1 headline of 0.6359 is the V8B comparison anchor |
| `V8A_*`, `V8B_11` | v7 | see above | Current |

## Ablations, and what each one settled

| Question | Tags | Script | Answer |
|---|---|---|---|
| Which architecture? | `ts`, `ts1`, *(bakeoff)* | `run_slim.sh`, `run_tinyseg.sh` | **No reliable gain above 1.43 M params.** `seen` clDice sits at 0.63–0.71 from 1.43 M to 6.25 M. Sub-0.5 M `tinyseg` variants have 10–15× worse FP area |
| Does model size help transfer? | `mseed*` | `run_seeds.sh` | No. The larger model's apparent wood advantage (0.579 in `ts`) did not replicate: 0.204 / 0.087 / 0.334 over three seeds |
| Mix the two synthetic crack sets? | `ds_A`–`ds_G`, `L6_*` | `run_bc.sh`, `run_e.sh`, `run_ts.sh`, `run_long.sh` | **No.** Measured five ways; combining consistently reduced cross-material clDice even where in-domain metrics improved. This is what ADR-013 protects |
| Is wood needed in training? | `abl_nowood`, `abl_woodsynth` | — | Wood is held out as the transfer test; these bound what that costs |
| Does EMA help? | `ema11/22/33` | `run_ema.sh` | Yes — cut seed variance ~3×. EMA 0.995 is in every recipe since |
| Are scratches learnable? | `SCR_*` | `run_scr.sh` | Yes, but only with sampler-level quota. See ADR-012 |
| Steel-specific check | `STEEL_*` | `run_v3.sh` | Steel remains the hardest material for geometry despite having the most masks |
| Does any input transform help? | — | `bench/prep_sweep.py` | **Only mild denoising, and NFR-03 picks which.** 22 configurations on `val`, reported on the frozen splits. `bilateral` selected; `median3` had the better headline and was rejected for breaching the FP ceiling. CLAHE costs 0.057–0.108. No pair beat its own weaker component. See ADR-018 |
| How much of the unseen-material loss is recoverable? | `V8A_22`, `V9_22` | — | **Not by thresholding.** Sweeping the foreground threshold 0.50 → 0.10 moves wood detection by 0.005, so the loss is representational. `bilateral` recovers +0.088 clDice of it |
| Split version history | `V2_*`, `V3_*`, `V4_*` | `run_v2.sh`–`run_v5.sh` | Binary-head runs across split freezes v2–v4 |
| Binary final | `FINAL*` | `run_final.sh`, `run_verify.sh` | The binary baseline in ADR-003: 1.43 M params, 5.8 MB ONNX, 26.1 ms |

## Reading rules

These exist because each was violated at least once during development.

1. **Single-seed numbers are not evidence.** See the `mseed` spread above.
2. **Quote the sample size below ~50 images.** `test_factory_scratch` is 31 images; the
   glass subset of `test_factory` is 2.
3. **Headline clDice and IoU are class-agnostic.** They are computed on `any_defect`, so
   a model can score 0.67 clDice while typing more than half of crack pixels as scratch
   — which is the currently measured state. Always read `crack_class_recall` beside them.
4. **The headline split is not one population.** Image-weighted clDice 0.651 vs
   material-weighted 0.525; see `bench/per_material.py` and ADR-016.
5. **Compare only within a split version.** Each freeze changes what the test sets
   contain. The split sha is recorded in every JSON.
6. **`SMOKE*` tags are syntax tests, not results.** They run 2–4 steps and score near
   zero by construction.
