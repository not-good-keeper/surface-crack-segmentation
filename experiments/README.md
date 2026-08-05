# Experiment scripts

One script per question. They are kept rather than deleted because each is the exact
command that produced a stored result in `data/bench/`, and a number whose command has
been thrown away is not reproducible. `docs/RESULTS.md` maps tags to claims.

| Script | Tags | Question it answered |
|---|---|---|
| `run_v9.sh` | `V9_22` | Does the 4-epoch cap cost anything? 20 epochs on the best seed (ADR-017) |
| `run_v8.sh` | `V8A_*`, `V8B_11` | Does a real-heavy batch beat the synthetic-heavy one? Includes the same-splits control |
| `run_3class.sh` | `V7_*` | First three-class run on splits v7 |
| `run_slim.sh` | `ts` | Architecture bake-off: nine models, identical config |
| `run_tinyseg.sh` | `ts1` | Can a sub-0.5 M model work at all? |
| `run_seeds.sh` | `mseed*` | Seed variance — the run that proved single seeds are not evidence |
| `run_ema.sh` | `ema*` | Does EMA reduce that variance? |
| `run_long.sh` | `L6_*` | Longer schedule on the synthetic-source comparison |
| `run_bc.sh`, `run_e.sh`, `run_ts.sh`, `run_seed2.sh` | `ds_*`, `G_*` | Synthetic-source ablations: ours / theirs / combined |
| `run_scr.sh` | `SCR_*` | Is the scratch class learnable? |
| `run_v2.sh` … `run_v5.sh` | `V2_*`–`V4_*`, `STEEL_*` | Split-freeze history under the binary head |
| `run_final.sh`, `run_verify.sh` | `FINAL*`, `v2` | The binary baseline quoted in ADR-003 |

The current run lives at the repository root as `run_v9.sh` — the one command worth
finding without reading this file.

Paths inside these scripts are relative to the repository root, so run them from there:

```bash
bash experiments/run_slim.sh
```
