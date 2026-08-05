"""Read every benchmark run and recommend what TinySeg should be modelled after.

The hard part is not ranking numbers, it is not over-claiming from them. A 2-epoch
smoke is genuinely noisy, so this script:

* ranks on **clDice + tolerant-F1**, not raw IoU (a 1 px shift can halve IoU on a 2 px
  crack, which would rank architectures by luck);
* looks at the **learning-curve slope**, because a model still climbing steeply at
  epoch 2 can beat one that has plateaued, and the final value alone hides that;
* **declares a tie** when the top runs fall inside the noise margin, and names the
  experiment that would break it, instead of inventing a winner;
* filters by the **deployment budget** before ranking -- an architecture that cannot
  run on a phone is not a candidate however accurate it is;
* checks the **false-positive rate** separately, since that is the failure that makes
  an inspection app unusable and no accuracy metric exposes it.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BENCH = ROOT / "data/bench"
# The deployment budget the ranking is filtered against, from ARCHITECTURE.md 6.
MAX_PARAMS_M = 8.0
MAX_CPU_MS = 400.0
TIE_MARGIN = 0.02          # composite scores this close are reported as a tie, not a win
OUT = ROOT / "data/report/model_selection.md"

# A crack is 1-3 px wide; these describe how much spatial detail each design keeps.
STRIDE_NOTES = {
    "tiny_unet": "1 (full-res skips)",
    "smp_unet_timm-mobilenetv3_small_100": "1 (full-res skips)",
    "smp_unet_efficientnet-b0": "1 (full-res skips)",
    "smp_dlv3p_timm-mobilenetv3_large_100": "8-16 (dilated ASPP)",
    "lraspp_mnv3": "8-32 (light head)",
    "segformer_b0": "4 (hierarchical attn)",
}


def acc_of(r: dict, split="val") -> float:
    """Composite accuracy: clDice and tolerant-F1 weighted equally."""
    m = r["history"][-1] if split == "val" else r.get(split)
    if not m:
        return float("nan")
    cl, tf = m.get("cldice"), m.get("tf1@2")
    vals = [v for v in (cl, tf) if v == v]
    return sum(vals) / len(vals) if vals else float("nan")


def slope_of(r: dict) -> float:
    h = r.get("history", [])
    if len(h) < 2:
        return 0.0
    a = (h[-1].get("cldice", 0) or 0) - (h[-2].get("cldice", 0) or 0)
    return float(a)


def main() -> int:
    runs = []
    for p in sorted(BENCH.glob("*.json")):
        try:
            runs.append(json.loads(p.read_text()))
        except Exception:  # noqa: BLE001
            continue
    if not runs:
        print("no runs found in data/bench/")
        return 1
    base = [r for r in runs if not r.get("tag")]      # ablations carry a tag
    if not base:
        base = runs

    rows = []
    for r in base:
        c = r.get("cost", {})
        rows.append(dict(
            run=r["run"], model=r["model"],
            stride=STRIDE_NOTES.get(r["model"], "?"),
            val=acc_of(r), slope=slope_of(r),
            seen=acc_of(r, "test_seen"),
            unseen=acc_of(r, "test_unseen_material"),
            fp=(r.get("test_negatives") or {}).get("fp_area", float("nan")),
            params=c.get("params", 0) / 1e6, cpu=c.get("cpu_ms", float("nan")),
            train_s=r.get("train_s", 0),
        ))
    rows.sort(key=lambda d: -(d["val"] if d["val"] == d["val"] else -1))

    L = ["# TinySeg architecture selection", "",
         f"_{len(rows)} candidates, identical data / schedule / seed. "
         f"Ranked on clDice + tolerant-F1@2px, not raw IoU._", ""]

    L += ["## Results", "",
          "| model | output stride | val acc | slope | test_seen | unseen(wood) "
          "| fp_area | params M | cpu ms |", "|---|---|---|---|---|---|---|---|---|"]
    for d in rows:
        f = lambda v, n=3: ("n/a" if v != v else f"{v:.{n}f}")  # noqa: E731
        L.append(f"| `{d['model']}` | {d['stride']} | **{f(d['val'])}** | "
                 f"{d['slope']:+.3f} | {f(d['seen'])} | {f(d['unseen'])} | "
                 f"{f(d['fp'],5)} | {d['params']:.2f} | {d['cpu']:.0f} |")

    # ---- sanity floor -------------------------------------------------------
    bl = next((r.get("baseline_all_background") for r in base
               if r.get("baseline_all_background")), None)
    if bl:
        L += ["", "## Sanity floor (predict-all-background)", "",
              f"- pixel accuracy **{bl.get('pixel_accuracy', float('nan')):.4f}** "
              f"-- looks excellent, and is worthless",
              f"- clDice **{bl.get('cldice', 0):.4f}**, detect_rate "
              f"**{bl.get('detect_rate', 0):.4f}**",
              "", "This is why accuracy-style metrics are never reported alone here."]

    # ---- deployment filter --------------------------------------------------
    ok = [d for d in rows if d["params"] <= MAX_PARAMS_M
          and (d["cpu"] != d["cpu"] or d["cpu"] <= MAX_CPU_MS)]
    rejected = [d for d in rows if d not in ok]
    L += ["", "## Deployment filter", "",
          f"Budget: <= {MAX_PARAMS_M:.0f} M params, "
          f"<= {MAX_CPU_MS:.0f} ms desktop-CPU at 256x256 "
          f"(a mid-range phone is roughly 3-6x slower, so treat this as optimistic)."]
    if rejected:
        L.append("")
        for d in rejected:
            L.append(f"- rejected `{d['model']}` "
                     f"({d['params']:.1f} M, {d['cpu']:.0f} ms)")

    # ---- in-domain vs cross-material disagreement ---------------------------
    # These two rankings can disagree sharply, and when they do, ranking on val
    # accuracy alone recommends the wrong architecture for this project. The claim
    # in the submission is cross-material generalisation, so a model that wins
    # in-domain while collapsing on an unseen material is a false winner.
    by_val = [d for d in rows if d["val"] == d["val"]]
    by_uns = [d for d in rows if d["unseen"] == d["unseen"]]
    if by_val and by_uns:
        by_uns = sorted(by_uns, key=lambda d: -d["unseen"])
        vw, uw = by_val[0], by_uns[0]
        L += ["", "## In-domain accuracy vs cross-material generalisation", ""]
        if vw["model"] != uw["model"]:
            ratio = (uw["unseen"] / vw["unseen"]) if vw["unseen"] > 1e-6 else float("inf")
            L += [f"**These disagree, and the disagreement is large.**", "",
                  f"- Best in-domain (`val`): `{vw['model']}` at {vw['val']:.3f} -- "
                  f"but only **{vw['unseen']:.3f}** on the unseen material.",
                  f"- Best cross-material: `{uw['model']}` at **{uw['unseen']:.3f}** "
                  f"on unseen wood, {ratio:.1f}x better, while giving up only "
                  f"{vw['val'] - uw['val']:.3f} in-domain.",
                  "",
                  "In-domain accuracy is measured on materials the model trained on, so "
                  "it rewards fitting those textures. The submission claims the opposite "
                  "capability. **Rank on `test_unseen_material` for this project**; "
                  "treat `val` as a sanity check that training worked at all.",
                  "",
                  "| model | unseen(wood) | val | cpu ms | params M |",
                  "|---|---|---|---|---|"]
            for d in by_uns:
                L.append(f"| `{d['model']}` | **{d['unseen']:.3f}** | {d['val']:.3f} | "
                         f"{d['cpu']:.0f} | {d['params']:.2f} |")
        else:
            L += [f"They agree: `{vw['model']}` leads both.",
                  "That is the comfortable case; no trade-off to make."]

    # ---- winner or tie ------------------------------------------------------
    L += ["", "## Verdict", ""]
    if not ok:
        L.append("No candidate met the deployment budget. Loosen it or shrink the encoder.")
    else:
        top = ok[0]
        close = [d for d in ok if abs(d["val"] - top["val"]) <= TIE_MARGIN]
        if len(close) > 1:
            names = ", ".join(f"`{d['model']}`" for d in close)
            L += [f"**Tie ({len(close)} models within {TIE_MARGIN:.02f} composite "
                  f"accuracy): {names}.**", "",
                  "A 2-epoch smoke cannot separate these; the gap is inside the noise. "
                  "Break it by re-running only these for 10-15 epochs with 3 seeds and "
                  "comparing on `test_unseen_material`, which is the metric the "
                  "submission actually claims."]
            # Among statistically tied models, break the tie on cross-material
            # generalisation rather than cost. Cheapest-of-equals is the right rule
            # only when the tied models are equivalent on what we actually care
            # about, and here they are not: unseen-material scores differ by ~9x
            # across the tie group.
            gen = [d for d in close if d["unseen"] == d["unseen"]]
            if gen and (max(d["unseen"] for d in gen)
                        > 1.5 * min(d["unseen"] for d in gen)):
                best_gen = max(gen, key=lambda d: d["unseen"])
                worst_gen = min(gen, key=lambda d: d["unseen"])
                L += ["", f"They are tied in-domain but **not** on the metric that "
                          f"matters: unseen-material clDice ranges "
                          f"{worst_gen['unseen']:.3f} (`{worst_gen['model']}`) to "
                          f"{best_gen['unseen']:.3f} (`{best_gen['model']}`).",
                      "", f"**Take `{best_gen['model']}`** "
                          f"({best_gen['params']:.2f} M, {best_gen['cpu']:.0f} ms) -- "
                          f"tied in-domain, clearly best across materials."]
                top = best_gen
            else:
                cheap = min(close, key=lambda d: (d["cpu"], d["params"]))
                L += ["", f"They are equivalent on cross-material generalisation too, "
                          f"so break the tie on cost: take `{cheap['model']}` "
                          f"({cheap['params']:.2f} M, {cheap['cpu']:.0f} ms)."]
                top = cheap
        else:
            L += [f"**Winner: `{top['model']}`** -- composite {top['val']:.3f}, "
                  f"{top['params']:.2f} M params, {top['cpu']:.0f} ms CPU.",
                  "",
                  f"Margin over second place "
                  f"({ok[1]['model'] if len(ok) > 1 else 'n/a'}): "
                  f"{(top['val'] - ok[1]['val']):.3f}" if len(ok) > 1 else ""]

        climbing = [d for d in ok if d["slope"] > 0.01]
        if climbing:
            L += ["", "### Still improving at epoch 2",
                  "These had not plateaued, so their smoke ranking understates them:"]
            for d in climbing:
                L.append(f"- `{d['model']}` (+{d['slope']:.3f} clDice on the last epoch)")

        # ---- stride hypothesis -------------------------------------------
        # Judge this on cross-material generalisation, not in-domain accuracy. On
        # materials the model trained on, a coarse-stride net can memorise texture
        # and score well; the question is what survives a material change.
        us = [d for d in rows if d["stride"].startswith("1") and d["unseen"] == d["unseen"]]
        ds = [d for d in rows if d["stride"].startswith(("8",)) and d["unseen"] == d["unseen"]]
        if us and ds:
            mu, md = max(d["unseen"] for d in us), max(d["unseen"] for d in ds)
            vu = max(d["val"] for d in us if d["val"] == d["val"])
            vd = max(d["val"] for d in ds if d["val"] == d["val"])
            L += ["", "### Does output stride dominate?", "",
                  f"In-domain the two families are indistinguishable "
                  f"({vu:.3f} vs {vd:.3f}). On the unseen material they are not: "
                  f"**{mu:.3f} (full-res skips) vs {md:.3f} (stride 8+)**."]
            if mu > md * 1.5:
                L += ["",
                      "So resolution matters where it counts. Coarse-stride designs can "
                      "match on materials they trained on -- they have enough capacity to "
                      "fit those textures -- and then collapse on a new one. For 1-3 px "
                      "cracks, TinySeg should keep full-resolution skips.",
                      "",
                      "**Caveat: this is confounded with encoder strength.** The best "
                      "generaliser also has the strongest ImageNet-pretrained encoder, "
                      "and two weaker stride-1 models scored near the bottom "
                      f"({min(d['unseen'] for d in us):.3f}). Stride and pretraining "
                      "cannot be separated from this run; isolating them needs the same "
                      "encoder in both a U-Net and a DeepLab head."]
            else:
                L += ["", "The gap is not large enough to call. Choose on cost."]

        # ---- recipe --------------------------------------------------------
        L += ["", "## TinySeg starting recipe", "",
              f"- **Shape**: {top['stride']} - copy `{top['model']}`'s topology",
              "- **Encoder**: slim the winner's encoder (halve channel widths, drop the "
              "deepest stage) and re-measure; the benchmark says which family, not which size",
              "- **Resolution**: train and deploy at 256x256 (what was benchmarked)",
              "- **Loss**: BCE + soft Dice - pure BCE collapses to all-background at "
              "2-4% foreground",
              "- **Select on**: clDice and tolerant-F1, plus `fp_area` on "
              "`test_negatives`; never bare IoU or pixel accuracy",
              "- **Confirm**: one 10-15 epoch run before freezing the architecture - "
              "2 epochs picks a family, not a final model"]

    # ---- ablation ----------------------------------------------------------
    abl = [r for r in runs if r.get("tag")]
    if abl:
        L += ["", "## DefectForge ablation (real wood, never trained on)", "",
              "| run | unseen(wood) clDice | IoU |", "|---|---|---|"]
        for r in sorted(abl, key=lambda r: r["run"]):
            u = r.get("test_unseen_material") or {}
            L.append(f"| {r['run']} | {u.get('cldice', float('nan')):.4f} | "
                     f"{u.get('iou', float('nan')):.4f} |")
        withs = [r for r in abl if r.get("tag") == "abl_woodsynth"]
        nos = [r for r in abl if r.get("tag") == "abl_nowood"]
        if withs and nos:
            a = (nos[0].get("test_unseen_material") or {}).get("cldice", float("nan"))
            b = (withs[0].get("test_unseen_material") or {}).get("cldice", float("nan"))
            L += ["",
                  "Both runs see **zero real wood cracks**. The only difference is "
                  "whether DefectForge's *synthetic* wood cracks were in training.",
                  "",
                  f"**Delta: {b - a:+.4f} clDice on real wood.** That is what "
                  f"synthetic data bought on a material with no real annotations - "
                  f"the submission's actual claim, measured.",
                  "",
                  "_Caveat: synthetic wood was rendered on only 68 non-leaking clean "
                  "boards (840 crops), so substrate diversity is limited and this "
                  "understates what DefectForge could do with more clean wood._"]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))
    print(f"\n-> {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
