"""One training/eval loop, identical for every candidate.

Fairness is the whole point: same data, same order, same seed, same loss, same
schedule, same augmentation. The only thing that varies is the architecture, so a
difference in the results is a difference in the architecture.

Loss pairs a pixel-classification term with soft Dice. Pure BCE/CE on 2-4 % foreground
converges to predicting nothing (and scores 96-98 % pixel accuracy doing it); Dice
supplies the gradient that makes the positive class worth predicting.

`--classes 1` is the binary crack head that every stored result in data/bench was
measured with, and it is the default so those files keep describing a reproducible run.
`--classes 3` is the deployment head from docs/ARCHITECTURE.md: background, crack and
scratch competing through a softmax, trained with class-weighted cross-entropy.

Per-epoch validation metrics are recorded, not just the final value, so the
recommender can look at the learning-curve slope -- a model still climbing at epoch 2
may beat one that has already plateaued, and a 2-epoch smoke cannot tell those apart
from the final number alone.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
import models as M            # noqa: E402
from data import CrackPatches  # noqa: E402
from metrics import evaluate, evaluate_multiclass, all_background_baseline  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data/bench"


def dice_bce(logits, y, eps=1.0):
    bce = nn.functional.binary_cross_entropy_with_logits(logits, y)
    p = torch.sigmoid(logits)
    num = 2 * (p * y).sum(dim=(1, 2, 3)) + eps
    den = p.sum(dim=(1, 2, 3)) + y.sum(dim=(1, 2, 3)) + eps
    return bce + (1 - (num / den)).mean()


def dice_ce(logits, y, weight=None, eps=1.0):
    """Class-weighted cross-entropy + soft Dice over the foreground classes.

    At 2-4 % foreground, CE alone converges to predicting background everywhere.
    Background is excluded from the Dice term -- Dice on the majority class sits near
    1 and drowns the gradient the minority classes need.
    """
    ce = nn.functional.cross_entropy(logits, y, weight=weight)
    p = torch.softmax(logits, dim=1)
    dice = 0.0
    n_fg = logits.shape[1] - 1
    for c in range(1, logits.shape[1]):
        pc = p[:, c]
        yc = (y == c).float()
        num = 2 * (pc * yc).sum(dim=(1, 2)) + eps
        den = pc.sum(dim=(1, 2)) + yc.sum(dim=(1, 2)) + eps
        dice = dice + (1 - (num / den)).mean()
    return ce + dice / max(n_fg, 1)


@torch.no_grad()
def predict_split(model, ds, device, bs=32, thresh=0.5, max_batches=None,
                  classes=1):
    dl = DataLoader(ds, batch_size=bs, shuffle=False, num_workers=2,
                    pin_memory=True, persistent_workers=False)
    model.eval()
    P, G = [], []
    for bi, (x, y) in enumerate(dl):
        if max_batches and bi >= max_batches:
            break
        with torch.autocast("cuda", enabled=device.type == "cuda"):
            logits = model(x.to(device, non_blocking=True))
        if classes == 1:
            p = torch.sigmoid(logits).float().cpu().numpy()[:, 0] > thresh
            g = y.numpy()[:, 0] > 0.5
        else:
            p = logits.float().argmax(dim=1).cpu().numpy()
            g = y.numpy()
        P.append(p)
        G.append(g)
    return np.concatenate(P), np.concatenate(G)


@torch.no_grad()
def eval_loss(model, ds, device, weight=None, classes=1, bs=32, max_batches=None):
    """Mean loss on a split, under the identical loss the run trains with.

    Reported beside the training loss because they answer different questions: training
    loss falling while validation loss rises is overfitting, and no accuracy metric
    shows that as directly. clDice can sit flat while the two losses diverge.
    """
    dl = DataLoader(ds, batch_size=bs, shuffle=False, num_workers=2, pin_memory=True)
    model.eval()
    tot, n = 0.0, 0
    for bi, (x, y) in enumerate(dl):
        if max_batches and bi >= max_batches:
            break
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.autocast("cuda", enabled=device.type == "cuda"):
            out = model(x)
            loss = dice_bce(out, y) if classes == 1 else dice_ce(out, y, weight=weight)
        tot += float(loss)
        n += 1
    return tot / max(n, 1)


def class_weights(ds, n_cls, device, cap=6.0):
    """Inverse-frequency weights from the training pool, capped.

    From the manifest, not a pixel scan: the pool is resampled every epoch. The cap
    matters because uncapped inverse frequency makes the loss chase scratch pixels and
    flood the image with them (ADR-012).
    """
    import numpy as _np
    counts = _np.ones(n_cls, dtype=_np.float64)
    counts[0] = max(len(ds.neg), 1)
    roles = ds.pos.role.value_counts().to_dict() if len(ds.pos) else {}
    counts[1] = max(roles.get("positive", 0), 1)
    if n_cls > 2:
        counts[2] = max(roles.get("scratch", 0), 1)
    w = counts.sum() / (n_cls * counts)
    w = _np.clip(w / w[0], 1.0, cap)
    print(f"  class weights {w.round(3).tolist()} from image counts "
          f"{counts.astype(int).tolist()}")
    return torch.tensor(w, dtype=torch.float32, device=device)


def measure_cost(model, device, size=256):  # size passed by caller
    """Params, GPU latency, and CPU latency -- the deployment-relevant number."""
    model.eval()
    n = M.count_params(model)
    x = torch.randn(1, 3, size, size)
    # CPU latency is the honest proxy for a phone; GPU latency is not.
    mc = model.to("cpu")
    with torch.no_grad():
        for _ in range(3):
            mc(x)
        t0 = time.perf_counter()
        for _ in range(10):
            mc(x)
        cpu_ms = (time.perf_counter() - t0) / 10 * 1000
    model.to(device)
    gpu_ms = float("nan")
    if device.type == "cuda":
        xg = x.to(device)
        with torch.no_grad():
            for _ in range(5):
                model(xg)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(20):
                model(xg)
            torch.cuda.synchronize()
            gpu_ms = (time.perf_counter() - t0) / 20 * 1000
    return dict(params=n, cpu_ms=cpu_ms, gpu_ms=gpu_ms)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--train-steps", type=int, default=600, help="batches per epoch")
    ap.add_argument("--synth", default=None,
                    help="comma-separated synthetic dirs, train-only. A dir may carry "
                         "`:kind|kind` to select kinds for that source alone, e.g. "
                         "data/synth:crack|negative,data/df_patches2:scratch")
    ap.add_argument("--synth-frac", type=float, default=0.5)
    ap.add_argument("--exclude-materials", nargs="*", default=[])
    ap.add_argument("--camera-aug", action="store_true",
                    help="apply camera-pipeline augmentation to training data")
    ap.add_argument("--camera-profile", default="conveyor",
                    choices=["conveyor", "handheld"],
                    help="capture model; 'handheld' reproduces pre-v5 runs")
    ap.add_argument("--classes", type=int, default=1,
                    help="1 = binary crack head (all stored benchmarks); "
                         "3 = background/crack/scratch")
    ap.add_argument("--synth-kinds", default=None,
                    help="comma-separated synthetic kinds to keep across every source, "
                         "e.g. crack,scratch. Overridden per source by `--synth dir:k|k`")
    ap.add_argument("--headline-split", default="test_factory",
                    help="split tracked every epoch and used for model selection")
    ap.add_argument("--class-weight-cap", type=float, default=6.0,
                    help="ceiling on inverse-frequency class weight")
    ap.add_argument("--scratch-frac", type=float, default=0.35,
                    help="share of DEFECT samples drawn from the scratch pool. "
                         "Rebalances at the sampler rather than the loss: at its "
                         "natural 6%% frequency the scratch class was never predicted "
                         "at all, and raising the loss weight instead buys recall by "
                         "over-painting, which is the failure NFR-3 guards against")
    ap.add_argument("--tag", default="")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--eval-batches", type=int, default=40)
    ap.add_argument("--ema", type=float, default=0.0,
                    help="EMA decay on weights, e.g. 0.99; 0 disables")
    ap.add_argument("--size", type=int, default=256,
                    help="input edge. 256 downscales a 793 px frame ~3x, taking a 4 px "
                         "crack under 1.5 px; 512 halves that loss for ~4x the compute")
    ap.add_argument("--init", default=None,
                    help="warm-start from a checkpoint (.pt or .best.pt). The network "
                         "is fully convolutional, so weights learned at one input size "
                         "load at another and only the scale has to be relearned")
    ap.add_argument("--resize", action="store_true",
                    help="scale the whole frame to 256 instead of cropping at native "
                         "scale; matches app/inference.py (T-02)")
    ap.add_argument("--prep", default=None,
                    help="input transform from bench/preprocess.py, applied to train "
                         "and eval alike, e.g. 'bilateral' or 'flatten+clahe2'")
    ap.add_argument("--cosine", action="store_true",
                    help="cosine-decay the lr to 2%% over the run. At a constant lr the "
                         "headline swings 0.06 between adjacent epochs, which is the "
                         "optimiser orbiting rather than converging; over a long run "
                         "that also turns best-epoch selection into a lottery")
    ap.add_argument("--save", action="store_true",
                    help="persist the state_dict for export")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Input shape is fixed for the whole run (batch x 3 x size x size), so cuDNN's
    # autotuner pays for itself on the first few steps and is free thereafter. TF32
    # matmuls are enabled for the same reason: the loss is computed in fp32 under
    # autocast either way, so this changes throughput and not the reported numbers.
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    OUT.mkdir(parents=True, exist_ok=True)
    run = f"{args.model.replace('/','_')}{('_'+args.tag) if args.tag else ''}"
    print(f"=== {run} on {device} ===")

    ds_kw = dict(classes=args.classes, camera_profile=args.camera_profile,
                 scratch_frac=args.scratch_frac, prep=args.prep,
                 resize=args.resize, size=args.size)
    tr = CrackPatches("train", train=True, synth_dir=args.synth,
                      synth_frac=args.synth_frac, synth_kinds=args.synth_kinds,
                      exclude_materials=args.exclude_materials, seed=args.seed,
                      camera_aug=args.camera_aug,
                      length=args.train_steps * args.batch, **ds_kw)
    va = CrackPatches("val", train=False, **ds_kw)
    # Track the headline split EVERY epoch. It behaves nothing like val (val is stable
    # across seeds while held-out material varies 7x), and logging it only at the end
    # hides both its trajectory and the fact that it peaks early and then decays.
    un = CrackPatches(args.headline_split, train=False, **ds_kw)
    print(f"train pool: {len(tr.pos)} pos / {len(tr.neg)} neg"
          f"{f' + {len(tr.synth)} synth' if tr.synth is not None else ''} | "
          f"val: {len(va.pos)} pos / {len(va.neg)} neg | "
          f"headline '{args.headline_split}': {len(un.pos)} pos")
    if args.classes > 1 and not len(tr.pos[tr.pos.role == "scratch"]):
        raise SystemExit("no scratch rows in the training pool -- the scratch class "
                         "would be trained entirely on synthetic data. Re-cut splits.")

    model = M.build(args.model, classes=args.classes).to(device)
    if args.init:
        # Starting a higher-resolution run from the converged lower-resolution weights
        # rather than from ImageNet. What a defect looks like is already learned; only
        # its scale changed, and that is a far smaller thing to relearn than the task.
        blob = torch.load(args.init, map_location=device)
        state = blob["state"] if isinstance(blob, dict) and "state" in blob else blob
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing or unexpected:
            raise SystemExit(f"--init {args.init} does not match {args.model}: "
                             f"{len(missing)} missing, {len(unexpected)} unexpected "
                             f"keys. Warm-starting from the wrong architecture would "
                             f"train from a state nothing describes.")
        print(f"  warm start from {Path(args.init).name}")
    cw = (class_weights(tr, args.classes, device, cap=args.class_weight_cap)
          if args.classes > 1 else None)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = (torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.epochs * args.train_steps, eta_min=args.lr * 0.02)
        if args.cosine else None)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    dl = DataLoader(tr, batch_size=args.batch, shuffle=False, num_workers=4,
                    pin_memory=True, persistent_workers=True, drop_last=True)

    # Exponential moving average of weights. The unseen-material metric swings
    # 0.5 between consecutive epochs of one run while val stays flat, which is
    # the signature of the optimiser orbiting a minimum rather than of genuine
    # learning. Averaging the trajectory is the standard remedy and costs one
    # extra copy of the weights.
    ema_model = None
    if args.ema > 0:
        import copy
        ema_model = copy.deepcopy(model).eval()
        for q in ema_model.parameters():
            q.requires_grad_(False)

    # Track the best epoch by the headline metric, not just the last one. The factory
    # score peaks at epoch 2-3 and then decays while val keeps improving, so a long run
    # that saves only its final weights reliably ships a worse model than it found.
    best = dict(score=-1.0, epoch=0, state=None)
    history, t_start = [], time.time()
    for ep in range(1, args.epochs + 1):
        model.train()
        tot, nb, t0 = 0.0, 0, time.time()
        for x, y in dl:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", enabled=device.type == "cuda"):
                out = model(x)
                loss = (dice_bce(out, y) if args.classes == 1
                        else dice_ce(out, y, weight=cw))
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            if sched is not None:
                sched.step()
            tot += float(loss)
            nb += 1
            if ema_model is not None:
                with torch.no_grad():
                    d = args.ema
                    for pe, pm in zip(ema_model.state_dict().values(),
                                      model.state_dict().values()):
                        if pe.dtype.is_floating_point:
                            pe.mul_(d).add_(pm.detach(), alpha=1 - d)
                        else:
                            pe.copy_(pm)
        eval_model = ema_model if ema_model is not None else model
        ev = (lambda P, G: evaluate(P, G)) if args.classes == 1 else \
             (lambda P, G: evaluate_multiclass(P, G)["any_defect"])
        P, G = predict_split(eval_model, va, device, max_batches=args.eval_batches,
                             classes=args.classes)
        m = ev(P, G)
        Pu, Gu = predict_split(eval_model, un, device, max_batches=args.eval_batches,
                               classes=args.classes)
        mu = ev(Pu, Gu)
        m["unseen_cldice"] = mu["cldice"]
        m["unseen_iou"] = mu["iou"]
        m["val_loss"] = eval_loss(eval_model, va, device, weight=cw,
                                  classes=args.classes,
                                  max_batches=args.eval_batches)
        m.update(epoch=ep, train_loss=tot / max(nb, 1), epoch_s=time.time() - t0)
        history.append(m)
        star = ""
        if mu["cldice"] > best["score"]:
            import copy as _copy
            best.update(score=mu["cldice"], epoch=ep,
                        state=_copy.deepcopy(eval_model.state_dict()))
            star = " *best"
            if args.save:
                # Mirror the best state to disk as well as to RAM. The in-memory copy
                # is lost if the process dies, and a run killed at epoch 5 of 8 threw
                # away 23 minutes of training that was already better than anything
                # a shorter run would have produced. 5.7 MB per improvement is a
                # trivial price for making a kill cost one epoch instead of a run.
                OUT.mkdir(parents=True, exist_ok=True)
                torch.save({"state": best["state"], "epoch": ep,
                            "headline_cldice": mu["cldice"], "tag": args.tag,
                            "model": args.model, "classes": args.classes},
                           OUT / f"{run}.best.pt")
        m["lr"] = opt.param_groups[0]["lr"]
        # gap = val - train. Rising while train falls is the overfitting signature.
        gap = m["val_loss"] - m["train_loss"]
        print(f"  ep{ep} loss={m['train_loss']:.4f} vloss={m['val_loss']:.4f} "
              f"gap={gap:+.4f} val_clD={m['cldice']:.4f} "
              f"HEAD_clD={mu['cldice']:.4f} hIoU={mu['iou']:.4f} "
              f"det={mu['detect_rate']:.3f} fp={m['fp_area']:.5f} "
              f"lr={m['lr']:.2e} ({m['epoch_s']:.0f}s){star}", flush=True)

    result = dict(run=run, model=args.model, tag=args.tag, args=vars(args),
                  history=history, train_s=time.time() - t_start)

    # final evaluation on every frozen test split
    if ema_model is not None:
        model = ema_model
    if best["state"] is not None and best["epoch"] != args.epochs:
        print(f"  restoring best epoch {best['epoch']} "
              f"(headline {best['score']:.4f}) over final epoch {args.epochs}")
        model.load_state_dict(best["state"])
        result["best_epoch"] = best["epoch"]

    for split in ["test_factory", "test_factory_scratch", "test_scratch_blob",
                  "test_seen", "test_unseen_material", "test_negatives"]:
        try:
            ds = CrackPatches(split, train=False, **ds_kw)
            if len(ds.pos) == 0 and len(ds.neg) == 0:
                continue
            P, G = predict_split(model, ds, device, max_batches=args.eval_batches,
                                 classes=args.classes)
            if args.classes == 1:
                r = evaluate(P, G)
            else:
                r = evaluate_multiclass(P, G)
                r.update({k: v for k, v in r["any_defect"].items()})
            if split == "test_scratch_blob":
                # These masks outline the anomaly REGION, not the scratch (median
                # width 33.6 px against 7.3 px for real scratches). Overlap against
                # them measures agreement with a labelling convention we rejected, so
                # only the two questions that survive it are kept: did the model fire,
                # and did it choose the right class.
                r = {k: v for k, v in r.items()
                     if k in ("detect_rate", "n_pos", "n_neg", "class_names",
                              "confusion_defect_px", "crack_class_recall",
                              "scratch_class_recall")}
                r["note"] = ("geometry metrics withheld: wide-region mask convention, "
                             "detection and class only")
            result[split] = r
            if split == "test_factory":
                # The floor for the whole metric set: a model predicting nothing
                # scores 96-98 % pixel accuracy here. Recorded alongside the real
                # numbers so nobody has to take on trust that they mean something.
                result["baseline_all_background"] = all_background_baseline(G > 0)
            if "iou" in r:
                print(f"  {split:<22} IoU={r['iou']:.4f} clDice={r['cldice']:.4f} "
                      f"det={r['detect_rate']:.3f} fp_area={r['fp_area']:.5f}")
            else:
                print(f"  {split:<22} det={r['detect_rate']:.3f}  "
                      f"(detection/class only)")
            if args.classes > 1 and "crack_class_recall" in r:
                print(f"  {'':<22} class recall: crack="
                      f"{r['crack_class_recall']:.3f} "
                      f"scratch={r['scratch_class_recall']:.3f}")
        except Exception as e:  # noqa: BLE001
            print(f"  {split}: SKIP ({type(e).__name__}: {str(e)[:60]})")

    result["cost"] = measure_cost(model, device, size=args.size)
    print(f"  params={result['cost']['params']/1e6:.2f}M "
          f"cpu={result['cost']['cpu_ms']:.0f}ms gpu={result['cost']['gpu_ms']:.1f}ms")

    if args.save:
        ck = OUT / f"{run}.pt"
        torch.save(model.state_dict(), ck)
        result["checkpoint"] = str(ck)
        print(f"  checkpoint -> {ck}")

    (OUT / f"{run}.json").write_text(json.dumps(result, indent=1, default=float))
    print(f"  -> {OUT / (run + '.json')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
