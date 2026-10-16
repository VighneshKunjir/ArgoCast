#!/usr/bin/env python
"""Mixed-layer-depth fix experiment for ArgoCast PI-BiLSTM.

Finding from the 10-model benchmark: ArgoCast PI-BiLSTM has the largest mixed layer depth
(MLD) error of the learned models, with a shallow bias. Diagnosis: in 39-45% of observed test
profiles the density between 5 and 20 dbar is almost uniform (a well-mixed surface layer), but
ArgoCast forecasts this in only 0.1-0.2% of cases; in ~95% of its forecasts density rises by
more than 0.03 kg m-3 between 5 and 20 dbar. The stability penalty max(0, -dsigma/dz) is zero
for any stable profile, but forecasts near dsigma = 0 risk tipping negative, so the network
learns to keep a safety margin of stratification everywhere, including inside the mixed layer.

This script retrains the same network (same data, splits, optimiser, early stopping as
run_benchmark.py) with loss variants that target that behaviour, and evaluates each on
validation and test windows.

Variants (all keep MSE(T) + 2.5 MSE(S) + 1.5 thermocline term unless stated)
  no_physics        stability weight 0 (reference: the ablation row of the benchmark)
  baseline          stability weight 10, no tolerance (the current ArgoCast PI-BiLSTM)
  tol_0.01          stability penalty ignores density drops smaller than 0.01 kg m-3 per interval
  tol_0.03          same with 0.03 kg m-3 (the MLD threshold)
  weight_1          stability weight 1 instead of 10
  surface_match     baseline + surface stratification term: MSE between forecast and observed
                    density differences over 5-20, 20-50 and 50-75 dbar (kg m-3), weight 10
  tol_0.01_surface  tol_0.01 + surface stratification term
  tol_0.03_surface  tol_0.03 + surface stratification term

Selection is made on the VALIDATION set only: the variant with the lowest validation MLD RMSE
among those whose validation temperature RMSE is within 0.005 degC of the baseline and whose
validation rate of inversions > 0.03 kg m-3 is at most 0.1%. Test numbers for every variant are
reported for transparency.

Examples
  python mld_fix_experiment.py                       (both splits, 3 seeds, ~50 min on an RTX 3060 Ti)
  python mld_fix_experiment.py --seeds 42            (one seed, ~17 min)
  python mld_fix_experiment.py --variants baseline tol_0.01 surface_match

Outputs: outputs/mld_fix/MLD_FIX_RESULTS.md, mld_fix_results.json, mld_fix_seeds.csv,
         figures/fig_mldfix_*.png|pdf
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_benchmark as rb  # noqa: E402  (also puts backend/ on sys.path)
import metrics_extra as mx  # noqa: E402
from app.ml.loss import PhysicsConstrainedLoss  # noqa: E402
from app.ml.model import PhysicsInformedBiLSTM  # noqa: E402

N_LEV = 16
SURF_IDX = [0, 1, 2]  # intervals 5-20, 20-50, 50-75 dbar

VARIANTS = {
    "no_physics":       dict(lam_stab=0.0,  tol=0.0,  lam_surf=0.0),
    "baseline":         dict(lam_stab=10.0, tol=0.0,  lam_surf=0.0),
    "tol_0.01":         dict(lam_stab=10.0, tol=0.01, lam_surf=0.0),
    "tol_0.03":         dict(lam_stab=10.0, tol=0.03, lam_surf=0.0),
    "weight_1":         dict(lam_stab=1.0,  tol=0.0,  lam_surf=0.0),
    "surface_match":    dict(lam_stab=10.0, tol=0.0,  lam_surf=10.0),
    "tol_0.01_surface": dict(lam_stab=10.0, tol=0.01, lam_surf=10.0),
    "tol_0.03_surface": dict(lam_stab=10.0, tol=0.03, lam_surf=10.0),
}
LABELS = {
    "no_physics": "No physics loss (ablation)",
    "baseline": "ArgoCast PI-BiLSTM (current loss)",
    "tol_0.01": "Stability tolerance 0.01 kg m-3",
    "tol_0.03": "Stability tolerance 0.03 kg m-3",
    "weight_1": "Stability weight 1 (instead of 10)",
    "surface_match": "Surface stratification term",
    "tol_0.01_surface": "Tolerance 0.01 + surface term",
    "tol_0.03_surface": "Tolerance 0.03 + surface term",
}
COLORS = {"no_physics": "#009E73", "baseline": "#0072B2", "tol_0.01": "#E69F00", "tol_0.03": "#D55E00",
          "weight_1": "#CC79A7", "surface_match": "#56B4E9", "tol_0.01_surface": "#882255",
          "tol_0.03_surface": "#7F3C8D"}


class VariantLoss(PhysicsConstrainedLoss):
    """ArgoCast loss with an optional stability tolerance and surface stratification term."""

    def __init__(self, tol=0.0, lam_surf=0.0, **kw):
        super().__init__(**kw)
        self.tol, self.lam_surf = float(tol), float(lam_surf)

    def compute_stability_loss(self, temp, sal):
        sigma = self.compute_potential_density(temp, sal)
        d_sigma = sigma[:, 1:] - sigma[:, :-1]
        d_depth = self.depths[1:] - self.depths[:-1]
        # tol = 0 reproduces the original penalty exactly: relu(-d_sigma / d_depth)
        return (torch.relu(-d_sigma - self.tol) / d_depth).sum(dim=1).mean()

    def forward(self, temp_pred, temp_true, sal_pred, sal_true):
        total, comp = super().forward(temp_pred, temp_true, sal_pred, sal_true)
        if self.lam_surf > 0:
            dp = torch.diff(self.compute_potential_density(temp_pred, sal_pred), dim=1)[:, SURF_IDX]
            dt = torch.diff(self.compute_potential_density(temp_true, sal_true), dim=1)[:, SURF_IDX]
            surf = torch.mean((dp - dt) ** 2)
            total = total + self.lam_surf * surf
            comp["surface"] = surf.item()
        return total, comp


def train(variant, parts, sc, seed, args, device):
    cfg = VARIANTS[variant]
    rb.set_seed(seed)
    model = PhysicsInformedBiLSTM().to(device)
    crit = VariantLoss(tol=cfg["tol"], lam_surf=cfg["lam_surf"], lambda_sal=2.5, lambda_stability=cfg["lam_stab"],
                       lambda_therm=1.5 if cfg["lam_stab"] > 0 else 0.0,
                       temp_mean=sc.t.mean_, temp_std=sc.t.scale_, sal_mean=sc.s.mean_, sal_std=sc.s.scale_).to(device)
    Ytr = sc.Y(parts["train"]["yT"], parts["train"]["yS"])
    Xtr = torch.tensor(sc.X(parts["train"]["X"]), device=device)
    YT, YS = torch.tensor(Ytr[:, :N_LEV], device=device), torch.tensor(Ytr[:, N_LEV:], device=device)
    Xva = torch.tensor(sc.X(parts["val"]["X"]), device=device)
    gen = torch.Generator(device="cpu").manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-6)
    best, best_state, bad, ep = np.inf, None, 0, 0
    t0 = time.time()
    for ep in range(1, args.epochs + 1):
        model.train()
        perm = torch.randperm(len(Xtr), generator=gen).to(device)
        for i in range(0, len(Xtr), args.batch_size):
            idx = perm[i:i + args.batch_size]
            opt.zero_grad(set_to_none=True)
            pT, pS = model(Xtr[idx])
            loss, _ = crit(pT, YT[idx], pS, YS[idx])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
        sched.step()
        vT, _ = sc.inv(rb.predict(model, Xva))
        r = rb.rmse(vT, parts["val"]["yT"])
        if r < best - 1e-6:
            best, bad = r, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
        if bad >= args.patience:
            break
    model.load_state_dict(best_state)
    return model, ep, time.time() - t0


def surface_stats(T, S, lat, lon):
    sig, _ = mx.teos(T, S, lat, lon)
    d = sig[:, 1] - sig[:, 0]
    return {"dsig_5_20_mean": float(d.mean()), "pct_mixed_5_20": float((d <= 0.005).mean() * 100),
            "pct_strat_5_20": float((d > 0.03).mean() * 100)}, d


def evaluate(pT, pS, ctx):
    r = mx.metrics_1_6(pT, pS, ctx)
    r.pop("_derived_arrays")
    s, d = surface_stats(pT, pS, ctx["lat"], ctx["lon"])
    row = {"temp_rmse": r["T"]["rmse"], "sal_rmse": r["S"]["rmse"], "temp_bias": r["T"]["bias"],
           "thermocline_rmse": r["T"]["thermocline_rmse"], "acc_T": r["T"]["acc"],
           "skill_vs_persistence_T": r["T"]["skill_vs_persistence"],
           "mld_rmse": r["derived"]["mld_m"]["rmse"], "mld_bias": r["derived"]["mld_m"]["bias"],
           "d20_rmse": r["derived"]["d20_m"]["rmse"], "ohc_rmse": r["derived"]["ohc_0_700_GJ_m2"]["rmse"],
           "inv_any_pct": r["physical"]["rate_pct"]["any"], "inv_001_pct": r["physical"]["rate_pct"]["0.01"],
           "inv_003_pct": r["physical"]["rate_pct"]["0.03"], **s}
    return row, d


def run_split(split, args, device, out):
    parts, _ = rb.load_arabian2000(args.proc_dir, split)
    sc = rb.Scaler(parts["train"]["yT"], parts["train"]["yS"])
    clim, _ = mx.fit_climatology(parts["train"], parts["val"])
    ctx = {k: mx.make_context(parts[k], clim) for k in ("val", "test")}
    obs = {}
    for k in ("val", "test"):
        s, d = surface_stats(ctx[k]["tT"], ctx[k]["tS"], ctx[k]["lat"], ctx[k]["lon"])
        obs[k] = {**s, "mld_mean": ctx[k]["observed"]["mld_mean_m"],
                  "inv_003_pct": ctx[k]["observed"]["inversion_rate_pct"]["0.03"]}
        if k == "test":
            obs["_hist_obs"] = d
    Xte = torch.tensor(sc.X(parts["test"]["X"]), device=device)
    Xva = torch.tensor(sc.X(parts["val"]["X"]), device=device)
    rows, hist = [], {}
    for v in args.variants:
        for j, seed in enumerate(args.seeds):
            model, ep, el = train(v, parts, sc, seed, args, device)
            for k, X in (("val", Xva), ("test", Xte)):
                pT, pS = sc.inv(rb.predict(model, X))
                row, d = evaluate(pT, pS, ctx[k])
                rows.append({"split": split, "variant": v, "seed": seed, "set": k, "epochs": ep, "train_seconds": el, **row})
                if k == "test" and j == 0:
                    hist[v] = d
            t = rows[-1]
            print(f"[{split}] {v:17s} seed {seed}: test T {t['temp_rmse']:.4f}  MLD RMSE {t['mld_rmse']:.2f} m "
                  f"(bias {t['mld_bias']:+.2f})  inv>0.03 {t['inv_003_pct']:.2f}%  any {t['inv_any_pct']:.2f}%  "
                  f"mixed5-20 {t['pct_mixed_5_20']:.1f}%  ep {ep} {el:.0f}s", flush=True)
    return rows, obs, hist


def select(df_split):
    """Choose a variant on validation means only."""
    v = df_split[df_split["set"] == "val"].groupby("variant").mean(numeric_only=True)
    if "baseline" not in v.index:
        return None, "baseline not run"
    base_T = v.loc["baseline", "temp_rmse"]
    ok = v[(v["temp_rmse"] <= base_T + 0.005) & (v["inv_003_pct"] <= 0.1)]
    ok = ok.drop(index=[i for i in ("no_physics",) if i in ok.index])
    if ok.empty:
        return None, "no variant met the validation criteria"
    best = ok["mld_rmse"].idxmin()
    return best, (f"lowest validation MLD RMSE ({ok.loc[best, 'mld_rmse']:.2f} m) among variants with validation "
                  f"temperature RMSE <= baseline + 0.005 ({base_T + 0.005:.4f}) and inversions > 0.03 <= 0.1%")


def report(df, obs, selected, args):
    f = lambda x, d=4: "-" if pd.isna(x) else f"{x:.{d}f}"
    L = ["# Mixed layer depth fix experiment: ArgoCast PI-BiLSTM loss variants", "",
         f"Finished {time.strftime('%Y-%m-%d %H:%M:%S')}. Seeds {args.seeds}; same data, splits, optimiser and early "
         "stopping as run_benchmark.py. Values are means over seeds (± std where shown).", "",
         "## Why this experiment", "",
         "In the benchmark, ArgoCast PI-BiLSTM had the largest MLD error of the learned models, with a shallow bias. In "
         "observed profiles the density between 5 and 20 dbar is often almost uniform (well-mixed surface layer); the "
         "current model almost never forecasts this and instead keeps a margin of stratification everywhere, because the "
         "stability penalty max(0, -dsigma/dz) punishes any forecast that tips slightly negative.", "",
         "## Variants", "", "| Variant | Stability weight | Tolerance (kg m⁻³ per interval) | Surface term weight |", "|---|---|---|---|"]
    for v in args.variants:
        c = VARIANTS[v]
        L.append(f"| {LABELS[v]} (`{v}`) | {c['lam_stab']:g} | {c['tol']:g} | {c['lam_surf']:g} |")
    L += ["", "Surface term = MSE between forecast and observed density differences over 5-20, 20-50 and 50-75 dbar.",
          "'Mixed 5-20 (%)' = share of profiles whose density rises by at most 0.005 kg m⁻³ between 5 and 20 dbar.", ""]
    for split in df["split"].unique():
        d = df[df["split"] == split]
        sel, why = selected[split]
        o = obs[split]
        L += [f"## {split}", "",
              f"Observed: mixed 5-20 dbar in {o['val']['pct_mixed_5_20']:.1f}% (validation) / {o['test']['pct_mixed_5_20']:.1f}% "
              f"(test) of profiles; mean MLD {o['test']['mld_mean']:.1f} m (test); inversions > 0.03 kg m⁻³ in "
              f"{o['test']['inv_003_pct']:.2f}% of observed test profiles.", "",
              f"**Selected on validation: {LABELS.get(sel, sel) if sel else 'none'}**" + (f" ({why})." if sel else f": {why}."), ""]
        for k, title in (("val", "Validation (used for selection)"), ("test", "Test (reported)")):
            g = d[d["set"] == k].groupby("variant")
            m, s = g.mean(numeric_only=True), g.std(numeric_only=True, ddof=0)
            L += [f"### {title}", "",
                  "| Variant | Temp RMSE (°C) | Sal RMSE (PSU) | MLD RMSE (m) | MLD bias (m) | Mixed 5-20 (%) | Inversions any (%) | > 0.01 (%) | > 0.03 (%) | D20 RMSE (m) | OHC RMSE |",
                  "|---|---|---|---|---|---|---|---|---|---|---|"]
            for v in [v for v in args.variants if v in m.index]:
                mark = " **(selected)**" if v == sel else ""
                L.append(f"| {LABELS[v]}{mark} | {f(m.loc[v,'temp_rmse'])} ± {f(s.loc[v,'temp_rmse'])} | {f(m.loc[v,'sal_rmse'])} | "
                         f"{f(m.loc[v,'mld_rmse'],2)} ± {f(s.loc[v,'mld_rmse'],2)} | {m.loc[v,'mld_bias']:+.2f} | "
                         f"{m.loc[v,'pct_mixed_5_20']:.1f} | {m.loc[v,'inv_any_pct']:.2f} | {m.loc[v,'inv_001_pct']:.2f} | "
                         f"{m.loc[v,'inv_003_pct']:.2f} | {f(m.loc[v,'d20_rmse'],2)} | {f(m.loc[v,'ohc_rmse'],3)} |")
            L.append("")
    L += ["## Reading the result", "",
          "- The fix works if the selected variant lowers MLD RMSE and the MLD bias towards the no-physics model while keeping "
          "inversions > 0.03 kg m⁻³ near zero and temperature RMSE essentially unchanged.",
          "- A tolerance allows tiny inversions by design, so the 'any' inversion rate is expected to rise; the "
          "> 0.01 and > 0.03 kg m⁻³ rates show whether physically meaningful inversions stay suppressed.",
          "- If no variant improves MLD, the trade-off between a hard stability constraint and mixed-layer realism is "
          "itself the finding.", "",
          "## Files", "", "- `mld_fix_seeds.csv`: one row per split, variant, seed and set",
          "- `mld_fix_results.json`: means, selection and observed reference values",
          "- `figures/fig_mldfix_*.png|pdf`", ""]
    return "\n".join(L)


def figures(out, df, hist_all, obs, selected, args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"savefig.dpi": 300, "savefig.bbox": "tight", "font.size": 8, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25, "legend.frameon": False})
    fd = out / "figures"; fd.mkdir(parents=True, exist_ok=True)
    splits = list(df["split"].unique())

    fig, axes = plt.subplots(1, len(splits), figsize=(5.2 * len(splits), 3.8), squeeze=False)
    for ax, sp in zip(axes[0], splits):
        bins = np.linspace(-0.05, 0.4, 46)
        ax.hist(np.clip(obs[sp]["_hist_obs"], -0.05, 0.4), bins, color="black", histtype="step", lw=1.6, label="Observed")
        show = [v for v in ("no_physics", "baseline", selected[sp][0]) if v and v in hist_all[sp]]
        for v in dict.fromkeys(show):
            ax.hist(np.clip(hist_all[sp][v], -0.05, 0.4), bins, color=COLORS[v], histtype="step", lw=1.1, label=LABELS[v])
        ax.axvline(0.03, color="grey", ls=":", lw=0.8)
        ax.set_xlabel("Density increase 5 → 20 dbar (kg m$^{-3}$)"); ax.set_ylabel("Test profiles")
        ax.set_title(sp.replace("_", " "), fontsize=8); ax.legend(fontsize=6)
    fig.suptitle("Near-surface stratification: observed vs forecast (dotted line = MLD threshold)", fontsize=9)
    rb.add_note(fig); fig.tight_layout()
    fig.savefig(fd / "fig_mldfix_surface_stratification.png"); fig.savefig(fd / "fig_mldfix_surface_stratification.pdf"); plt.close(fig)

    fig, axes = plt.subplots(1, len(splits), figsize=(5.2 * len(splits), 3.8), squeeze=False)
    for ax, sp in zip(axes[0], splits):
        m = df[(df["split"] == sp) & (df["set"] == "test")].groupby("variant").mean(numeric_only=True)
        for v in [v for v in args.variants if v in m.index]:
            ax.scatter(m.loc[v, "inv_003_pct"], m.loc[v, "mld_rmse"], s=60 if v == selected[sp][0] else 35,
                       color=COLORS[v], edgecolor="black", lw=0.5, zorder=3)
            ax.annotate(LABELS[v], (m.loc[v, "inv_003_pct"], m.loc[v, "mld_rmse"]), fontsize=6, xytext=(4, 3),
                        textcoords="offset points")
        ax.set_xscale("symlog", linthresh=0.1)
        ax.set_xlabel("Inversions > 0.03 kg m$^{-3}$ (% of test profiles)"); ax.set_ylabel("MLD RMSE (m)")
        ax.set_title(sp.replace("_", " "), fontsize=8)
    fig.suptitle("Mixed layer depth error vs physical consistency, by loss variant (test, mean over seeds)", fontsize=9)
    rb.add_note(fig); fig.tight_layout()
    fig.savefig(fd / "fig_mldfix_tradeoff.png"); fig.savefig(fd / "fig_mldfix_tradeoff.pdf"); plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", choices=["chronological", "unseen_floats", "both"], default="both")
    ap.add_argument("--proc-dir", type=Path,
                    default=Path(os.environ.get("FC_DATA_DIR", rb.BACKEND / "research" / "pipeline" / "data")) / "processed")
    ap.add_argument("--out", type=Path, default=HERE / "outputs" / "mld_fix")
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=list(VARIANTS))
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--allow-cpu", action="store_true")
    ap.add_argument("--threads", type=int, default=None)
    args = ap.parse_args()
    if "baseline" not in args.variants:
        args.variants = ["baseline"] + args.variants
    if not torch.cuda.is_available() and not args.allow_cpu:
        sys.exit("No CUDA GPU is visible to PyTorch, so nothing was run (add --allow-cpu to run on CPU anyway).")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.threads:
        torch.set_num_threads(args.threads)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    args.out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    splits = ["chronological", "unseen_floats"] if args.split == "both" else [args.split]
    rows, obs, hist, selected = [], {}, {}, {}
    for sp in splits:
        r, o, h = run_split(sp, args, device, args.out)
        rows += r; obs[sp] = o; hist[sp] = h
        selected[sp] = select(pd.DataFrame(r))
        print(f"[{sp}] selected on validation: {selected[sp][0]} - {selected[sp][1]}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(args.out / "mld_fix_seeds.csv", index=False)
    means = {sp: {k: df[(df["split"] == sp) & (df["set"] == k)].groupby("variant").mean(numeric_only=True)
                  .drop(columns=["seed"]).to_dict(orient="index") for k in ("val", "test")} for sp in splits}
    payload = {"finished": time.strftime("%Y-%m-%d %H:%M:%S"), "seeds": args.seeds, "variants": {v: VARIANTS[v] for v in args.variants},
               "selected": {sp: {"variant": s[0], "reason": s[1]} for sp, s in selected.items()},
               "observed": {sp: {k: v for k, v in o.items() if not k.startswith("_")} for sp, o in obs.items()},
               "means": means}
    (args.out / "mld_fix_results.json").write_text(json.dumps(rb.to_jsonable(payload), indent=2))
    try:
        figures(args.out, df, hist, obs, selected, args)
    except Exception as exc:
        print(f"figures failed: {exc}")
    (args.out / "MLD_FIX_RESULTS.md").write_text(report(df, obs, selected, args), encoding="utf-8")
    print(f"Saved {args.out / 'MLD_FIX_RESULTS.md'}  (total {(time.time() - t0) / 60:.1f} min)")


if __name__ == "__main__":
    main()
