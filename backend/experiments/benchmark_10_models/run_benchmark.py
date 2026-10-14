#!/usr/bin/env python
"""FloatChat 10-model benchmark: train, evaluate and record every comparison model.

Models (all forecast the full next-cycle 16-level T and S profile from 3 past cycles)
  1  Persistence               next profile = last observed profile (t-1)
  2  Climatology               monthly mean profile of the training targets
  3  Ridge                     linear regression on the 96 inputs, alpha chosen on validation
  4  LightGBM                  one gradient-boosted tree model per output (32), early stopping on validation
  5  GRU                       2-layer GRU, 128 units
  6  Seq2Seq LSTM + attention  LSTM encoder-decoder with additive attention
  7  TCN                       causal dilated temporal convolutional network
  8  DLinear                   trend / remainder decomposition + per-channel linear layers
  9  N-HiTS                    multi-rate pooled MLP stacks with doubly residual stacking
  10 TFT                       compact Temporal Fusion Transformer
  Reference rows: Bi-LSTM (MSE only) and FloatChat Bi-LSTM (physics-constrained loss)

All neural models use the same data loss as FloatChat without the physics terms
(MSE_T + 2.5 MSE_S), the same optimiser (AdamW, lr 1e-3, wd 1e-4, cosine schedule),
gradient clipping, batch size and early stopping on validation temperature RMSE.
Only FloatChat Bi-LSTM adds the stability and thermocline penalties.

Datasets
  --dataset canonical     backend/data/processed/argo_floats_canonical.csv, the 109-float
                          set used in the paper, with the identical float-level split of
                          scripts/train_model.py (seed 42; 5 test floats, 916 test windows)
  --dataset arabian2000   windows and splits written by experiments/arabian_sea_2000_2026
                          stage 3 (s03_split.py); choose --split chronological or unseen_floats

Examples
  python run_benchmark.py --dataset canonical
  python run_benchmark.py --dataset arabian2000 --split chronological
  python run_benchmark.py --dataset arabian2000 --split unseen_floats --seeds 42

Outputs: outputs/<dataset>_<split>/ metrics.json, metrics.csv, per_depth_*.csv,
         figures/*.png|pdf, predictions/*.npz, models/*.pt, RESULTS.md
"""
from __future__ import annotations

import argparse
import warnings
import json
import os
import platform
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=UserWarning, module="lightgbm")
warnings.filterwarnings("ignore", message=".*eval_set.*")
HERE = Path(__file__).resolve().parent
BACKEND = HERE.parents[1]
sys.path.insert(0, str(BACKEND))
sys.path.insert(0, str(HERE))

from app.ml.loss import PhysicsConstrainedLoss  # noqa: E402
from models import build, n_params  # noqa: E402

DEPTHS = np.array([5, 20, 50, 75, 100, 150, 200, 250, 300, 400, 500, 600, 700, 800, 900, 1000], float)
N_LEV = 16
TEMP_COLS = [f"temp_{int(d)}" for d in DEPTHS]
SAL_COLS = [f"sal_{int(d)}" for d in DEPTHS]
THERM_IDX = [2, 3, 4, 5]
DEEP_IDX = [10, 11, 12, 13, 14, 15]
ARCTIC_WMOS = {6903058, 6903059, 6903060, 6903062, 6903063}

MODEL_ORDER = ["Persistence", "Climatology", "Ridge", "LightGBM", "GRU", "Seq2Seq LSTM + attention",
               "TCN", "DLinear", "N-HiTS", "TFT", "Bi-LSTM (MSE only)", "FloatChat Bi-LSTM"]
NEURAL = ["GRU", "Seq2Seq LSTM + attention", "TCN", "DLinear", "N-HiTS", "TFT",
          "Bi-LSTM (MSE only)", "FloatChat Bi-LSTM"]
REFERENCE = "FloatChat Bi-LSTM"
COLORS = {
    "Persistence": "#999999", "Climatology": "#CC79A7", "Ridge": "#E69F00", "LightGBM": "#56B4E9",
    "GRU": "#8C6D31", "Seq2Seq LSTM + attention": "#D55E00", "TCN": "#7F3C8D", "DLinear": "#F0E442",
    "N-HiTS": "#11A579", "TFT": "#E73F74", "Bi-LSTM (MSE only)": "#009E73", "FloatChat Bi-LSTM": "#0072B2",
}
SHORT = {"Seq2Seq LSTM + attention": "Seq2Seq+Attn", "Bi-LSTM (MSE only)": "Bi-LSTM (MSE)",
         "FloatChat Bi-LSTM": "FloatChat"}


# ------------------------------------------------------------------ helpers
def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def mae(a, b):
    return float(np.mean(np.abs(np.asarray(a) - np.asarray(b))))


def r2(p, t):
    ss = np.sum((t - t.mean()) ** 2)
    return float(1 - np.sum((t - p) ** 2) / ss) if ss > 0 else float("nan")


def sigma_poly(t, s):
    return 28.14 - 0.0735 * t - 0.00469 * t ** 2 + 0.802 * (s - 35.0)


def inv_poly(t, s, tol=1e-6):
    return float(((np.diff(sigma_poly(t, s), axis=1) / np.diff(DEPTHS)) < -tol).any(axis=1).mean() * 100)


def inv_teos10(t, s, lat, lon, tol=1e-6):
    import gsw
    p = np.broadcast_to(DEPTHS, t.shape)
    la = np.broadcast_to(lat[:, None], t.shape); lo = np.broadcast_to(lon[:, None], t.shape)
    sa = gsw.SA_from_SP(s, p, lo, la)
    sig = gsw.sigma0(sa, gsw.CT_from_t(sa, t, p))
    return float(((np.diff(sig, axis=1) / np.diff(DEPTHS)) < -tol).any(axis=1).mean() * 100)


def to_jsonable(o):
    if isinstance(o, dict):
        return {str(k): to_jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [to_jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return to_jsonable(o.tolist())
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    return o


# --------------------------------------------------------------------- data
def load_canonical(csv: Path, seed: int = 42):
    """Exact reproduction of scripts/train_model.py --split-mode spatial."""
    df = pd.read_csv(csv).dropna(subset=TEMP_COLS + SAL_COLS).reset_index(drop=True)
    phys = ((df[TEMP_COLS].ge(-2.0) & df[TEMP_COLS].le(35.0)).all(axis=1)
            & (df[SAL_COLS].ge(30.0) & df[SAL_COLS].le(42.0)).all(axis=1))
    df = df[phys].reset_index(drop=True)
    wmos = sorted(df["wmo"].unique().tolist())
    arctic = sorted(set(wmos) & ARCTIC_WMOS)
    arabian = sorted(set(wmos) - ARCTIC_WMOS)
    train_ar, rest_ar = train_test_split(arabian, test_size=10, random_state=seed)
    val_ar, test_ar = train_test_split(rest_ar, test_size=5, random_state=seed)
    split_wmos = {"train": sorted(train_ar + arctic[:-1]), "val": sorted(val_ar + arctic[-1:]),
                  "test": sorted(test_ar)}

    def windows(frame):
        X, yT, yS, meta = [], [], [], []
        for wmo, g in frame.sort_values(["wmo", "cycle"]).groupby("wmo"):
            g = g.dropna(subset=TEMP_COLS + SAL_COLS + ["cycle"]).reset_index(drop=True)
            if len(g) < 4:
                continue
            t, s = g[TEMP_COLS].to_numpy(np.float32), g[SAL_COLS].to_numpy(np.float32)
            for i in range(len(g) - 3):
                X.append(np.concatenate([t[i:i + 3], s[i:i + 3]], axis=1))
                yT.append(t[i + 3]); yS.append(s[i + 3])
                r = g.iloc[i + 3]
                meta.append((int(wmo), int(r["cycle"]), r["date"], float(r["latitude"]), float(r["longitude"])))
        m = pd.DataFrame(meta, columns=["wmo", "cycle", "target_date", "lat", "lon"])
        m["target_date"] = pd.to_datetime(m["target_date"])
        m["month"] = m["target_date"].dt.month
        return {"X": np.stack(X), "yT": np.stack(yT), "yS": np.stack(yS), "meta": m}

    parts = {k: windows(df[df["wmo"].isin(w)]) for k, w in split_wmos.items()}
    info = {"source": str(csv.relative_to(BACKEND)) if csv.is_relative_to(BACKEND) else str(csv),
            "split": "float-level, stratified arctic/arabian, seed 42 (identical to scripts/train_model.py)",
            "floats": {k: len(v) for k, v in split_wmos.items()}, "test_floats": split_wmos["test"]}
    return parts, info


def load_arabian2000(proc_dir: Path, split: str):
    """Windows/splits written by experiments/arabian_sea_2000_2026/s03_split.py."""
    w = np.load(proc_dir / "windows.npz", allow_pickle=False)
    s = np.load(proc_dir / f"split_{split}.npz")
    meta = pd.DataFrame({"wmo": w["wmo"], "cycle": w["cycle"], "lat": w["lat"], "lon": w["lon"],
                         "month": w["month"], "target_date": pd.to_datetime(w["target_date"], unit="s")})
    parts = {k: {"X": w["X"][s[k]], "yT": w["yT"][s[k]], "yS": w["yS"][s[k]],
                 "meta": meta.iloc[s[k]].reset_index(drop=True)} for k in ("train", "val", "test")}
    info = {"source": str(proc_dir), "split": split,
            "floats": {k: int(parts[k]["meta"]["wmo"].nunique()) for k in parts}}
    return parts, info


class Scaler:
    """Per-depth standardisation fitted on training targets only."""

    def __init__(self, yT, yS):
        self.t = StandardScaler().fit(yT.astype(np.float64))
        self.s = StandardScaler().fit(yS.astype(np.float64))

    def X(self, X):
        out = np.empty_like(X, dtype=np.float32)
        for k in range(X.shape[1]):
            out[:, k, :N_LEV] = self.t.transform(X[:, k, :N_LEV])
            out[:, k, N_LEV:] = self.s.transform(X[:, k, N_LEV:])
        return out

    def Y(self, yT, yS):
        return np.concatenate([self.t.transform(yT), self.s.transform(yS)], 1).astype(np.float32)

    def inv(self, Y):
        return self.t.inverse_transform(Y[:, :N_LEV]), self.s.inverse_transform(Y[:, N_LEV:])


# ------------------------------------------------------------ neural models
@torch.no_grad()
def predict(model, X, bs=4096):
    model.eval()
    T, S = [], []
    for i in range(0, len(X), bs):
        a, b = model(X[i:i + bs]); T.append(a.cpu().numpy()); S.append(b.cpu().numpy())
    return np.concatenate([np.concatenate(T), np.concatenate(S)], 1)


def train_neural(name, parts, sc, seed, args, device):
    set_seed(seed)
    model = build(name).to(device)
    physics = name == REFERENCE
    lam = dict(lambda_stability=10.0, lambda_therm=1.5) if physics else dict(lambda_stability=0.0, lambda_therm=0.0)
    crit = PhysicsConstrainedLoss(lambda_sal=2.5, temp_mean=sc.t.mean_, temp_std=sc.t.scale_,
                                  sal_mean=sc.s.mean_, sal_std=sc.s.scale_, **lam).to(device)
    Ytr = sc.Y(parts["train"]["yT"], parts["train"]["yS"])
    # Whole training set lives on the device; batches are index slices (much faster
    # on a GPU than a DataLoader for a dataset this small).
    Xtr = torch.tensor(sc.X(parts["train"]["X"]), device=device)
    YTtr = torch.tensor(Ytr[:, :N_LEV], device=device)
    YStr = torch.tensor(Ytr[:, N_LEV:], device=device)
    gen = torch.Generator(device="cpu").manual_seed(seed)
    n_tr = len(Xtr)
    Xva = torch.tensor(sc.X(parts["val"]["X"]), device=device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-6)
    best, best_state, bad, hist = np.inf, None, 0, []
    t0 = time.time()
    for ep in range(1, args.epochs + 1):
        model.train()
        tot = torch.zeros((), device=device)
        perm = torch.randperm(n_tr, generator=gen).to(device)
        for i in range(0, n_tr, args.batch_size):
            idx = perm[i:i + args.batch_size]
            X, yT, yS = Xtr[idx], YTtr[idx], YStr[idx]
            opt.zero_grad(set_to_none=True)
            pT, pS = model(X)
            loss, _ = crit(pT, yT, pS, yS)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tot += loss.detach() * len(idx)
        tot, n = float(tot), n_tr
        sched.step()
        vT, vS = sc.inv(predict(model, Xva))
        rT, rS = rmse(vT, parts["val"]["yT"]), rmse(vS, parts["val"]["yS"])
        hist.append({"epoch": ep, "train_loss": tot / n, "val_temp_rmse": rT, "val_sal_rmse": rS})
        if rT < best - 1e-6:
            best, bad = rT, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
        if bad >= args.patience:
            break
    model.load_state_dict(best_state)
    return model, hist, time.time() - t0


# ------------------------------------------------------------------ metrics
def metric_row(pT, pS, tT, tS, lat, lon):
    return {"temp_rmse": rmse(pT, tT), "temp_mae": mae(pT, tT), "temp_r2": r2(pT, tT),
            "therm_rmse": rmse(pT[:, THERM_IDX], tT[:, THERM_IDX]),
            "deep_rmse": rmse(pT[:, DEEP_IDX], tT[:, DEEP_IDX]),
            "sal_rmse": rmse(pS, tS), "sal_mae": mae(pS, tS), "sal_r2": r2(pS, tS),
            "inversion_teos10_pct": inv_teos10(pT, pS, lat, lon), "inversion_poly_pct": inv_poly(pT, pS)}


def bootstrap(preds, tT, tS, groups, B, seed=0):
    """Float-level block bootstrap with the SAME resamples for every model, so paired
    differences (model vs FloatChat, model vs persistence) are valid."""
    rng = np.random.default_rng(seed)
    uniq, inv = np.unique(groups, return_inverse=True)
    G = len(uniq)
    cnt = np.bincount(inv, minlength=G).astype(float)
    W = rng.multinomial(G, np.full(G, 1 / G), size=B).astype(float)
    n = W @ cnt
    boot = {}
    for m, (pT, pS) in preds.items():
        sT = np.bincount(inv, weights=((pT - tT) ** 2).mean(1), minlength=G)
        sS = np.bincount(inv, weights=((pS - tS) ** 2).mean(1), minlength=G)
        boot[m] = (np.sqrt(W @ sT / n), np.sqrt(W @ sS / n))
    q = lambda a: [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]
    out = {}
    for m, (bT, bS) in boot.items():
        r = {"temp_rmse_ci": q(bT), "sal_rmse_ci": q(bS)}
        sk = 1 - bT / boot["Persistence"][0]
        r["temp_skill_ci"] = q(sk)
        r["p_not_better_than_persistence"] = float((sk <= 0).mean())
        if REFERENCE in boot and m != REFERENCE:
            dT = bT - boot[REFERENCE][0]; dS = bS - boot[REFERENCE][1]
            r["temp_rmse_diff_vs_floatchat_ci"] = q(dT)
            r["sal_rmse_diff_vs_floatchat_ci"] = q(dS)
            r["p_beats_floatchat_temp"] = float((dT < 0).mean())
            r["p_beats_floatchat_sal"] = float((dS < 0).mean())
        out[m] = r
    return out


# ------------------------------------------------------------------ figures
def make_figures(fig_dir, tag, results, extras):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"savefig.dpi": 300, "savefig.bbox": "tight", "font.size": 8,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.alpha": 0.25, "legend.frameon": False})

    def save(fig, name):
        fig.savefig(fig_dir / f"{name}.png"); fig.savefig(fig_dir / f"{name}.pdf"); plt.close(fig)

    models = [m for m in MODEL_ORDER if m in results]
    lbl = [SHORT.get(m, m) for m in models]
    x = np.arange(len(models))

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    for ax, key, ylab in zip(axes, ["temp_rmse", "sal_rmse", "inversion_teos10_pct"],
                             ["Temperature RMSE (°C)", "Salinity RMSE (PSU)", "Density inversions (% of profiles)"]):
        v = np.array([results[m][key] for m in models])
        ax.bar(x, v, color=[COLORS[m] for m in models], edgecolor="black", linewidth=0.3)
        if key != "inversion_teos10_pct":
            ci = np.array([results[m][key + "_ci"] for m in models])
            ax.errorbar(x, v, yerr=[np.clip(v - ci[:, 0], 0, None), np.clip(ci[:, 1] - v, 0, None)],
                        fmt="none", ecolor="black", capsize=2, lw=0.7)
            lo = min(ci[:, 0].min(), v.min())
            hi = max(ci[:, 1][[m != "Climatology" for m in models]].max(), v[[m != "Climatology" for m in models]].max())
            if results.get("Climatology", {}).get(key, 0) > 1.6 * hi:
                ax.set_ylim(lo * 0.9, hi * 1.08)
                i = models.index("Climatology")
                ax.annotate(f"{v[i]:.3f}", (i, hi * 1.05), ha="center", fontsize=7)
        else:
            obs = extras["observed_inversion_teos10_pct"]
            ax.axhline(obs, color="black", ls="--", lw=0.8, label=f"observed profiles {obs:.1f}%")
            ax.legend(fontsize=7)
        ax.set_xticks(x); ax.set_xticklabels(lbl, rotation=40, ha="right", fontsize=7)
        ax.set_ylabel(ylab)
    fig.suptitle(f"Test accuracy (95% float-bootstrap CI) and physical consistency - {tag}", fontsize=9)
    fig.tight_layout(); save(fig, "fig_bench_model_comparison")

    fig, axes = plt.subplots(1, 3, figsize=(11, 4.3), sharey=True)
    for m in models:
        if m == "Climatology":
            continue
        st = dict(color=COLORS[m], lw=2.2 if m == REFERENCE else 1.0, ls="--" if m == "Persistence" else "-")
        axes[0].plot(results[m]["per_depth_temp_rmse"], DEPTHS, label=SHORT.get(m, m), **st)
        axes[1].plot(results[m]["per_depth_sal_rmse"], DEPTHS, **st)
        if m != "Persistence":
            sk = 1 - np.array(results[m]["per_depth_temp_rmse"]) / np.array(results["Persistence"]["per_depth_temp_rmse"])
            axes[2].plot(sk * 100, DEPTHS, **st)
    axes[2].axvline(0, color="black", lw=0.8)
    axes[0].set_ylim(1020, 0); axes[0].set_ylabel("Pressure (dbar)")
    axes[0].set_xlabel("Temperature RMSE (°C)"); axes[1].set_xlabel("Salinity RMSE (PSU)")
    axes[2].set_xlabel("Temperature skill vs persistence (%)")
    axes[0].legend(fontsize=6, loc="lower right")
    fig.suptitle(f"Error by depth (climatology omitted for scale) - {tag}", fontsize=9)
    fig.tight_layout(); save(fig, "fig_bench_depth_rmse")

    nm = [m for m in models if results[m].get("n_params")]
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    for m in nm:
        ax.scatter(results[m]["n_params"], results[m]["temp_rmse"], s=40, color=COLORS[m], edgecolor="black", lw=0.4, zorder=3)
        ax.annotate(SHORT.get(m, m), (results[m]["n_params"], results[m]["temp_rmse"]), fontsize=6,
                    xytext=(4, 3), textcoords="offset points")
    ax.set_xscale("log"); ax.set_xlabel("Trainable parameters (log scale)"); ax.set_ylabel("Test temperature RMSE (°C)")
    ax.set_title("Model size vs accuracy", fontsize=9)
    save(fig, "fig_bench_params_vs_rmse")

    if "tft_var_weights" in extras:
        w = np.array(extras["tft_var_weights"])  # (3, 32)
        fig, axes = plt.subplots(1, 2, figsize=(8, 3.0), sharey=True)
        for ax, sl, ttl in ((axes[0], slice(0, 16), "Temperature inputs"), (axes[1], slice(16, 32), "Salinity inputs")):
            im = ax.imshow(w[:, sl].T, aspect="auto", cmap="viridis", vmin=0, vmax=w.max())
            ax.set_xticks([0, 1, 2]); ax.set_xticklabels(["t-3", "t-2", "t-1"])
            ax.set_yticks(range(16)); ax.set_yticklabels([str(int(d)) for d in DEPTHS], fontsize=6)
            ax.set_title(ttl, fontsize=8); ax.grid(False)
        axes[0].set_ylabel("Input depth (dbar)")
        fig.colorbar(im, ax=axes, shrink=0.85, label="Mean selection weight")
        fig.suptitle("TFT variable-selection weights (mean over test windows)", fontsize=9)
        save(fig, "fig_bench_tft_variable_selection")

    att = {k: v for k, v in extras.items() if k.endswith("_temporal_attention")}
    if att:
        fig, ax = plt.subplots(figsize=(4.6, 3.0))
        width = 0.8 / len(att)
        for i, (k, v) in enumerate(att.items()):
            name = k.replace("_temporal_attention", "")
            ax.bar(np.arange(3) + i * width - 0.4 + width / 2, v, width, color=COLORS.get(name, "#777"), label=SHORT.get(name, name))
        ax.set_xticks(range(3)); ax.set_xticklabels(["t-3", "t-2", "t-1"])
        ax.set_ylabel("Mean attention weight"); ax.legend(fontsize=7)
        ax.set_title("Attention over past cycles", fontsize=9)
        save(fig, "fig_bench_temporal_attention")


# --------------------------------------------------------------------- main
def run(args):
    if args.dataset == "canonical":
        parts, info = load_canonical(args.csv)
        tag = "canonical_spatial"
    else:
        parts, info = load_arabian2000(args.proc_dir, args.split)
        tag = f"arabian2000_{args.split}"
    out = args.out / tag
    for sub in ("figures", "predictions", "models"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    if args.threads:
        torch.set_num_threads(args.threads)
    wanted = [m for m in MODEL_ORDER if m in (args.models or MODEL_ORDER)]
    for k in ("train", "val", "test"):
        print(f"{k}: {len(parts[k]['X']):,} windows, {parts[k]['meta']['wmo'].nunique()} floats")
    print(f"Device: {device} | models: {len(wanted)} | seeds: {args.seeds}", flush=True)

    sc = Scaler(parts["train"]["yT"], parts["train"]["yS"])
    te, va = parts["test"], parts["val"]
    tT, tS = te["yT"].astype(float), te["yS"].astype(float)
    lat, lon = te["meta"]["lat"].to_numpy(), te["meta"]["lon"].to_numpy()
    Xtr_f = sc.X(parts["train"]["X"]).reshape(len(parts["train"]["X"]), -1)
    Xva_f = sc.X(va["X"]).reshape(len(va["X"]), -1)
    Xte_s = sc.X(te["X"]); Xte_f = Xte_s.reshape(len(Xte_s), -1)
    Ytr = sc.Y(parts["train"]["yT"], parts["train"]["yS"])
    Yva = sc.Y(va["yT"], va["yS"])

    preds, info_m, seed_rows, histories, extras = {}, {}, {}, {}, {}

    def timed_predict(fn):
        t0 = time.time(); p = fn(); return p, (time.time() - t0) / len(Xte_s) * 1000

    if "Persistence" in wanted:
        preds["Persistence"] = (te["X"][:, 2, :N_LEV].astype(float), te["X"][:, 2, N_LEV:].astype(float))
        info_m["Persistence"] = {"n_params": 0, "fit_seconds": 0.0, "settings": "next = last observed profile"}
    if "Climatology" in wanted:
        m_tr = parts["train"]["meta"]["month"].to_numpy()
        clT = {m: parts["train"]["yT"][m_tr == m].mean(0) for m in range(1, 13) if (m_tr == m).any()}
        clS = {m: parts["train"]["yS"][m_tr == m].mean(0) for m in range(1, 13) if (m_tr == m).any()}
        gT, gS = parts["train"]["yT"].mean(0), parts["train"]["yS"].mean(0)
        mt = te["meta"]["month"].to_numpy()
        preds["Climatology"] = (np.stack([clT.get(m, gT) for m in mt]).astype(float), np.stack([clS.get(m, gS) for m in mt]).astype(float))
        info_m["Climatology"] = {"n_params": 0, "fit_seconds": 0.0, "settings": f"monthly means of training targets ({len(clT)} months)"}

    if "Ridge" in wanted:
        t0 = time.time()
        scores = {}
        for a in (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0):
            p = Ridge(alpha=a).fit(Xtr_f, Ytr).predict(Xva_f)
            vT, vS = sc.inv(p)
            scores[a] = rmse(vT, va["yT"])
        alpha = min(scores, key=scores.get)
        rg = Ridge(alpha=alpha).fit(Xtr_f, Ytr)
        fit_s = time.time() - t0
        p, ms = timed_predict(lambda: rg.predict(Xte_f))
        preds["Ridge"] = sc.inv(p)
        info_m["Ridge"] = {"n_params": int(rg.coef_.size + rg.intercept_.size), "fit_seconds": fit_s,
                           "infer_ms_per_window": ms, "settings": f"alpha={alpha} (validation grid {list(scores)})",
                           "val_temp_rmse_by_alpha": scores}

    if "LightGBM" in wanted:
        import lightgbm as lgb
        print("LightGBM: 32 regressors with early stopping ...", flush=True)
        t0 = time.time()
        # Native API (lgb.train) rather than LGBMRegressor: the scikit-learn wrapper breaks
        # when the installed scikit-learn is newer than the LightGBM release.
        params = dict(objective="regression", learning_rate=0.05, num_leaves=31, min_child_samples=20,
                      subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
                      seed=42, num_threads=max(args.lgb_jobs, 0), verbose=-1)
        regs, iters = [], []
        for j in range(Ytr.shape[1]):
            dtr = lgb.Dataset(Xtr_f, Ytr[:, j], free_raw_data=False)
            dva = lgb.Dataset(Xva_f, Yva[:, j], reference=dtr, free_raw_data=False)
            b = lgb.train(params, dtr, num_boost_round=args.lgb_estimators, valid_sets=[dva],
                          callbacks=[lgb.early_stopping(50, verbose=False)])
            regs.append(b); iters.append(int(b.best_iteration or args.lgb_estimators))
        fit_s = time.time() - t0
        p, ms = timed_predict(lambda: np.stack([b.predict(Xte_f, num_iteration=b.best_iteration) for b in regs], 1))
        preds["LightGBM"] = sc.inv(p)
        n_leaves = sum(int(b.trees_to_dataframe()["node_index"].shape[0]) for b in regs)
        info_m["LightGBM"] = {"n_params": n_leaves, "fit_seconds": fit_s, "infer_ms_per_window": ms,
                              "settings": f"lr 0.05, 31 leaves, subsample 0.8, colsample 0.8, early stop 50 on val; "
                                          f"trees per output median {int(np.median(iters))} (range {min(iters)}-{max(iters)})",
                              "best_iterations": iters, "n_params_note": "total tree nodes, not trainable weights"}
        # TreeSHAP-free summary: gain importance aggregated by input time step and variable
        gain = np.zeros(96)
        for r in regs:
            gain += r.feature_importance("gain")
        gain = gain.reshape(3, 32) / gain.sum()
        extras["lightgbm_gain_by_step"] = gain.sum(1).tolist()
        extras["lightgbm_gain_by_step_var"] = {"temp": gain[:, :16].sum(1).tolist(), "sal": gain[:, 16:].sum(1).tolist()}
        print(f"  done in {fit_s:.0f}s", flush=True)

    Xte_t = torch.tensor(Xte_s, device=device)
    for name in [m for m in NEURAL if m in wanted]:
        seed_rows[name] = []
        for j, seed in enumerate(args.seeds):
            model, hist, el = train_neural(name, parts, sc, seed, args, device)
            p, ms = timed_predict(lambda: predict(model, Xte_t))
            pT, pS = sc.inv(p)
            row = metric_row(pT, pS, tT, tS, lat, lon)
            row.update(seed=seed, epochs=len(hist), best_epoch=int(np.argmin([h["val_temp_rmse"] for h in hist]) + 1),
                       train_seconds=el)
            seed_rows[name].append(row)
            print(f"{name:26s} seed {seed}: T {row['temp_rmse']:.4f} S {row['sal_rmse']:.4f} "
                  f"inv {row['inversion_teos10_pct']:.2f}% ep {len(hist)} {el:.0f}s", flush=True)
            if j == 0:
                preds[name] = (pT, pS)
                histories[name] = hist
                info_m[name] = {"n_params": n_params(model), "fit_seconds": el, "infer_ms_per_window": ms}
                torch.save(model.state_dict(), out / "models" / f"{name.replace(' ', '_').replace('+', 'plus').replace('(', '').replace(')', '')}_seed{seed}.pt")
                if hasattr(model, "last_attention"):
                    model.eval()
                    with torch.no_grad():
                        model(Xte_t)
                    extras[f"{name}_temporal_attention"] = model.last_attention.mean(0).cpu().numpy().tolist()
                if hasattr(model, "last_var_weights"):
                    extras["tft_var_weights"] = model.last_var_weights.mean(0).cpu().numpy().tolist()
                    w = model.last_var_weights.mean(0).cpu().numpy()
                    extras["tft_var_weights_by_step"] = w.sum(1).tolist()
                    extras["tft_var_weights_by_variable"] = {"temp": float(w[:, :16].sum() / w.sum()), "sal": float(w[:, 16:].sum() / w.sum())}

    # ---- evaluation (first-seed predictions for neural models)
    groups = te["meta"]["wmo"].to_numpy()
    boot = bootstrap(preds, tT, tS, groups, args.bootstrap)
    pers = preds["Persistence"]
    results = {}
    for m in MODEL_ORDER:
        if m not in preds:
            continue
        pT, pS = preds[m]
        row = metric_row(pT, pS, tT, tS, lat, lon)
        row.update(boot[m])
        row["temp_skill_vs_persistence"] = 1 - row["temp_rmse"] / rmse(pers[0], tT)
        row["sal_skill_vs_persistence"] = 1 - row["sal_rmse"] / rmse(pers[1], tS)
        row["per_depth_temp_rmse"] = np.sqrt(((pT - tT) ** 2).mean(0)).tolist()
        row["per_depth_sal_rmse"] = np.sqrt(((pS - tS) ** 2).mean(0)).tolist()
        row.update(info_m.get(m, {}))
        if m in seed_rows:
            d = pd.DataFrame(seed_rows[m])
            row["seeds"] = seed_rows[m]
            for k in ("temp_rmse", "sal_rmse", "therm_rmse", "deep_rmse", "inversion_teos10_pct"):
                row[f"{k}_seed_mean"] = float(d[k].mean()); row[f"{k}_seed_std"] = float(d[k].std(ddof=0))
        results[m] = row
        np.savez_compressed(out / "predictions" / f"pred_{m.replace(' ', '_').replace('+', 'plus')}.npz", T=pT, S=pS)
    extras["observed_inversion_teos10_pct"] = inv_teos10(tT, tS, lat, lon)
    extras["observed_inversion_poly_pct"] = inv_poly(tT, tS)

    env = {"python": sys.version.split()[0], "platform": platform.platform(), "torch": torch.__version__,
           "device": f"cuda:{torch.cuda.get_device_name(0)}" if device.type == "cuda" else "cpu",
           "torch_threads": torch.get_num_threads(),
           "note": "neural models trained on the device above; Ridge and LightGBM always run on CPU"}
    payload = {"tag": tag, "data": info,
               "n_windows": {k: int(len(parts[k]["X"])) for k in parts},
               "settings": {"epochs": args.epochs, "patience": args.patience, "batch_size": args.batch_size,
                            "lr": args.lr, "seeds": args.seeds, "bootstrap": args.bootstrap},
               "environment": env, "finished": time.strftime("%Y-%m-%d %H:%M:%S"),
               "models": results, "extras": extras, "histories": histories}
    (out / "metrics.json").write_text(json.dumps(to_jsonable(payload), indent=2))
    flat = [{"model": m, **{k: v for k, v in r.items() if np.isscalar(v) or v is None}} for m, r in results.items()]
    pd.DataFrame(flat).to_csv(out / "metrics.csv", index=False)
    pd.DataFrame({m: r["per_depth_temp_rmse"] for m, r in results.items()}, index=DEPTHS.astype(int)).to_csv(out / "per_depth_temp_rmse.csv", index_label="dbar")
    pd.DataFrame({m: r["per_depth_sal_rmse"] for m, r in results.items()}, index=DEPTHS.astype(int)).to_csv(out / "per_depth_sal_rmse.csv", index_label="dbar")
    make_figures(out / "figures", tag.replace("_", " "), results, extras)
    (out / "RESULTS.md").write_text(write_report(to_jsonable(payload)), encoding="utf-8")
    print(f"Saved results to {out}")
    return out


# ------------------------------------------------------------------- report
DESCRIPTIONS = {
    "Persistence": "Next profile equals the last observed profile (t-1)",
    "Climatology": "Monthly mean profile of the training targets",
    "Ridge": "Linear regression on the 96 standardised inputs, L2 penalty chosen on validation",
    "LightGBM": "One gradient-boosted tree ensemble per output (32), early stopping on validation",
    "GRU": "2-layer unidirectional GRU, 128 units, dropout 0.2",
    "Seq2Seq LSTM + attention": "2-layer LSTM encoder, one-step LSTMCell decoder, additive attention",
    "TCN": "Two residual blocks of causal dilated convolutions (kernel 2, dilations 1 and 2, 128 channels)",
    "DLinear": "Moving-average trend/remainder split, one linear map per channel (no cross-depth mixing)",
    "N-HiTS": "Three MLP stacks (512 units) with multi-rate pooling and doubly residual stacking",
    "TFT": "Compact Temporal Fusion Transformer: variable selection, LSTM encoder, interpretable attention",
    "Bi-LSTM (MSE only)": "FloatChat architecture trained without the physics terms (ablation)",
    "FloatChat Bi-LSTM": "FloatChat architecture with the TEOS-10 stability and thermocline penalties",
}


def _f(v, d=4):
    return "-" if v is None or (isinstance(v, float) and not np.isfinite(v)) else f"{v:.{d}f}"


def write_report(p):
    R, ex = p["models"], p["extras"]
    L = [f"# FloatChat 10-model benchmark: {p['tag']}", "",
         f"Finished {p['finished']}. Device: {p['environment']['device']} (Ridge and LightGBM on CPU). "
         f"Torch {p['environment']['torch']}, Python {p['environment']['python']}.", "",
         "## Data", "",
         f"- Source: `{p['data']['source']}`; split: {p['data']['split']}",
         f"- Windows: train {p['n_windows']['train']:,}, validation {p['n_windows']['val']:,}, test {p['n_windows']['test']:,}",
         f"- Floats: train {p['data']['floats']['train']}, validation {p['data']['floats']['val']}, test {p['data']['floats']['test']}",
         f"- Training settings: epochs <= {p['settings']['epochs']}, patience {p['settings']['patience']}, batch {p['settings']['batch_size']}, "
         f"lr {p['settings']['lr']}, seeds {p['settings']['seeds']}, float-level bootstrap resamples {p['settings']['bootstrap']}", "",
         "## Models", "", "| # | Model | Description | Parameters |", "|---|---|---|---|"]
    for i, m in enumerate([m for m in MODEL_ORDER if m in R], 1):
        npar = R[m].get("n_params", 0)
        npar = f"{npar:,}" + (" (tree nodes)" if m == "LightGBM" else "") if npar else "0"
        L.append(f"| {i} | {m} | {DESCRIPTIONS[m]} | {npar} |")
    L += ["", "## Main results (test set)", "",
          "RMSE values use the first seed for neural models; 95% intervals are from a float-level block bootstrap "
          "with identical resamples for all models. Skill = 1 - RMSE / RMSE(persistence).", "",
          "| Model | Temp RMSE (°C) [95% CI] | Sal RMSE (PSU) [95% CI] | Thermocline RMSE | Deep RMSE | Temp R² | Temp skill | Sal skill | Inversions TEOS-10 (%) |",
          "|---|---|---|---|---|---|---|---|---|"]
    for m in [m for m in MODEL_ORDER if m in R]:
        r = R[m]
        L.append(f"| {m} | {_f(r['temp_rmse'])} [{_f(r['temp_rmse_ci'][0])}, {_f(r['temp_rmse_ci'][1])}] | "
                 f"{_f(r['sal_rmse'])} [{_f(r['sal_rmse_ci'][0])}, {_f(r['sal_rmse_ci'][1])}] | {_f(r['therm_rmse'])} | "
                 f"{_f(r['deep_rmse'])} | {_f(r['temp_r2'])} | {r['temp_skill_vs_persistence']*100:+.1f}% | "
                 f"{r['sal_skill_vs_persistence']*100:+.1f}% | {_f(r['inversion_teos10_pct'], 2)} |")
    L += ["", f"Observed test profiles themselves contain {ex['observed_inversion_teos10_pct']:.2f}% TEOS-10 inversions "
          f"({ex['observed_inversion_poly_pct']:.2f}% with the polynomial used in the loss).", ""]

    neu = [m for m in MODEL_ORDER if m in R and "seeds" in R[m]]
    if neu:
        L += ["## Seed stability (neural models, mean ± std over seeds)", "",
              "| Model | Temp RMSE | Sal RMSE | Thermocline RMSE | Inversions (%) | Best epoch per seed | Train time per seed (s) |",
              "|---|---|---|---|---|---|---|"]
        for m in neu:
            r = R[m]
            L.append(f"| {m} | {r['temp_rmse_seed_mean']:.4f} ± {r['temp_rmse_seed_std']:.4f} | "
                     f"{r['sal_rmse_seed_mean']:.4f} ± {r['sal_rmse_seed_std']:.4f} | "
                     f"{r['therm_rmse_seed_mean']:.4f} ± {r['therm_rmse_seed_std']:.4f} | "
                     f"{r['inversion_teos10_pct_seed_mean']:.2f} ± {r['inversion_teos10_pct_seed_std']:.2f} | "
                     f"{', '.join(str(s['best_epoch']) for s in r['seeds'])} | "
                     f"{', '.join('%.0f' % s['train_seconds'] for s in r['seeds'])} |")
        L.append("")

    if REFERENCE in R:
        L += ["## Paired comparison against FloatChat Bi-LSTM", "",
              "Difference = model RMSE - FloatChat RMSE (negative means the model is more accurate). "
              "P(beats) = share of bootstrap resamples in which the model has lower RMSE.", "",
              "| Model | ΔTemp RMSE [95% CI] | P(beats FloatChat, T) | ΔSal RMSE [95% CI] | P(beats FloatChat, S) | Inversions (%) vs FloatChat |",
              "|---|---|---|---|---|---|"]
        fc_inv = R[REFERENCE]["inversion_teos10_pct"]
        for m in [m for m in MODEL_ORDER if m in R and m != REFERENCE]:
            r = R[m]
            L.append(f"| {m} | {r['temp_rmse_diff_vs_floatchat_ci'][0]:+.4f} to {r['temp_rmse_diff_vs_floatchat_ci'][1]:+.4f} | "
                     f"{r['p_beats_floatchat_temp']:.2f} | {r['sal_rmse_diff_vs_floatchat_ci'][0]:+.4f} to "
                     f"{r['sal_rmse_diff_vs_floatchat_ci'][1]:+.4f} | {r['p_beats_floatchat_sal']:.2f} | "
                     f"{r['inversion_teos10_pct']:.2f} vs {fc_inv:.2f} |")
        L.append("")

    L += ["## Error by depth: temperature RMSE (°C)", "",
          "| dbar | " + " | ".join(SHORT.get(m, m) for m in MODEL_ORDER if m in R) + " |",
          "|---|" + "---|" * len([m for m in MODEL_ORDER if m in R])]
    for k, d in enumerate(DEPTHS.astype(int)):
        L.append(f"| {d} | " + " | ".join(f"{R[m]['per_depth_temp_rmse'][k]:.3f}" for m in MODEL_ORDER if m in R) + " |")
    L += ["", "## Error by depth: salinity RMSE (PSU)", "",
          "| dbar | " + " | ".join(SHORT.get(m, m) for m in MODEL_ORDER if m in R) + " |",
          "|---|" + "---|" * len([m for m in MODEL_ORDER if m in R])]
    for k, d in enumerate(DEPTHS.astype(int)):
        L.append(f"| {d} | " + " | ".join(f"{R[m]['per_depth_sal_rmse'][k]:.4f}" for m in MODEL_ORDER if m in R) + " |")
    L.append("")

    L += ["## Cost", "", "| Model | Fit time (s, first seed) | Inference (ms per window) |", "|---|---|---|"]
    for m in [m for m in MODEL_ORDER if m in R]:
        L.append(f"| {m} | {_f(R[m].get('fit_seconds'), 1)} | {_f(R[m].get('infer_ms_per_window'), 4)} |")
    L.append("")

    interp = []
    if "tft_var_weights_by_step" in ex:
        w = np.array(ex["tft_var_weights_by_step"]); w = w / w.sum()
        v = ex["tft_var_weights_by_variable"]
        interp.append(f"- TFT variable selection, share by input cycle (t-3, t-2, t-1): {w[0]:.3f}, {w[1]:.3f}, {w[2]:.3f}; "
                      f"temperature inputs {v['temp']:.3f}, salinity inputs {v['sal']:.3f}")
    for k, v in ex.items():
        if k.endswith("_temporal_attention"):
            interp.append(f"- {k.replace('_temporal_attention', '')} mean attention over (t-3, t-2, t-1): "
                          + ", ".join(f"{x:.3f}" for x in v))
    if "lightgbm_gain_by_step" in ex:
        g = ex["lightgbm_gain_by_step"]
        interp.append(f"- LightGBM gain importance by input cycle (t-3, t-2, t-1): {g[0]:.3f}, {g[1]:.3f}, {g[2]:.3f}")
    if interp:
        L += ["## Built-in interpretability signals", "",
              "These are the models' own importance signals, recorded for the XAI comparison. They are not post-hoc attributions.", ""] + interp + [""]

    L += ["## Files", "", "- `metrics.json`: every number above, plus training histories",
          "- `metrics.csv`, `per_depth_temp_rmse.csv`, `per_depth_sal_rmse.csv`",
          "- `figures/`: fig_bench_model_comparison, fig_bench_depth_rmse, fig_bench_params_vs_rmse, "
          "fig_bench_tft_variable_selection, fig_bench_temporal_attention (PNG 300 dpi and PDF)",
          "- `predictions/`: test-set predictions per model (first seed); `models/`: trained weights (first seed)", ""]
    return "\n".join(L)


def write_combined(out_root: Path):
    """One record across every completed run in outputs/."""
    runs = sorted(p for p in out_root.glob("*/metrics.json"))
    if not runs:
        return None
    L = ["# FloatChat 10-model benchmark: combined results", "",
         "Each section links to the full record for that dataset and split (`<run>/RESULTS.md`).", ""]
    for f in runs:
        p = json.loads(f.read_text())
        R = p["models"]
        L += [f"## {p['tag']}", "",
              f"Test windows {p['n_windows']['test']:,} from {p['data']['floats']['test']} floats; device {p['environment']['device']}; "
              f"finished {p['finished']}. Full record: `{f.parent.name}/RESULTS.md`.", "",
              "| Rank (Temp RMSE) | Model | Temp RMSE (°C) | Sal RMSE (PSU) | Thermocline RMSE | Temp skill vs persistence | Inversions (%) |",
              "|---|---|---|---|---|---|---|"]
        order = sorted(R, key=lambda m: R[m]["temp_rmse"])
        for i, m in enumerate(order, 1):
            r = R[m]
            L.append(f"| {i} | {m} | {r['temp_rmse']:.4f} | {r['sal_rmse']:.4f} | {r['therm_rmse']:.4f} | "
                     f"{r['temp_skill_vs_persistence']*100:+.1f}% | {r['inversion_teos10_pct']:.2f} |")
        L += ["", f"Observed test profiles: {p['extras']['observed_inversion_teos10_pct']:.2f}% inversions.", ""]
    path = out_root / "BENCHMARK_10_MODELS_RESULTS.md"
    path.write_text("\n".join(L), encoding="utf-8")
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", choices=["canonical", "arabian2000"], default="canonical")
    ap.add_argument("--split", choices=["chronological", "unseen_floats", "both"], default="both",
                    help="only for --dataset arabian2000; 'both' runs the two splits one after the other")
    ap.add_argument("--allow-cpu", action="store_true",
                    help="run even when no CUDA GPU is visible (slow; meant for smoke tests)")
    ap.add_argument("--quick", action="store_true",
                    help="smoke test: 1 seed, 2 epochs, 50 trees, 50 bootstrap resamples")
    ap.add_argument("--csv", type=Path, default=BACKEND / "data" / "processed" / "argo_floats_canonical.csv")
    ap.add_argument("--proc-dir", type=Path,
                    default=Path(os.environ.get("FC_DATA_DIR", BACKEND / "experiments" / "arabian_sea_2000_2026" / "data")) / "processed")
    ap.add_argument("--out", type=Path, default=HERE / "outputs")
    ap.add_argument("--models", nargs="+", default=None, help="subset of model names (Persistence is always added)")
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--lgb-estimators", type=int, default=2000)
    ap.add_argument("--lgb-jobs", type=int, default=-1)
    ap.add_argument("--threads", type=int, default=None)
    args = ap.parse_args()
    if args.models and "Persistence" not in args.models:
        args.models = ["Persistence"] + args.models
    if args.quick:
        args.seeds, args.epochs, args.lgb_estimators, args.bootstrap = args.seeds[:1], 2, 50, 50
        args.out = args.out / "_quick"
    if not torch.cuda.is_available() and not args.allow_cpu:
        sys.exit("No CUDA GPU is visible to PyTorch, so nothing was run.\n"
                 "Check with:  python -c \"import torch; print(torch.cuda.is_available())\"\n"
                 "If it prints False, install the CUDA build of PyTorch, for example:\n"
                 "  pip install torch --index-url https://download.pytorch.org/whl/cu118\n"
                 "or add --allow-cpu to run on CPU anyway.")
    t0 = time.time()
    splits = ["chronological", "unseen_floats"] if args.split == "both" else [args.split]
    if args.dataset == "canonical":
        splits = [None]
    for sp in splits:
        args.split = sp
        run(args)
    path = write_combined(args.out)
    print(f"Combined record: {path}  (total {(time.time() - t0) / 60:.1f} min)")


if __name__ == "__main__":
    main()
