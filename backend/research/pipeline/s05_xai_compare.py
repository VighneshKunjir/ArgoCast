#!/usr/bin/env python
"""Stage 5 - compare three explanation methods on the trained ArgoCast model.

Methods (all explain one scalar output, e.g. temperature at 100 dbar, with
respect to the 96 inputs = 3 cycles x (16 T + 16 S levels), in scaled units)
  LIME          local weighted linear surrogate (lime_tabular, regression).
                Coefficients are turned into contributions w_i (x_i - mu_i)/s_i
                so that all three methods give additive attributions.
  SHAP          KernelSHAP (model-agnostic Shapley estimate) with a k-means
                background of 25 training windows.
  ArgoCast IG  the project's method: Integrated Gradients through the
                SingleOutputModelWrapper, climatological (all-zero scaled)
                baseline, 64 path steps, plus temporal weights and the
                16x16 depth-attribution matrix.

Evaluation metrics
  Deletion AUC (higher = better)   remove the most important inputs first
                                   (set to climatology) and track how fast the
                                   output moves away from f(x); normalised by
                                   |f(x) - f(baseline)|; random order as reference.
  Insertion error AUC (lower = better)  start from climatology, insert the most
                                   important inputs first, track |f - f(x)|.
  Faithfulness correlation (higher)  Pearson r between the summed attribution
                                   of a random subset of 16 inputs and the output
                                   change when that subset is removed (Bhatt et al.).
  Robustness (higher)              cosine similarity of attributions for x and
                                   x + N(0, 0.05) noise.
  Repeatability (higher)           cosine similarity of two runs on the same x.
  Completeness error (lower)       |sum(a) - (f(x) - f(reference))| / |f(x) - f(reference)|.
  Runtime per explanation (s).
  Agreement between methods        Spearman rank correlation and top-10 overlap.
  Physical plausibility            share of attribution on the most recent
                                   cycle (t-1), on the target variable, and
                                   within +/- 1 level of the target depth.

Outputs: outputs/tables/xai_comparison.json|csv, outputs/figures/fig_xai_*.png|pdf
"""
from __future__ import annotations

import argparse
import time
import warnings

import joblib
import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr, spearmanr

from common import (MODEL_DIR, N_LEV, PALETTE, PROCESSED_DIR, STANDARD_DEPTHS,
                    TAB_DIR, savefig, set_seed, setup_matplotlib, write_json)
from app.ml.model import PhysicsInformedBiLSTM, SingleOutputModelWrapper

warnings.filterwarnings("ignore")
METHODS = ["LIME", "SHAP", "ArgoCast IG"]
FEAT_NAMES = [f"{c}_{v}{int(d)}" for c in ("t-3", "t-2", "t-1") for v in ("T", "S") for d in STANDARD_DEPTHS]
NF = 96


def scale_X(X, sc):
    out = np.empty_like(X, dtype=np.float32)
    for k in range(3):
        out[:, k, :N_LEV] = sc["temp_scaler"].transform(X[:, k, :N_LEV])
        out[:, k, N_LEV:] = sc["sal_scaler"].transform(X[:, k, N_LEV:])
    return out


class Target:
    def __init__(self, model, var, depth_idx, device):
        self.var, self.d, self.device = var, depth_idx, device
        self.wrapper = SingleOutputModelWrapper(model, target_variable=var, target_depth_idx=depth_idx).eval()
        self.label = f"{'Temperature' if var == 'temp' else 'Salinity'} at {int(STANDARD_DEPTHS[depth_idx])} dbar"
        self.key = f"{var}_{int(STANDARD_DEPTHS[depth_idx])}"
        self.n_calls = 0

    @torch.no_grad()
    def f(self, xflat):
        xflat = np.asarray(xflat, dtype=np.float32)
        self.n_calls += len(xflat)
        out = []
        for i in range(0, len(xflat), 8192):
            xb = torch.tensor(xflat[i:i + 8192].reshape(-1, 3, 32), device=self.device)
            out.append(self.wrapper(xb).squeeze(1).cpu().numpy())
        return np.concatenate(out).astype(np.float64)


# ------------------------------------------------------------- explainers
class IGExplainer:
    name = "ArgoCast IG"

    def __init__(self, target, steps):
        from captum.attr import IntegratedGradients
        self.t, self.steps = target, steps
        self.ig = IntegratedGradients(target.wrapper)
        self.deltas = []

    def explain(self, x):
        xt = torch.tensor(x.reshape(1, 3, 32), dtype=torch.float32, device=self.t.device, requires_grad=True)
        a, d = self.ig.attribute(xt, baselines=torch.zeros_like(xt), n_steps=self.steps,
                                 return_convergence_delta=True)
        self.deltas.append(float(abs(d.item())))
        return a.detach().cpu().numpy().reshape(-1)

    def reference(self, x):
        return float(self.t.f(np.zeros((1, NF)))[0])


class SHAPExplainer:
    name = "SHAP"

    def __init__(self, target, background, nsamples):
        import shap
        self.t, self.ns = target, nsamples
        self.bg = shap.kmeans(background, 25)
        self.ex = shap.KernelExplainer(target.f, self.bg, link="identity")

    def explain(self, x):
        v = self.ex.shap_values(x.reshape(1, -1), nsamples=self.ns, silent=True)
        return np.asarray(v).reshape(-1)

    def reference(self, x):
        return float(np.asarray(self.ex.expected_value).reshape(-1)[0])


class LIMEExplainer:
    name = "LIME"

    def __init__(self, target, background, nsamples, seed):
        from lime.lime_tabular import LimeTabularExplainer
        self.t, self.ns = target, nsamples
        self.ex = LimeTabularExplainer(background, mode="regression", feature_names=FEAT_NAMES,
                                       discretize_continuous=False, sample_around_instance=True,
                                       random_state=seed)
        self.mu, self.sd = self.ex.scaler.mean_, self.ex.scaler.scale_

    def explain(self, x):
        e = self.ex.explain_instance(x, self.t.f, num_features=NF, num_samples=self.ns)
        w = np.zeros(NF)
        # regression explanations are stored under label 1 (label 0 holds the negated copy)
        for i, c in e.local_exp.get(1, next(iter(e.local_exp.values()))):
            w[i] = c
        return w * (x - self.mu) / self.sd

    def reference(self, x):
        return float(self.t.f(self.mu.reshape(1, -1))[0])


# ---------------------------------------------------------------- metrics
STEPS = np.arange(0, NF + 1, 4)


def del_ins_curves(t: Target, x, order):
    fx = t.f(x[None])[0]
    f0 = t.f(np.zeros((1, NF)))[0]
    D = abs(fx - f0)
    dele, ins = [], []
    for k in STEPS:
        xd = x.copy(); xd[order[:k]] = 0.0
        xi = np.zeros(NF); xi[order[:k]] = x[order[:k]]
        dele.append(xd); ins.append(xi)
    fd, fi = t.f(np.array(dele)), t.f(np.array(ins))
    return np.abs(fd - fx) / D, np.abs(fi - fx) / D, D


def auc(c):
    c = np.asarray(c, dtype=float)
    return float(((c[1:] + c[:-1]) / 2).sum() / (len(c) - 1))  # trapezoid on [0, 1]


def faithfulness_corr(t: Target, x, a, rng, n_sub=64, size=16):
    fx = t.f(x[None])[0]
    subs = [rng.choice(NF, size, replace=False) for _ in range(n_sub)]
    xs = np.repeat(x[None], n_sub, 0)
    for j, s in enumerate(subs):
        xs[j, s] = 0.0
    drop = fx - t.f(xs)
    att = np.array([a[s].sum() for s in subs])
    if np.std(att) < 1e-12 or np.std(drop) < 1e-12:
        return np.nan
    return float(pearsonr(att, drop)[0])


def cos(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na > 0 and nb > 0 else np.nan


def plausibility(a, var, d):
    m = np.abs(a).reshape(3, 2, N_LEV)  # cycle, variable, depth
    tot = m.sum() + 1e-12
    vi = 0 if var == "temp" else 1
    lo, hi = max(0, d - 1), min(N_LEV, d + 2)
    return {"share_t1": float(m[2].sum() / tot), "share_target_var": float(m[:, vi].sum() / tot),
            "share_near_depth": float(m[:, vi, lo:hi].sum() / tot),
            "temporal": (m.sum(axis=(1, 2)) / tot).tolist()}


# ------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="chronological", choices=["chronological", "unseen_floats"])
    ap.add_argument("--seed-model", type=int, default=42, help="which trained ArgoCast seed to explain")
    ap.add_argument("--n", type=int, default=100, help="test windows to explain (season-stratified)")
    ap.add_argument("--stab-n", type=int, default=30, help="windows used for robustness/repeatability")
    ap.add_argument("--matrix-n", type=int, default=20, help="windows used for the 16x16 depth matrices")
    ap.add_argument("--targets", nargs="+", default=["temp:4", "sal:4"],
                    help="var:depth_index, e.g. temp:4 = temperature at 100 dbar")
    ap.add_argument("--ig-steps", type=int, default=64)
    ap.add_argument("--shap-nsamples", type=int, default=1000)
    ap.add_argument("--lime-samples", type=int, default=2000)
    ap.add_argument("--noise", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    set_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device("cpu")  # RNN backward is simplest and deterministic on CPU

    w = np.load(PROCESSED_DIR / "windows.npz", allow_pickle=False)
    sp = np.load(PROCESSED_DIR / f"split_{args.split}.npz")
    sc = joblib.load(MODEL_DIR / f"scalers_{args.split}.joblib")
    model = PhysicsInformedBiLSTM()
    model.load_state_dict(torch.load(MODEL_DIR / f"argocast_{args.split}_seed{args.seed_model}.pt", map_location="cpu"))
    model.eval()

    test_ix = sp["test"]
    seasons = w["season"][test_ix]
    pick = []
    per = max(1, args.n // len(np.unique(seasons)))
    for s in np.unique(seasons):
        cand = test_ix[seasons == s]
        pick += list(rng.choice(cand, min(per, len(cand)), replace=False))
    pick = np.array(sorted(pick))[: args.n]
    Xe = scale_X(w["X"][pick], sc).reshape(len(pick), -1).astype(np.float64)
    tr = sp["train"]
    bg = scale_X(w["X"][rng.choice(tr, min(2000, len(tr)), replace=False)], sc).reshape(-1, NF).astype(np.float64)
    print(f"Explaining {len(pick)} test windows ({args.split} split); background {len(bg)} training windows")

    report = {"split": args.split, "n_explained": int(len(pick)), "settings": vars(args), "targets": {}}
    store = {}
    for tspec in args.targets:
        var, d = tspec.split(":"); d = int(d)
        t = Target(model, var, d, device)
        print(f"\n--- target: {t.label}")
        exps = {"LIME": LIMEExplainer(t, bg, args.lime_samples, args.seed),
                "SHAP": SHAPExplainer(t, bg, args.shap_nsamples),
                "ArgoCast IG": IGExplainer(t, args.ig_steps)}
        A, R, times = {}, {}, {}
        for mname, ex in exps.items():
            t0 = time.time(); calls0 = t.n_calls
            A[mname] = np.array([ex.explain(x) for x in Xe])
            times[mname] = (time.time() - t0) / len(Xe)
            R[mname] = np.array([ex.reference(x) for x in Xe])
            calls = "IG path" if mname == "ArgoCast IG" else f"{(t.n_calls - calls0) / len(Xe):,.0f} model calls"
            print(f"  {mname:13s} {times[mname]*1000:8.1f} ms / explanation ({calls})")
        fx = t.f(Xe)
        res = {}
        curves = {}
        for mname in METHODS:
            a = A[mname]
            dels, inss, rdel, rins, faith, plaus = [], [], [], [], [], []
            for i, x in enumerate(Xe):
                order = np.argsort(-np.abs(a[i]))
                dc, ic, D = del_ins_curves(t, x, order)
                if D < 1e-3:
                    continue
                dels.append(dc); inss.append(ic)
                if mname == METHODS[0]:
                    rr = [del_ins_curves(t, x, rng.permutation(NF))[:2] for _ in range(5)]
                    rdel.append(np.mean([r[0] for r in rr], 0)); rins.append(np.mean([r[1] for r in rr], 0))
                faith.append(faithfulness_corr(t, x, a[i], rng))
                plaus.append(plausibility(a[i], var, d))
            denom = fx - R[mname]
            ok = np.abs(denom) > 1e-3
            comp = np.abs(a.sum(1)[ok] - denom[ok]) / np.abs(denom[ok])
            curves[mname] = (np.mean(dels, 0), np.mean(inss, 0))
            if mname == METHODS[0]:
                curves["Random"] = (np.mean(rdel, 0), np.mean(rins, 0))
            res[mname] = {
                "deletion_auc": float(np.mean([auc(c) for c in dels])),
                "insertion_error_auc": float(np.mean([auc(c) for c in inss])),
                "faithfulness_corr": float(np.nanmean(faith)),
                "faithfulness_corr_std": float(np.nanstd(faith)),
                "completeness_error_median": float(np.median(comp)) if len(comp) else np.nan,
                "runtime_s": float(times[mname]),
                "share_t1": float(np.mean([p["share_t1"] for p in plaus])),
                "share_target_var": float(np.mean([p["share_target_var"] for p in plaus])),
                "share_near_depth": float(np.mean([p["share_near_depth"] for p in plaus])),
                "temporal_weights": np.mean([p["temporal"] for p in plaus], 0).tolist(),
                "mean_abs_attr": (np.abs(a) / (np.abs(a).sum(1, keepdims=True) + 1e-12)).mean(0).tolist(),
            }
        res["Random"] = {"deletion_auc": auc(curves["Random"][0]), "insertion_error_auc": auc(curves["Random"][1])}

        # robustness and repeatability
        sub = Xe[: args.stab_n]
        for mname, ex in exps.items():
            rob, rep = [], []
            for i, x in enumerate(sub):
                a0 = A[mname][i]
                xn = x + rng.normal(0, args.noise, NF)
                rob.append(cos(a0, ex.explain(xn)))
                rep.append(cos(a0, ex.explain(x)))
            res[mname]["robustness_cos"] = float(np.nanmean(rob))
            res[mname]["repeatability_cos"] = float(np.nanmean(rep))
        res["ArgoCast IG"]["ig_convergence_delta_median"] = float(np.median(exps["ArgoCast IG"].deltas))
        res["ArgoCast IG"]["ig_convergence_delta_max"] = float(np.max(exps["ArgoCast IG"].deltas))

        # agreement
        agree = {}
        for i, m1 in enumerate(METHODS):
            for m2 in METHODS[i + 1:]:
                rho = [spearmanr(np.abs(A[m1][k]), np.abs(A[m2][k]))[0] for k in range(len(Xe))]
                top = [len(set(np.argsort(-np.abs(A[m1][k]))[:10]) & set(np.argsort(-np.abs(A[m2][k]))[:10])) / 10
                       for k in range(len(Xe))]
                agree[f"{m1} vs {m2}"] = {"spearman": float(np.nanmean(rho)), "top10_overlap": float(np.mean(top))}
        res["agreement"] = agree

        # 16x16 depth matrices (same-variable inputs -> outputs at every depth)
        mats = {}
        Xm = Xe[: args.matrix_n]
        for mname in METHODS:
            M = np.zeros((N_LEV, N_LEV))
            for jd in range(N_LEV):
                tj = Target(model, var, jd, device)
                ex = {"LIME": lambda: LIMEExplainer(tj, bg, args.lime_samples, args.seed),
                      "SHAP": lambda: SHAPExplainer(tj, bg, max(300, args.shap_nsamples // 2)),
                      "ArgoCast IG": lambda: IGExplainer(tj, args.ig_steps)}[mname]()
                vi = 0 if var == "temp" else 1
                acc = np.zeros(N_LEV)
                for x in Xm:
                    m = np.abs(ex.explain(x)).reshape(3, 2, N_LEV)[:, vi].sum(0)
                    acc += m / (m.sum() + 1e-12)
                M[jd] = acc / len(Xm)
            mats[mname] = M
            print(f"  depth matrix done: {mname}")
        res["depth_matrix_similarity"] = {
            f"{m1} vs {m2}": float(pearsonr(mats[m1].ravel(), mats[m2].ravel())[0])
            for i, m1 in enumerate(METHODS) for m2 in METHODS[i + 1:]}
        res["depth_matrix_diagonal_share"] = {
            m: float(np.mean([mats[m][j, max(0, j - 1):j + 2].sum() for j in range(N_LEV)])) for m in METHODS}
        report["targets"][t.key] = {"label": t.label, **res}
        store[t.key] = {"A": A, "curves": curves, "mats": mats, "label": t.label, "var": var, "d": d,
                        "x": Xe, "pick": pick}
        np.savez_compressed(TAB_DIR / f"xai_attributions_{t.key}.npz", windows=pick,
                            **{m.replace(" ", "_"): A[m] for m in METHODS},
                            **{f"matrix_{m.replace(' ', '_')}": mats[m] for m in METHODS})

    write_json(TAB_DIR / "xai_comparison.json", report)
    rows = []
    for k, r in report["targets"].items():
        for m in METHODS:
            rows.append({"target": r["label"], "method": m,
                         **{kk: vv for kk, vv in r[m].items() if np.isscalar(vv)}})
    pd.DataFrame(rows).to_csv(TAB_DIR / "xai_comparison.csv", index=False)
    figures(store, report, w, sc)
    print(f"\nXAI comparison saved: {TAB_DIR/'xai_comparison.json'}")


# ---------------------------------------------------------------- figures
def figures(store, report, w, sc):
    plt = setup_matplotlib()
    dl = [str(int(d)) for d in STANDARD_DEPTHS]
    for key, s in store.items():
        # mean |attribution| heatmaps
        fig, axes = plt.subplots(3, 1, figsize=(10, 6.2))
        vmax = max(np.max(report["targets"][key][m]["mean_abs_attr"]) for m in METHODS)
        for ax, m in zip(axes, METHODS):
            img = np.array(report["targets"][key][m]["mean_abs_attr"]).reshape(3, 32)
            im = ax.imshow(img, aspect="auto", cmap="viridis", vmin=0, vmax=vmax)
            ax.set_yticks([0, 1, 2]); ax.set_yticklabels(["t-3", "t-2", "t-1"])
            ax.set_xticks(range(32)); ax.set_xticklabels(dl + dl, fontsize=6, rotation=90)
            ax.axvline(15.5, color="white", lw=1.5)
            ax.set_title(f"{m}", fontsize=9, loc="left")
            ax.grid(False)
            fig.colorbar(im, ax=ax, fraction=0.02, pad=0.01, label="share of |attribution|")
        axes[-1].set_xlabel("Input level (dbar): temperature (left of white line) | salinity (right)")
        fig.suptitle(f"Where each method puts importance - target: {s['label']} "
                     f"(mean over {len(s['x'])} test windows)", fontsize=9)
        fig.tight_layout()
        savefig(fig, f"fig_xai_heatmap_{key}")

        # importance vs input depth (summed over cycles)
        fig, axes = plt.subplots(1, 2, figsize=(7, 4.2), sharey=True)
        for m in METHODS:
            img = np.array(report["targets"][key][m]["mean_abs_attr"]).reshape(3, 2, N_LEV).sum(0)
            axes[0].plot(img[0], STANDARD_DEPTHS, "-o", ms=3, color=PALETTE[m], label=m)
            axes[1].plot(img[1], STANDARD_DEPTHS, "-o", ms=3, color=PALETTE[m])
        for ax in axes:
            ax.axhline(STANDARD_DEPTHS[s["d"]], color="black", ls=":", lw=0.8)
        axes[0].set_ylim(1020, 0); axes[0].set_ylabel("Input pressure level (dbar)")
        axes[0].set_xlabel("Share of |attr| - temperature inputs"); axes[1].set_xlabel("Share of |attr| - salinity inputs")
        axes[0].legend(fontsize=7)
        fig.suptitle(f"Importance by input depth - target {s['label']} (dotted line)", fontsize=9)
        fig.tight_layout()
        savefig(fig, f"fig_xai_depth_profile_{key}")

        # deletion / insertion curves
        fig, axes = plt.subplots(1, 2, figsize=(8, 3.3))
        x = STEPS / NF * 100
        for m in METHODS + ["Random"]:
            dcur, icur = s["curves"][m]
            st = dict(color=PALETTE[m], ls="--" if m == "Random" else "-", lw=2 if m == "ArgoCast IG" else 1.2)
            axes[0].plot(x, dcur, label=f"{m} (AUC {auc(dcur):.3f})", **st)
            axes[1].plot(x, icur, label=f"{m} (AUC {auc(icur):.3f})", **st)
        axes[0].set_title("Deletion: higher and earlier is better", fontsize=9)
        axes[1].set_title("Insertion error: lower is better", fontsize=9)
        for ax in axes:
            ax.set_xlabel("Inputs removed / inserted (% of 96, most important first)")
            ax.legend(fontsize=7)
        axes[0].set_ylabel("|f(x_k) - f(x)| / |f(x) - f(climatology)|")
        fig.suptitle(f"Faithfulness curves - target {s['label']}", fontsize=9)
        fig.tight_layout()
        savefig(fig, f"fig_xai_deletion_insertion_{key}")

        # depth matrices
        fig, axes = plt.subplots(1, 3, figsize=(11, 3.8))
        for ax, m in zip(axes, METHODS):
            im = ax.imshow(s["mats"][m], cmap="magma", vmin=0, vmax=max(v.max() for v in s["mats"].values()))
            ax.set_xticks(range(N_LEV)); ax.set_xticklabels(dl, rotation=90, fontsize=6)
            ax.set_yticks(range(N_LEV)); ax.set_yticklabels(dl, fontsize=6)
            ax.set_xlabel("Input level (dbar)"); ax.set_title(m, fontsize=9); ax.grid(False)
        axes[0].set_ylabel("Forecast level (dbar)")
        fig.colorbar(im, ax=axes, fraction=0.015, label="share of row attribution")
        v = "temperature" if s["var"] == "temp" else "salinity"
        fig.suptitle(f"16x16 depth attribution matrix ({v} inputs -> {v} forecast)", fontsize=9)
        savefig(fig, f"fig_xai_depth_matrix_{s['var']}")

        # single example
        i = 0
        fig, axes = plt.subplots(1, 3, figsize=(10, 4.2), sharey=True)
        xin = w["X"][s["pick"][i]]
        axes[0].plot(xin[2, :N_LEV], STANDARD_DEPTHS, "k-o", ms=3, label="t-1 temperature")
        ax2 = axes[0].twiny(); ax2.plot(xin[2, N_LEV:], STANDARD_DEPTHS, "-o", color="#D55E00", ms=3)
        ax2.set_xlabel("Salinity (PSU)", color="#D55E00"); ax2.grid(False)
        axes[0].set_xlabel("Temperature (°C)"); axes[0].set_ylim(1020, 0); axes[0].set_ylabel("Pressure (dbar)")
        for m in METHODS:
            a = s["A"][m][i].reshape(3, 2, N_LEV)[2]
            axes[1].plot(a[0], STANDARD_DEPTHS, "-o", ms=3, color=PALETTE[m], label=m)
            axes[2].plot(a[1], STANDARD_DEPTHS, "-o", ms=3, color=PALETTE[m])
        for ax in axes[1:]:
            ax.axvline(0, color="black", lw=0.6)
            ax.axhline(STANDARD_DEPTHS[s["d"]], color="black", ls=":", lw=0.8)
        axes[1].set_xlabel("Attribution - t-1 temperature inputs"); axes[2].set_xlabel("Attribution - t-1 salinity inputs")
        axes[1].legend(fontsize=7)
        fig.suptitle(f"One test window (float {w['wmo'][s['pick'][i]]}, cycle {w['cycle'][s['pick'][i]]}) - "
                     f"signed attributions for {s['label']}", fontsize=9)
        fig.tight_layout()
        savefig(fig, f"fig_xai_example_{key}")

    # metric summary across targets
    keys = list(store)
    mets = [("deletion_auc", "Deletion AUC (↑)"), ("insertion_error_auc", "Insertion error AUC (↓)"),
            ("faithfulness_corr", "Faithfulness r (↑)"), ("robustness_cos", "Robustness cosine (↑)"),
            ("repeatability_cos", "Repeatability cosine (↑)"), ("completeness_error_median", "Completeness error (↓)"),
            ("runtime_s", "Runtime per explanation, s (↓, log)"), ("share_t1", "Share on t-1 cycle")]
    fig, axes = plt.subplots(2, 4, figsize=(12, 5.4))
    wbar = 0.8 / len(METHODS)
    for ax, (k, lab) in zip(axes.ravel(), mets):
        for j, m in enumerate(METHODS):
            vals = [report["targets"][t][m].get(k, np.nan) for t in keys]
            ax.bar(np.arange(len(keys)) + j * wbar - 0.4 + wbar / 2, vals, width=wbar, color=PALETTE[m], label=m)
        if k in ("deletion_auc", "insertion_error_auc"):
            ax.scatter(np.arange(len(keys)), [report["targets"][t]["Random"][k] for t in keys], marker="_", s=400,
                       color="black", label="Random order", zorder=3)
        if k == "runtime_s":
            ax.set_yscale("log")
        ax.set_xticks(range(len(keys))); ax.set_xticklabels([store[t]["label"].replace(" at ", "\n") for t in keys], fontsize=7)
        ax.set_title(lab, fontsize=8)
    axes[0, 0].legend(fontsize=6)
    fig.suptitle("Explanation quality - LIME vs SHAP vs ArgoCast IG", fontsize=10)
    fig.tight_layout()
    savefig(fig, "fig_xai_metric_summary")

    # temporal weights
    fig, ax = plt.subplots(figsize=(7, 3))
    labels = []
    for j, t in enumerate(keys):
        for i, m in enumerate(METHODS):
            tw = report["targets"][t][m]["temporal_weights"]
            x = j * (len(METHODS) + 1) + i
            left = 0
            for c, (val, col) in enumerate(zip(tw, ["#d9d9d9", "#969696", "#252525"])):
                ax.barh(x, val, left=left, color=col, label=["t-3", "t-2", "t-1"][c] if (j == 0 and i == 0) else None)
                left += val
            labels.append((x, f"{m} | {store[t]['label']}"))
    ax.set_yticks([p for p, _ in labels]); ax.set_yticklabels([l for _, l in labels], fontsize=7)
    ax.set_xlabel("Share of total |attribution| by input cycle"); ax.legend(fontsize=7, ncol=3, loc="lower right")
    ax.invert_yaxis()
    ax.set_title("Temporal attribution: how much each past cycle drives the forecast", fontsize=9)
    savefig(fig, "fig_xai_temporal_weights")

    # agreement
    fig, axes = plt.subplots(1, len(keys), figsize=(3.6 * len(keys), 3.2), squeeze=False)
    for ax, t in zip(axes[0], keys):
        M = np.eye(3)
        for i, m1 in enumerate(METHODS):
            for j, m2 in enumerate(METHODS):
                if i < j:
                    M[i, j] = M[j, i] = report["targets"][t]["agreement"][f"{m1} vs {m2}"]["spearman"]
        im = ax.imshow(M, cmap="Blues", vmin=0, vmax=1)
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=8,
                        color="white" if M[i, j] > 0.6 else "black")
        ax.set_xticks(range(3)); ax.set_xticklabels(METHODS, fontsize=7)
        ax.set_yticks(range(3)); ax.set_yticklabels(METHODS, fontsize=7)
        ax.set_title(f"Rank agreement - {store[t]['label']}", fontsize=8); ax.grid(False)
    fig.tight_layout()
    savefig(fig, "fig_xai_agreement")


if __name__ == "__main__":
    main()
