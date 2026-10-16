#!/usr/bin/env python
"""Condense the benchmark result files into the JSON served by GET /api/benchmark.

Inputs (written by the experiment scripts):
  experiments/benchmark_10_models/outputs/arabian2000_<split>/metrics.json
  experiments/benchmark_10_models/outputs/arabian2000_<split>/metrics_1_7.json
  experiments/benchmark_10_models/outputs/arabian2000_<split>/classification_metrics.json
  experiments/benchmark_10_models/outputs/mld_fix/mld_fix_results.json (+ mld_fix_seeds.csv)
  experiments/arabian_sea_2000_2026/outputs/tables/xai_comparison.json
  experiments/arabian_sea_2000_2026/data/processed/dataset_summary.json (optional)

Output:
  app/ml/artifacts/benchmark_results.json

Usage (from backend/):
  python scripts/build_benchmark_summary.py [--bench-out DIR] [--xai FILE]
"""
import argparse
import csv
import json
from datetime import date
from pathlib import Path

import numpy as np

BACKEND = Path(__file__).resolve().parents[1]
DEPTHS = [5, 20, 50, 75, 100, 150, 200, 250, 300, 400, 500, 600, 700, 800, 900, 1000]
PROPOSED = "FloatChat PI-BiLSTM"
SPLITS = {
    "chronological": {"label": "Future years (2019-2026)",
                      "description": "Trained on profiles up to 2016, validated on 2017-2018, tested on 2019-2026."},
    "unseen_floats": {"label": "Unseen floats",
                      "description": "Tested on floats never used in training or validation (all years)."},
}
MODELS = [
    # name, short label, family, plain-language description
    ("Persistence", "Persistence", "Simple rule", "Next profile = the last profile (assumes no change)."),
    ("Climatology (regional)", "Clim (regional)", "Simple rule",
     "Long-term average profile of the whole Arabian Sea for that calendar month."),
    ("Climatology (local)", "Clim (local)", "Simple rule",
     "Long-term average for the same grid cell and calendar month."),
    ("Ridge", "Ridge", "Statistics", "Linear regression on the 96 inputs (3 profiles x 32 values)."),
    ("LightGBM", "LightGBM", "Machine learning", "Gradient-boosted decision trees, one per output value."),
    ("GRU", "GRU", "Deep learning", "Recurrent network that reads the three profiles in order."),
    ("Seq2Seq LSTM + attention", "Seq2Seq+Attn", "Deep learning",
     "LSTM encoder-decoder that can focus on the most useful past profile."),
    ("TCN", "TCN", "Deep learning", "Temporal convolution network (dilated causal filters)."),
    ("DLinear", "DLinear", "Deep learning", "Trend + remainder decomposition with linear layers."),
    ("N-HiTS", "N-HiTS", "Deep learning", "Stacked MLP blocks working at different time resolutions."),
    ("TFT", "TFT", "Deep learning", "Temporal Fusion Transformer: attention plus input selection."),
    ("Bi-LSTM (no physics loss)", "Bi-LSTM (no physics)", "Deep learning",
     "Same network as FloatChat, trained on error only (ablation)."),
    (PROPOSED, "FloatChat PI-BiLSTM", "Deep learning + physics",
     "2-layer bidirectional LSTM with a TEOS-10 density-stability penalty (proposed)."),
]
COLORS = {
    "Persistence": "#999999", "Climatology (regional)": "#CC79A7", "Climatology (local)": "#882255",
    "Ridge": "#E69F00", "LightGBM": "#56B4E9", "GRU": "#8C6D31", "Seq2Seq LSTM + attention": "#D55E00",
    "TCN": "#7F3C8D", "DLinear": "#C9B200", "N-HiTS": "#11A579", "TFT": "#E73F74",
    "Bi-LSTM (no physics loss)": "#009E73", PROPOSED: "#0072B2",
}
MLD_VARIANTS = [
    ("baseline", "Current loss (FloatChat)"), ("tol_0.01", "Tolerance 0.01"), ("tol_0.03", "Tolerance 0.03"),
    ("weight_1", "Stability weight 1"), ("surface_match", "Surface term"),
    ("tol_0.01_surface", "Tolerance 0.01 + surface term"), ("tol_0.03_surface", "Tolerance 0.03 + surface term"),
    ("no_physics", "No physics loss"),
]


def r(x, d=4):
    if x is None:
        return None
    try:
        if np.isnan(x):
            return None
    except TypeError:
        pass
    return round(float(x), d)


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def split_block(out: Path, split: str) -> dict:
    d = out / f"arabian2000_{split}"
    M, P, C = load(d / "metrics.json"), load(d / "metrics_1_7.json"), load(d / "classification_metrics.json")
    rows, per_depth_T, per_depth_S, season, uq = {}, {}, {}, {}, {}
    for name, *_ in MODELS:
        m, e, c = M["models"][name], P["models"][name], C["models"][name]
        ph, der = e["physical"]["rate_pct"], e["derived"]
        dirT = (c.get("direction") or {}).get("T")
        st = P["seed_stats"].get(name, {})
        rows[name] = {
            "temp_rmse": r(m["temp_rmse"]), "temp_rmse_ci": [r(v) for v in m["temp_rmse_ci"]],
            "sal_rmse": r(m["sal_rmse"]), "temp_mae": r(m["temp_mae"]), "sal_mae": r(m["sal_mae"]),
            "therm_rmse": r(m["therm_rmse"]), "deep_rmse": r(m["deep_rmse"]),
            "temp_r2": r(m["temp_r2"]), "temp_bias": r(e["T"]["bias"]), "acc_T": r(e["T"]["acc"]),
            "p95_abs_error_T": r(e["T"]["p95_abs_error"]),
            "skill_vs_persistence": r(m.get("temp_skill_vs_persistence", 0.0)),
            "skill_vs_climatology": r(e["T"]["skill_vs_climatology"]),
            "inv_any_pct": r(ph["any"], 3), "inv_001_pct": r(ph["0.01"], 3), "inv_003_pct": r(ph["0.03"], 3),
            "mld_rmse": r(der["mld_m"]["rmse"], 2), "mld_bias": r(der["mld_m"]["bias"], 2),
            "d20_rmse": r(der["d20_m"]["rmse"], 2), "ohc_rmse": r(der["ohc_0_700_GJ_m2"]["rmse"]),
            "crps_point_T": r(e["crps_point"]["T"]),
            "direction_acc_T": r(dirT["accuracy"]) if dirT else None,
            "direction_f1_T": r(dirT["macro_f1"]) if dirT else None,
            "n_params": int(m.get("n_params", 0)),
            "fit_seconds": r(m.get("fit_seconds"), 1),
            "seed_temp_rmse_mean": r(m.get("temp_rmse_seed_mean")),
            "seed_temp_rmse_std": r(m.get("temp_rmse_seed_std"), 5),
            "seed_inv_003_mean": r(st["inversion_rate_0.03_pct"]["mean"], 3) if st else None,
            "seed_inv_003_std": r(st["inversion_rate_0.03_pct"]["std"], 3) if st else None,
            "seed_inv_any_mean": r(st["inversion_rate_any_pct"]["mean"], 3) if st else None,
            "seed_inv_any_std": r(st["inversion_rate_any_pct"]["std"], 3) if st else None,
            "seed_mld_rmse_mean": r(st["mld_m_rmse"]["mean"], 2) if st else None,
            "diff_vs_floatchat_ci": [r(v) for v in m["temp_rmse_diff_vs_floatchat_ci"]]
            if "temp_rmse_diff_vs_floatchat_ci" in m else None,
        }
        per_depth_T[name] = [r(v) for v in e["T"]["per_depth_rmse"]]
        per_depth_S[name] = [r(v) for v in e["S"]["per_depth_rmse"]]
        season[name] = {k: r(v) for k, v in e["T"]["season_rmse"].items()}
        if name in P["uq"]:
            u = P["uq"][name]["T"]
            uq[name] = {
                "coverage95_raw": r(u["coverage95_raw"] * 100, 2), "coverage95_cal": r(u["coverage95_cal"] * 100, 2),
                "coverage90_raw": r(u["coverage90_raw"] * 100, 2), "coverage90_cal": r(u["coverage90_cal"] * 100, 2),
                "scale_k": r(u["calibration_scale_k"], 3), "width95_cal": r(u["width95_cal"], 3),
                "crps_raw": r(u["crps_raw"]), "crps_cal": r(u["crps_cal"]),
                "spread_skill_ratio": r(u["spread_skill_ratio"], 3),
                "reliability_raw": [r(v * 100, 2) for v in u["reliability_raw"]],
                "reliability_cal": [r(v * 100, 2) for v in u["reliability_cal"]],
            }
    obs = P["observed"]
    return {
        **SPLITS[split],
        "n_windows": M["n_windows"], "n_floats": M["data"]["floats"],
        "rows": rows, "per_depth_temp_rmse": per_depth_T, "per_depth_sal_rmse": per_depth_S,
        "season_temp_rmse": season, "season_counts": obs.get("season_counts"),
        "uq": uq, "reliability_nominal": list(range(10, 100, 10)),
        "observed": {"inv_any_pct": r(obs["inversion_rate_pct"]["any"], 3),
                     "inv_001_pct": r(obs["inversion_rate_pct"]["0.01"], 3),
                     "inv_003_pct": r(obs["inversion_rate_pct"]["0.03"], 3),
                     "mld_mean_m": r(obs["mld_mean_m"], 1)},
        "seeds": P.get("seeds"),
    }


def mld_block(out: Path):
    path = out / "mld_fix" / "mld_fix_results.json"
    if not path.exists():
        return None
    X = load(path)
    seeds = {}
    csv_path = out / "mld_fix" / "mld_fix_seeds.csv"
    if csv_path.exists():
        with open(csv_path, newline="") as f:
            for row in csv.DictReader(f):
                if row["set"] == "test":
                    seeds.setdefault((row["split"], row["variant"]), []).append(float(row["mld_rmse"]))
    variants = []
    for vid, label in MLD_VARIANTS:
        v = {"id": vid, "label": label, **X["variants"].get(vid, {})}
        for split in ("chronological", "unseen_floats"):
            mm = X["means"][split]["test"].get(vid)
            if mm is None:
                continue
            sd = seeds.get((split, vid))
            v[split] = {"mld_rmse": r(mm["mld_rmse"], 2), "mld_rmse_std": r(np.std(sd), 2) if sd else None,
                        "mld_bias": r(mm["mld_bias"], 2), "temp_rmse": r(mm["temp_rmse"]),
                        "inv_003_pct": r(mm["inv_003_pct"], 3), "pct_mixed_5_20": r(mm["pct_mixed_5_20"], 2)}
        variants.append(v)
    return {
        "variants": variants,
        "observed_pct_mixed_5_20": {s: r(X["observed"][s]["test"]["pct_mixed_5_20"], 1) for s in X["observed"]},
        "selected_on_validation": {s: X["selected"][s]["variant"] for s in X["selected"]},
        "seeds": X.get("seeds"),
    }


def xai_block(path: Path):
    if not path.exists():
        return None
    X = load(path)
    t = X["targets"]["temp_100"]
    methods = {}
    for key, label in (("FloatChat IG", "Integrated Gradients"), ("LIME", "LIME"), ("SHAP", "KernelSHAP")):
        m = t[key]
        methods[label] = {
            "faithfulness": r(m["faithfulness_corr"], 3), "repeatability": r(m["repeatability_cos"], 3),
            "robustness": r(m["robustness_cos"], 3), "completeness_error": r(m["completeness_error_median"], 3),
            "runtime_ms": r(m["runtime_s"] * 1000, 1), "share_latest_cycle": r(m["share_t1"], 3),
            "temporal_weights": [r(v, 3) for v in m["temporal_weights"]],
            "deletion_auc": r(m["deletion_auc"], 3),
        }
    return {"target": t["label"], "n_windows": X["n_explained"], "split": X["split"], "methods": methods,
            "depth_matrix_similarity": {k: r(v, 3) for k, v in t["depth_matrix_similarity"].items()}}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bench-out", type=Path, default=BACKEND / "experiments" / "benchmark_10_models" / "outputs")
    ap.add_argument("--xai", type=Path,
                    default=BACKEND / "experiments" / "arabian_sea_2000_2026" / "outputs" / "tables" / "xai_comparison.json")
    ap.add_argument("--dataset-summary", type=Path,
                    default=BACKEND / "experiments" / "arabian_sea_2000_2026" / "data" / "processed" / "dataset_summary.json")
    ap.add_argument("--out", type=Path, default=BACKEND / "app" / "ml" / "artifacts" / "benchmark_results.json")
    args = ap.parse_args()

    splits = {s: split_block(args.bench_out, s) for s in SPLITS}
    ds = load(args.dataset_summary) if args.dataset_summary.exists() else {"n_profiles": 36164, "n_floats": 321}
    ch, un = splits["chronological"], splits["unseen_floats"]
    learned = [m for m, *_ in MODELS if m not in ("Persistence", "Climatology (regional)", "Climatology (local)", PROPOSED)]
    others_inv = [s["rows"][m]["inv_003_pct"] for s in (ch, un) for m in learned]
    fc = PROPOSED
    summary = {
        "n_baselines": len(MODELS) - 1, "n_profiles": ds["n_profiles"], "n_floats": ds["n_floats"],
        "n_test_chrono": ch["n_windows"]["test"], "n_test_unseen": un["n_windows"]["test"],
        "n_unseen_floats": un["n_floats"]["test"],
        "temp_rmse_chrono": ch["rows"][fc]["temp_rmse"], "temp_rmse_unseen": un["rows"][fc]["temp_rmse"],
        "sal_rmse_chrono": ch["rows"][fc]["sal_rmse"], "sal_rmse_unseen": un["rows"][fc]["sal_rmse"],
        "skill_chrono": ch["rows"][fc]["skill_vs_persistence"] * 100,
        "skill_unseen": un["rows"][fc]["skill_vs_persistence"] * 100,
        "inv_chrono": ch["rows"][fc]["inv_003_pct"], "inv_unseen": un["rows"][fc]["inv_003_pct"],
        "inv_others_min": min(others_inv), "inv_others_max": max(others_inv),
        "inv_obs_min": min(ch["observed"]["inv_003_pct"], un["observed"]["inv_003_pct"]),
        "inv_obs_max": max(ch["observed"]["inv_003_pct"], un["observed"]["inv_003_pct"]),
        "coverage95_raw_chrono": ch["uq"][fc]["coverage95_raw"], "coverage95_raw_unseen": un["uq"][fc]["coverage95_raw"],
        "coverage95_calibrated_chrono": ch["uq"][fc]["coverage95_cal"],
        "coverage95_calibrated_unseen": un["uq"][fc]["coverage95_cal"],
        "mld_rmse_chrono": ch["rows"][fc]["mld_rmse"], "mld_rmse_unseen": un["rows"][fc]["mld_rmse"],
        "mld_rmse_no_physics_chrono": ch["rows"]["Bi-LSTM (no physics loss)"]["mld_rmse"],
        "mld_rmse_no_physics_unseen": un["rows"]["Bi-LSTM (no physics loss)"]["mld_rmse"],
    }
    payload = {
        "generated": date.today().isoformat(),
        "proposed_model": PROPOSED,
        "depths_dbar": DEPTHS,
        "dataset": {"n_profiles": ds["n_profiles"], "n_floats": ds["n_floats"],
                    "date_range": [ds.get("date_min", "2001-12-18"), ds.get("date_max", "2026-10-11")],
                    "region": "Arabian Sea, 10-25N 50-75E", "levels": "16 levels, 5-1000 dbar",
                    "task": "Forecast the next temperature and salinity profile (about 10 days ahead) "
                            "from the three preceding profiles"},
        "models": [{"name": n, "short": s, "family": f, "description": d, "color": COLORS[n],
                    "proposed": n == PROPOSED} for n, s, f, d in MODELS],
        "splits": splits,
        "mld_fix": mld_block(args.bench_out),
        "xai": xai_block(args.xai),
        "summary": summary,
    }
    args.out.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    print(f"Wrote {args.out} ({args.out.stat().st_size / 1024:.0f} KB)")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
