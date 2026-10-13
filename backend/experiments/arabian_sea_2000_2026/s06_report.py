#!/usr/bin/env python
"""Stage 6 - write the results record (Markdown) from the stage 2-5 outputs.

Every number in the report is read from the JSON files produced by the
pipeline; nothing is typed in by hand except the previous-run reference
values, which are copied from the project documentation and labelled as such.

Output: outputs/RESULTS_ArabianSea_2000_2026.md  (figures linked from outputs/figures/)
"""
from __future__ import annotations

import time

import numpy as np

from common import (MAX_LAT, MAX_LON, MIN_LAT, MIN_LON, OUT_DIR, PROCESSED_DIR, STANDARD_DEPTHS,
                    TAB_DIR, read_json, DATA_DIR)

MODEL_ORDER = ["Persistence", "Climatology", "Ridge", "Gradient Boosting",
               "Bi-LSTM (MSE only)", "FloatChat Bi-LSTM"]
METHODS = ["LIME", "SHAP", "FloatChat IG"]
PREVIOUS = {  # from "FloatChat Project Documentation" / project description (109-float run)
    "FloatChat Bi-LSTM": (0.512, 0.107, 0.803, "0.00"),
    "Gradient Boosting": (0.507, 0.075, 0.805, "23.9"),
    "Persistence": (0.544, 0.110, 0.859, "21.0"),
}


def f(x, n=3):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "-"
    return f"{x:.{n}f}"


def ci(c, n=3):
    return f"[{c[0]:.{n}f}, {c[1]:.{n}f}]"


def table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def fig(name, caption):
    p = OUT_DIR / "figures" / f"{name}.png"
    return f"![{caption}](figures/{name}.png)\n\n*{caption}*\n" if p.exists() else ""


def main():
    ds = read_json(PROCESSED_DIR / "dataset_summary.json")
    sp = read_json(PROCESSED_DIR / "split_summary.json")
    dl = read_json(DATA_DIR / "download_log.json") if (DATA_DIR / "download_log.json").exists() else {}
    run = read_json(TAB_DIR / "run_info.json") if (TAB_DIR / "run_info.json").exists() else {}
    xai = read_json(TAB_DIR / "xai_comparison.json") if (TAB_DIR / "xai_comparison.json").exists() else None
    M = {s: read_json(TAB_DIR / f"metrics_{s}.json") for s in ("chronological", "unseen_floats")
         if (TAB_DIR / f"metrics_{s}.json").exists()}

    L = []
    L.append("# FloatChat - Extended Evaluation on All Arabian Sea Argo Floats (2000-2026)\n")
    L.append(f"Record generated {time.strftime('%Y-%m-%d %H:%M')} by `experiments/arabian_sea_2000_2026/s06_report.py`. "
             "All values below are read directly from the pipeline output files in `outputs/tables/`.\n")

    # ---------------------------------------------------------- 1 data
    L.append("## 1. Data\n")
    L.append(table(["Item", "Value"], [
        ["Region", f"{MIN_LAT:g}-{MAX_LAT:g} N, {MIN_LON:g}-{MAX_LON:g} E (Arabian Sea)"],
        ["Period", f"{dl.get('years', ['2000', '2026'])[0]}-{dl.get('years', ['2000', '2026'])[1]}"],
        ["Source", "Argo GDAC (IFREMER, US GODAE fallback), per-float `*_prof.nc` files"],
        ["Floats with >=1 profile in box (index)", dl.get("n_floats_in_index", "-")],
        ["Profiles in box (index, all modes)", f"{dl.get('n_profiles_in_index', 0):,}"],
        ["Float files that failed to download", dl.get("n_failed", "-")],
        ["Clean profiles kept", f"{ds['n_profiles']:,}"],
        ["Floats with clean profiles", ds["n_floats"]],
        ["Date range of clean profiles", f"{ds['date_min']} to {ds['date_max']}"],
        ["Data modes kept", f"{ds['modes_kept']} -> " + ", ".join(f"{k}: {v:,}" for k, v in ds["mode_counts"].items())],
        ["Observed profiles statically stable (TEOS-10)", f"{ds['stable_fraction_teos10']*100:.1f}%"],
        ["Forecast windows (3 -> 1)", f"{sp['n_windows']:,}"],
        ["Windows rejected by time-gap rule", f"{sp['gap_rejected_windows']:,} (steps outside {sp['gap_rule_days'][0]:g}-{sp['gap_rule_days'][1]:g} days)"],
    ]))
    cc = ds["cleaning_counts"]
    L.append("\nCleaning log (profiles removed at each rule):\n")
    L.append(table(["Rule", "Profiles"], [[k.replace("_", " "), f"{v:,}"] for k, v in cc.items() if k not in ("seen",)]
                   + [["profiles examined", f"{cc.get('seen', 0):,}"]]))
    L.append("")
    L.append(fig("fig_data_and_splits", "Figure 1. Target-profile positions and yearly window counts for both splits."))

    # ---------------------------------------------------------- 2 splits
    L.append("## 2. Train / validation / test splits\n")
    L.append("Two leakage-safe splits are used. Scalers are fitted on the training part only.\n")
    L.append("- **Chronological (primary).** Train on the past, test on the future. Windows are assigned by the "
             f"date of the target profile: validation from {sp['chronological']['t1_val_start']}, test from "
             f"{sp['chronological']['t2_test_start']}. Boundaries sit on 1 January near the 70 % / 85 % points of "
             f"the timeline. Windows whose inputs straddle a boundary are purged "
             f"({sp['chronological']['purged_boundary_windows']:,} windows), so no test input was a training target.\n"
             "- **Unseen floats (generalisation).** 70 / 15 / 15 of floats, stratified by each float's median year, "
             f"seed {sp['unseen_floats']['seed']}. Test floats are never seen during training.\n")
    rows = []
    for s in ("chronological", "unseen_floats"):
        for k in ("train", "val", "test"):
            p = sp[s][k]
            rows.append([s.replace("_", " "), k, f"{p['windows']:,}", p["floats"], p.get("date_min", "-"),
                         p.get("date_max", "-"), f"{p.get('share_of_windows', 0)*100:.1f}%"])
    L.append(table(["Split", "Part", "Windows", "Floats", "First target", "Last target", "Share"], rows))
    L.append("")

    # ---------------------------------------------------------- 3 accuracy
    sec = 3
    for s, m in M.items():
        R = m["models"]
        L.append(f"## {sec}. Forecast accuracy - {s.replace('_', ' ')} split\n")
        L.append(f"Test set: {m['n_test_windows']:,} windows from {m['n_test_floats']} floats. "
                 "RMSE in physical units; 95 % CI from a float-level block bootstrap "
                 f"({m['settings']['bootstrap']} resamples). Neural models: seed {m['settings']['seeds'][0]} "
                 "for the CI columns, mean +/- std over seeds "
                 f"{', '.join(map(str, m['settings']['seeds']))} in the last columns.\n")
        rows = []
        for name in MODEL_ORDER:
            if name not in R:
                continue
            r = R[name]
            seedc = (f"{r['temp_rmse_seed_mean']:.3f} +/- {r['temp_rmse_seed_std']:.3f}" if "temp_rmse_seed_mean" in r else "-")
            seeds = (f"{r['sal_rmse_seed_mean']:.3f} +/- {r['sal_rmse_seed_std']:.3f}" if "sal_rmse_seed_mean" in r else "-")
            rows.append([f"**{name}**" if "FloatChat" in name else name,
                         f(r["temp_rmse"]), ci(r["temp_rmse_ci"]), f(r["temp_mae"]), f(r["therm_rmse"]),
                         f(r["deep_rmse"]), f(r["temp_r2"]), f(r["sal_rmse"]), ci(r["sal_rmse_ci"]), f(r["sal_r2"]),
                         f(r["violation_teos10_pct"], 2), f(r["violation_poly_pct"], 2), seedc, seeds])
        obs = R["Observed (reference)"]
        rows.append(["Observed profiles", "-", "-", "-", "-", "-", "-", "-", "-", "-",
                     f(obs["violation_teos10_pct"], 2), f(obs["violation_poly_pct"], 2), "-", "-"])
        L.append(table(["Model", "T RMSE (C)", "T 95% CI", "T MAE", "Thermocline RMSE 50-150", "Deep RMSE 500-1000",
                        "T R2", "S RMSE (PSU)", "S 95% CI", "S R2", "Inversions TEOS-10 (%)", "Inversions poly (%)",
                        "T RMSE seeds", "S RMSE seeds"], rows))
        L.append("")
        rows = []
        for name in MODEL_ORDER:
            if name in R and name != "Persistence":
                r = R[name]
                rows.append([name, f"{r['temp_skill_vs_persistence']*100:.1f}%",
                             f"[{r['temp_skill_vs_persistence_ci'][0]*100:.1f}%, {r['temp_skill_vs_persistence_ci'][1]*100:.1f}%]",
                             f"{r['sal_skill_vs_persistence']*100:.1f}%", f"{r['p_not_better_than_persistence']:.3f}"])
        L.append("Skill against persistence (1 - RMSE_model / RMSE_persistence; positive = better):\n")
        L.append(table(["Model", "T skill", "T skill 95% CI", "S skill", "Bootstrap share with skill <= 0"], rows))
        L.append("")
        fc, pe = R.get("FloatChat Bi-LSTM"), R["Persistence"]
        if fc:
            best_t = min((n for n in MODEL_ORDER if n in R), key=lambda n: R[n]["temp_rmse"])
            best_s = min((n for n in MODEL_ORDER if n in R), key=lambda n: R[n]["sal_rmse"])
            L.append(f"Summary: FloatChat temperature RMSE is {fc['temp_rmse']:.3f} C against {pe['temp_rmse']:.3f} C "
                     f"for persistence (skill {fc['temp_skill_vs_persistence']*100:.1f} %, 95 % CI "
                     f"{fc['temp_skill_vs_persistence_ci'][0]*100:.1f} to {fc['temp_skill_vs_persistence_ci'][1]*100:.1f} %). "
                     f"Lowest temperature RMSE on this split: {best_t}; lowest salinity RMSE: {best_s}. "
                     f"FloatChat produced density inversions in {fc['violation_teos10_pct']:.2f} % of forecast profiles "
                     f"(TEOS-10 check), compared with {obs['violation_teos10_pct']:.2f} % in the observed profiles. "
                     f"Removing the physics terms (Bi-LSTM MSE only) gives "
                     f"{R['Bi-LSTM (MSE only)']['violation_teos10_pct']:.2f} % inversions and "
                     f"{R['Bi-LSTM (MSE only)']['temp_rmse']:.3f} C RMSE.\n" if "Bi-LSTM (MSE only)" in R else "\n")
        # per depth table
        rows = []
        cols = [n for n in MODEL_ORDER if n in R]
        for j, d in enumerate(STANDARD_DEPTHS):
            rows.append([int(d)] + [f"{R[n]['per_depth_temp_rmse'][j]:.3f}" for n in cols])
        L.append("Temperature RMSE by depth (C):\n")
        L.append(table(["dbar"] + cols, rows))
        L.append("")
        rows = []
        for j, d in enumerate(STANDARD_DEPTHS):
            rows.append([int(d)] + [f"{R[n]['per_depth_sal_rmse'][j]:.4f}" for n in cols])
        L.append("Salinity RMSE by depth (PSU):\n")
        L.append(table(["dbar"] + cols, rows))
        L.append("")
        seas = list(R["Persistence"]["by_season"])
        rows = [[n] + [f"{R[n]['by_season'][x]['temp_rmse']:.3f}" if x in R[n]["by_season"] else "-" for x in seas] for n in cols]
        L.append("Temperature RMSE by monsoon season (C):\n")
        L.append(table(["Model"] + [f"{x} (n={R['Persistence']['by_season'][x]['n']:,})" for x in seas], rows))
        L.append("")
        yrs = list(R["Persistence"]["by_year"])
        if len(yrs) > 1:
            rows = [[n] + [f"{R[n]['by_year'][y]['temp_rmse']:.3f}" for y in yrs] for n in cols]
            L.append("Temperature RMSE by test year (C):\n")
            L.append(table(["Model"] + [f"{y} (n={R['Persistence']['by_year'][y]['n']:,})" for y in yrs], rows))
            L.append("")
        uq = m["uq_mc_dropout"]
        L.append(f"Uncertainty (FloatChat, {m['settings']['mc_passes']}-pass MC dropout):\n")
        L.append(table(["Variable", "90% coverage (Gaussian)", "95% coverage (Gaussian)", "90% coverage (quantile)",
                        "95% coverage (quantile)", "Mean 95% width"],
                       [[v, f"{uq[v]['gauss_90_coverage']*100:.1f}%", f"{uq[v]['gauss_95_coverage']*100:.1f}%",
                         f"{uq[v]['quantile_90_coverage']*100:.1f}%", f"{uq[v]['quantile_95_coverage']*100:.1f}%",
                         f"{uq[v]['mean_interval_width_95']:.3f}"] for v in ("T", "S")]))
        L.append("\nA well-calibrated model covers close to 90 % and 95 %. Values well below mean the intervals are too narrow.\n")
        rows = [[n, f"{R[n].get('fit_seconds') or 0:.0f}"] for n in cols if R[n].get("fit_seconds")]
        if rows:
            L.append("Training time on this machine (seconds, first seed):\n")
            L.append(table(["Model", "Seconds"], rows)); L.append("")
        k = sec
        L.append(fig(f"fig_model_comparison_{s}", f"Figure {k}.1. Test RMSE with 95 % CI and inversion rate - {s.replace('_', ' ')} split."))
        L.append(fig(f"fig_depth_rmse_{s}", f"Figure {k}.2. RMSE and skill by depth - {s.replace('_', ' ')} split."))
        L.append(fig(f"fig_seasonal_rmse_{s}", f"Figure {k}.3. RMSE by monsoon season - {s.replace('_', ' ')} split."))
        L.append(fig(f"fig_uq_calibration_{s}", f"Figure {k}.4. MC-dropout calibration - {s.replace('_', ' ')} split."))
        L.append(fig(f"fig_scatter_100dbar_{s}", f"Figure {k}.5. Forecast vs observed temperature at 100 dbar."))
        L.append(fig(f"fig_example_forecast_{s}", f"Figure {k}.6. Example forecast (median-error window)."))
        L.append(fig(f"fig_training_curves_{s}", f"Figure {k}.7. Validation curves of the neural models."))
        sec += 1

    # ---------------------------------------------------------- previous run
    L.append(f"## {sec}. Comparison with the previous FloatChat run\n")
    L.append("Previous values are copied from the project documentation (109 floats, 18,192 profiles, 916 test "
             "windows from 5 held-out floats, including 5 sub-polar floats in training). The data, split and "
             "test set differ, so the comparison shows the change in scale and setting, not a paired test.\n")
    rows = []
    for n, (t, sl, th, v) in PREVIOUS.items():
        new = {s: M[s]["models"].get(n) for s in M}
        rows.append([n, f"{t:.3f}", f"{sl:.3f}", f"{th:.3f}", v]
                    + [f"{new[s]['temp_rmse']:.3f} / {new[s]['sal_rmse']:.3f} / {new[s]['therm_rmse']:.3f} / "
                       f"{new[s]['violation_poly_pct']:.2f}" if new[s] else "-" for s in M])
    L.append(table(["Model", "Prev T RMSE", "Prev S RMSE", "Prev thermocline", "Prev inversions %"]
                   + [f"New {s.replace('_', ' ')} (T / S / therm / inv-poly %)" for s in M], rows))
    L.append("")
    sec += 1

    # ---------------------------------------------------------- XAI
    if xai:
        L.append(f"## {sec}. Explainability: LIME vs SHAP vs FloatChat IG\n")
        st = xai["settings"]
        L.append(f"Model explained: FloatChat Bi-LSTM, {xai['split'].replace('_', ' ')} split, seed {st['seed_model']}. "
                 f"{xai['n_explained']} test windows (stratified by season). LIME: {st['lime_samples']} perturbations; "
                 f"KernelSHAP: {st['shap_nsamples']} coalitions, 25-centre k-means background; IG: {st['ig_steps']} steps, "
                 "climatological baseline. Robustness noise: N(0, "
                 f"{st['noise']}) in scaled units on {st['stab_n']} windows. Depth matrices from {st['matrix_n']} windows.\n")
        for key, r in xai["targets"].items():
            L.append(f"### Target: {r['label']}\n")
            rows = []
            for mth in METHODS:
                q = r[mth]
                rows.append([f"**{mth}**" if "IG" in mth else mth, f(q["deletion_auc"]), f(q["insertion_error_auc"]),
                             f"{q['faithfulness_corr']:.3f} +/- {q['faithfulness_corr_std']:.3f}",
                             f(q["robustness_cos"]), f(q["repeatability_cos"]), f(q["completeness_error_median"]),
                             f"{q['runtime_s']*1000:.1f}", f"{q['share_t1']*100:.1f}%",
                             f"{q['share_target_var']*100:.1f}%", f"{q['share_near_depth']*100:.1f}%"])
            rows.append(["Random order", f(r["Random"]["deletion_auc"]), f(r["Random"]["insertion_error_auc"]),
                         "-", "-", "-", "-", "-", "-", "-", "-"])
            L.append(table(["Method", "Deletion AUC (higher better)", "Insertion error AUC (lower better)",
                            "Faithfulness r", "Robustness cos", "Repeatability cos", "Completeness error (median)",
                            "ms / explanation", "Share on t-1", "Share on target variable", "Share within +/-1 level"], rows))
            L.append("")
            ig = r["FloatChat IG"]
            L.append(f"IG convergence delta: median {ig['ig_convergence_delta_median']:.2e}, max {ig['ig_convergence_delta_max']:.2e}.\n")
            rows = [[k, f"{v['spearman']:.3f}", f"{v['top10_overlap']*100:.0f}%"] for k, v in r["agreement"].items()]
            L.append(table(["Pair", "Spearman rank correlation", "Top-10 overlap"], rows))
            L.append("")
            rows = [[k, f"{v:.3f}"] for k, v in r["depth_matrix_similarity"].items()]
            L.append("Similarity of the 16x16 depth matrices (Pearson r):\n")
            L.append(table(["Pair", "r"], rows))
            L.append("")
            tw = {m_: r[m_]["temporal_weights"] for m_ in METHODS}
            L.append(table(["Method", "t-3", "t-2", "t-1"], [[m_, *(f"{x:.3f}" for x in tw[m_])] for m_ in METHODS]))
            L.append("")
            best_del = max(METHODS, key=lambda m_: r[m_]["deletion_auc"])
            best_faith = max(METHODS, key=lambda m_: r[m_]["faithfulness_corr"])
            fastest = min(METHODS, key=lambda m_: r[m_]["runtime_s"])
            L.append(f"Summary: highest deletion AUC - {best_del}; highest faithfulness correlation - {best_faith}; "
                     f"fastest - {fastest} ({r[fastest]['runtime_s']*1000:.1f} ms vs "
                     f"{max(r[m_]['runtime_s'] for m_ in METHODS)*1000:.0f} ms for the slowest).\n")
            L.append(fig(f"fig_xai_heatmap_{key}", f"Mean share of |attribution| per input (cycle x level) - {r['label']}."))
            L.append(fig(f"fig_xai_depth_profile_{key}", f"Importance by input depth - {r['label']}."))
            L.append(fig(f"fig_xai_deletion_insertion_{key}", f"Deletion and insertion curves - {r['label']}."))
            L.append(fig(f"fig_xai_example_{key}", f"Signed attributions for one test window - {r['label']}."))
            L.append(fig(f"fig_xai_depth_matrix_{key.split('_')[0]}", "16x16 depth attribution matrices for the three methods."))
        L.append(fig("fig_xai_metric_summary", "Explanation-quality metrics side by side."))
        L.append(fig("fig_xai_temporal_weights", "Temporal attribution by method and target."))
        L.append(fig("fig_xai_agreement", "Rank agreement between methods."))
        sec += 1

    # ---------------------------------------------------------- reproducibility
    L.append(f"## {sec}. Reproducibility\n")
    env = run.get("environment", {})
    L.append(table(["Item", "Value"], [[k, v] for k, v in env.items()]
                   + [["Total stage-4 runtime", f"{run.get('total_seconds', 0)/60:.1f} min"],
                      ["Command", "`python run_all.py` (see README.md in this folder)"]]))
    L.append("\nFiles: `outputs/tables/` (all metrics as JSON/CSV), `outputs/predictions/` (test predictions), "
             "`outputs/models/` (weights and scalers), `outputs/figures/` (PNG 300 dpi and PDF).\n")

    out = OUT_DIR / "RESULTS_ArabianSea_2000_2026.md"
    out.write_text("\n".join(L), encoding="utf-8")
    print(f"Report written: {out}")


if __name__ == "__main__":
    main()
