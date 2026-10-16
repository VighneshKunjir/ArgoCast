# FloatChat Extended Evaluation - Experiment Protocol (Arabian Sea, 2000-2026)

Code: `backend/research/pipeline/` (run with `python run_all.py`). Results tables are written to `outputs/tables/`.

## 1. Objective

The goal is to re-measure FloatChat forecast accuracy and physical consistency on the complete Arabian Sea Argo record. The previous run used 109 floats, 18,192 profiles, 5 test floats and 916 test windows. This run uses all floats in the box, under two leakage-safe splits. It also compares the FloatChat explanation method (Integrated Gradients) against LIME and SHAP using quantitative faithfulness, stability, cost and plausibility metrics.

## 2. Data

| Item | Setting |
|---|---|
| Region | 10-25 N, 50-75 E |
| Period | 1 Jan 2000 - 31 Dec 2026 (target profiles) |
| Float selection | Every float with at least one profile in the box and period in the GDAC profile index (`ar_index_global_prof.txt.gz`), any data mode |
| Files | One multi-profile file per float (`dac/<dac>/<WMO>/<WMO>_prof.nc`); IFREMER mirror, US GODAE fallback |
| Data modes | D (delayed) and A (adjusted real-time), using `*_ADJUSTED` variables; R excluded. Option `--modes D` reproduces the earlier delayed-only setting |
| Profile type | Ascending only (descending duplicates removed) |
| QC | Position and date QC 1/2; level QC 1/2 on PRES, TEMP, PSAL; fill values masked |
| Physical bounds | T in [-2, 35] C, S in [30, 42] PSU per level and after interpolation |
| Depth coverage | At least 10 good levels; shallowest good level <= 10 dbar, deepest >= 950 dbar |
| Interpolation | PCHIP to 16 standard levels (5 ... 1000 dbar). A level at most 50 dbar beyond the measured range takes the nearest measured value; otherwise the profile is dropped (no extrapolation) |
| Duplicates | One profile per (WMO, cycle), delayed mode preferred |
| Stability flag | TEOS-10 sigma0 with gsw at the true position; kept as a label, never used to remove profiles |

Every rule logs how many profiles it removed (`data/processed/dataset_summary.json`).

### Differences from the earlier pipeline (stated so results are read correctly)

- **Download unit.** The earlier pipeline downloaded single-cycle `D*.nc` files. This one uses per-float `_prof.nc` files, which give the same profiles in far fewer downloads.
- **Data modes.** Mode A is now included so that 2023-2026 is represented; the earlier pipeline used D only. Delayed-mode data lag by 1-2 years, which is why A matters for the recent period.
- **Gap filling.** The earlier pipeline filled levels outside the measured range with the boundary value at any distance. Here filling is allowed only within 50 dbar.
- **Stability.** The TEOS-10 stability check uses each profile's own latitude and longitude. The earlier pipeline used a fixed 20 N, 65 E.
- **Region.** No sub-polar floats are included; the data are Arabian Sea only.

## 3. Forecast task and windows

- **Input.** 3 consecutive profiles (t-3, t-2, t-1) of one float. Each profile has 16 temperature and 16 salinity levels (96 inputs).
- **Output.** The next profile (16 T + 16 S).
- **Time-gap rule.** Each step between consecutive profiles must be 3-20 days. This keeps every window a true ~10-day-ahead forecast; windows that jump over data gaps are rejected and counted.

## 4. Splits

Scalers (per-depth standardisation) are fitted on the training part only.

**Split A - Chronological (primary, operational).**

- Windows are ordered by the date of the target profile.
- Train is everything before T1, validation is T1 to T2, and test is everything after T2.
- T1 and T2 are 1 January boundaries placed nearest to the 70 % and 85 % points of the window timeline. They can be fixed by hand with `--t1/--t2`.
- **Purge rule.** A validation or test window must have all of its inputs on or after its own boundary. Windows that straddle a boundary are dropped, so no test input was ever a training target.
- This answers the question: trained on the past, how well does the model forecast the future?

**Split B - Unseen floats (generalisation).**

- 70 / 15 / 15 % of floats go to train / validation / test.
- The assignment is stratified by each float's median year, so all periods appear in every part. Seed 42.
- Test floats are never seen in training.
- This answers the question: how well does the model forecast a float it has never seen?

## 5. Models

| Model | Description |
|---|---|
| Persistence | Next profile = t-1 profile |
| Climatology | Monthly mean training profile for the target month |
| Ridge | Linear regression on 96 scaled inputs, alpha = 1 |
| Gradient Boosting | HistGradientBoosting, one regressor per output (32), 200 iterations |
| Bi-LSTM (MSE only) | FloatChat architecture; loss = MSE(T) + 2.5 MSE(S). Ablation without physics terms |
| FloatChat Bi-LSTM | 2-layer Bi-LSTM, 128 units per direction, dropout 0.2, dual heads. Full loss with lambda_sal = 2.5, lambda_phys = 10, lambda_therm = 1.5 |

Neural training:

- AdamW, learning rate 1e-3, weight decay 1e-4.
- Cosine schedule over 60 epochs.
- Batch 64.
- Gradient clipping at norm 5.
- Early stopping on validation temperature RMSE, patience 10.
- Seeds 42, 43, 44.

## 6. Accuracy metrics

All metrics are computed on the test set in physical units:

- Temperature RMSE, MAE and R2.
- Thermocline RMSE (50-150 dbar).
- Deep RMSE (500-1000 dbar).
- Salinity RMSE, MAE and R2.
- Per-depth RMSE.
- RMSE per monsoon season: DJF, MAM, JJAS, ON.
- RMSE per test year.

Physical consistency:

- Percentage of forecast profiles with any density inversion, using the exact TEOS-10 sigma0 (gsw).
- The same percentage using the polynomial sigma from the FloatChat loss, for continuity with earlier numbers.
- The observed test profiles are reported as a reference row.

Statistics:

- 95 % CIs from a float-level block bootstrap (1000 resamples; floats are resampled because windows of one float are correlated).
- Skill against persistence, with its CI and the bootstrap share with skill <= 0.
- Mean +/- std over the three seeds.
- CIs and breakdowns use seed 42, fixed in advance and not chosen as the best seed.

Uncertainty:

- 50-pass MC dropout.
- Empirical coverage of the nominal 90 % and 95 % intervals (Gaussian and quantile).
- Reliability curve and per-depth coverage.

## 7. Explainability comparison

- **Model and data.** FloatChat Bi-LSTM, chronological split, seed 42. 100 test windows, stratified by season.
- **Targets.** Temperature at 100 dbar and salinity at 100 dbar.
- **Attribution space.** All methods attribute in scaled input units (96 features).

| Method | Setting |
|---|---|
| LIME | lime_tabular regression, 2000 perturbations, no discretisation. Coefficients converted to additive contributions w_i (x_i - mu_i) / s_i |
| SHAP | KernelSHAP, 1000 coalitions, k-means background of 25 training windows |
| FloatChat IG | Integrated Gradients via SingleOutputModelWrapper, climatological (zero scaled) baseline, 64 steps. Also produces temporal weights and the 16x16 depth matrix |

| Metric | Meaning |
|---|---|
| Deletion AUC (higher better) | Inputs replaced by climatology in order of importance; normalised output change. Random ordering reported as a reference |
| Insertion error AUC (lower better) | Inputs restored from climatology in order of importance; normalised distance to f(x) |
| Faithfulness correlation | Pearson r between the summed attribution of 64 random 16-input subsets and the output change on removing them |
| Robustness | Cosine similarity of attributions for x and x + N(0, 0.05), on 30 windows |
| Repeatability | Cosine similarity of two runs on the same x (tests randomness in LIME and SHAP) |
| Completeness error | \|sum(a) - (f(x) - f(reference))\| / \|f(x) - f(reference)\| |
| Runtime | Seconds per explanation on the same machine |
| Agreement | Spearman rank correlation and top-10 overlap between methods |
| Physical plausibility | Share of attribution on t-1, on the target variable, and within +/- 1 level of the target depth |
| Depth matrices | 16x16 matrices for all three methods (20 windows); similarity between methods reported |
| IG convergence delta | Median and maximum |

## 8. Figures produced (PNG 300 dpi + PDF)

**Data**

- `fig_data_and_splits`: target positions and windows per year for both splits.

**Accuracy** (one set per split)

- `fig_model_comparison_*`: RMSE with CIs and inversion rate.
- `fig_depth_rmse_*`: RMSE by depth and skill against persistence.
- `fig_seasonal_rmse_*`
- `fig_uq_calibration_*`
- `fig_scatter_100dbar_*`
- `fig_example_forecast_*`
- `fig_training_curves_*`

**Explainability**

- `fig_xai_heatmap_*`: mean attribution share over cycle x level.
- `fig_xai_depth_profile_*`
- `fig_xai_deletion_insertion_*`
- `fig_xai_example_*`
- `fig_xai_depth_matrix_*`
- `fig_xai_metric_summary`
- `fig_xai_temporal_weights`
- `fig_xai_agreement`

## 9. Reproducibility

Every stage writes its settings and counts to JSON, including the download log, cleaning counts, split summary, metrics with settings, and environment versions. The report includes all of these. Re-running with the same downloaded files and seeds reproduces the splits exactly. Neural results can still vary slightly across hardware and library versions.
