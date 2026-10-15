# FloatChat: Architectural Memory & Decision Log

## 1. Architectural Decisions (ADRs)

### ADR-001: Merging Evidence-Link Protocol into Profile Forecast Tab
- **Date:** 2026-09-21
- **Status:** Accepted & Implemented
- **Context:** Previously, "Evidence-Link Protocol" was a separate tab in the top navigation. Users had to switch tabs to verify the raw NetCDF data citations corresponding to the forecast they were viewing.
- **Decision:** Remove the standalone tab and merge the `EvidenceLinkViewer` component directly at the bottom of the `ProfilePlot` view, positioned right beneath the `Target Profile:` summary banner.
- **Consequences:** Provides an uninterrupted scientific workflow from high-level visual forecast curve down to granular NetCDF provenance on a single scrollable page.

### ADR-002: Tab Re-sequencing & Renaming
- **Date:** 2026-09-21
- **Status:** Accepted & Implemented
- **Context:** Tab labels contained redundant regional words ("Arabian Sea Map"), and the order did not match the typical researcher workflow.
- **Decision:** Removed "Arabian" from tab labels and established the fixed sequence:
  1. Profile Forecast & UQ
  2. FloatChat
  3. Sea Map
  4. XAI & TEOS-10 Physics
  5. Model Benchmarks
- **Consequences:** Cleaner navigation hierarchy aligned with user expectations.

### ADR-003: Selection of Dual-Mode Database Architecture (SQLite + PostgreSQL)
- **Date:** 2026-09-21
- **Status:** Accepted
- **Context:** CLI coding agents and developers may run in diverse environments (some without Docker or local PostgreSQL daemons installed).
- **Decision:** Use SQLAlchemy 2.0 with standard SQL DDL so the backend defaults to zero-config file-based SQLite (`sqlite:///./data/floatchat.db`), while supporting PostgreSQL seamlessly via the `DATABASE_URL` environment variable.
- **Consequences:** Zero-friction local startup while remaining fully production-ready.

### ADR-004: High-End GPU Training Prioritization with CPU Fallback
- **Date:** 2026-09-21
- **Status:** Accepted
- **Context:** User has a high-end dedicated GPU workstation. Training and forward-backward propagation should leverage CUDA first.
- **Decision:** All PyTorch code defaults to `torch.device("cuda" if torch.cuda.is_available() else "mps" if ... else "cpu")`. Training loops enable `torch.backends.cudnn.benchmark = True` and mixed precision.
- **Consequences:** Fast model training (< 30 seconds) on GPU with transparent fallback on CPU-only CI environments.

### ADR-005: Self-Contained Embedded Dataset Default with Future Scope for Live FTP
- **Date:** 2026-09-21
- **Status:** Accepted
- **Context:** Remote GDAC FTP mirrors are prone to firewall blocks, timeouts, and network failures in sandboxed environments.
- **Decision:** Default local development database seeding strictly to `backend/data/seed_reference_profiles.json` (35 historical cycles across the 4 operational Arabian Sea reference floats: 3902114, 2903334, 1902442, 2902789).
- **Consequences:** Database seeding completes 100% offline in $< 2$ seconds.

### ADR-006: Captum Multi-Output Single Target Wrapper
- **Date:** 2026-09-21
- **Status:** Accepted
- **Context:** Captum Integrated Gradients crashes on multi-head Bi-LSTM outputs returning tuples or multi-column tensors.
- **Decision:** Require `SingleOutputModelWrapper` isolating a specific target variable and depth index before computing attributions.

### ADR-007: Dual-Runtime Compatibility & Pydantic Request Alias Handling
- **Date:** 2026-09-21
- **Status:** Accepted
- **Context:** The frontend React components (`src/App.tsx`, `src/components/ChatPanel.tsx`) submit payload parameters in camelCase (`wmoId`, `cycle`), while standard Python conventions use snake_case (`wmo_id`, `cycle_number`). Furthermore, AI Studio enforces Port 3000 as the sole reverse proxy port.
- **Decision:** All FastAPI Pydantic request models must enable `populate_by_name=True` with explicit aliases (e.g., `Field(alias="wmoId")` and `Field(alias="cycle")`) to accept both conventions seamlessly. The dev environment routes requests through Port 3000 to maintain uninterrupted live preview.
- **Consequences:** Eliminates 422 Unprocessable Entity runtime errors when switching between frontend mock server and real Python FastAPI backend.

### ADR-008: 30-Float 4-Year NetCDF Pipeline for Operational Model Scaling
- **Date:** 2026-09-22
- **Status:** Accepted
- **Context:** While the 4-float seed dataset is ideal for instant automated testing and offline development, training deep sequence models requires statistical diversity across seasonal monsoonal cycles and broader Arabian Sea hydrography.
- **Decision:** Implement `backend/scripts/download_argo.py` to ingest 4-year NetCDF data for 30 operational Arabian Sea floats, clean fill values and unphysical outliers, filter by QC flags (1/2), and interpolate to the canonical 16-level depth grid. Export as a unified wide-format CSV (`backend/data/processed/argo_30floats_canonical.csv`) with one row per profile.
- **Consequences:** Provides a scalable, high-volume training corpus for the PyTorch Bi-LSTM and Gradient Boosting baselines while preserving fast offline testing with `seed_reference_profiles.json`.

### ADR-009: Evidence-Based Recalibration of Model Acceptance Targets
- **Date:** 2026-09-24
- **Status:** Accepted (user-directed)
- **Context:** The original targets (temp ≤0.23 °C, sal ≤0.052 PSU, inversion ≤0.05%) were set against a reference 30-float training program. Six measured CPU runs (seed 42, float-level spatial splits, 47-float clean CSV) plateau at temp 0.51-0.52 °C with measured persistence at 0.544 °C on identical splits — the model beats persistence (+6% temp, +11% sal) but cannot reach 0.23 under spatial generalization across water-mass regimes.
- **Decision:** Recalibrate to temp RMSE ≤0.50 °C (≥5% better than persistence on identical splits), sal RMSE ≤0.11 PSU, thermocline RMSE ≤0.75 °C, inversion rate ≤1.0%. Each bar is either met with margin (sal, inversions) or a small reachable step away (temp, thermocline), and all remain operationally meaningful alongside UQ, physics validation, and evidence grounding.
- **Consequences:** `ml_spec.md` acceptance table and TASK-303 criteria updated. Original 0.23 °C retained as a stretch research goal, not a gate.

---

## 2. Tracking `[ASSUMPTION]` and Resolved Decisions

### Active Assumptions:
- `[ASSUMPTION-006]` (2026-09-25, UI bugfix hardening): The "only 4 floats in dropdown" report is an environment/seed artifact, not a query bug — `/api/floats` verified live returning all 47 floats from `backend/data/floatchat.db` (9,687 profiles). A DB seeded with only `seed_db.py` yields exactly 4 floats; canonical full seeding is `seed_all_floats.py` (idempotent). Floats endpoint now orders by WMO for a stable dropdown.
- `[ASSUMPTION-005]` (2026-09-24, TASK-303 iteration): 6 CPU runs (deterministic, seed 42). (a)-(c) contaminated CSV: 0.5565 / 0.6330-basin / 0.5565-plateau. (d) temporal last-10-cycle holdout → 1.0291 C (worse; recent cycles harder). (e) physics-loss units fix (denormalize before TEOS-10 polynomial) → inversions 36.5% → 0.225%, RMSE flat. (f) 47-float clean CSV (9,680 rows, 100% retention) + stratified splits → test 0.5127 C / sal 0.1065 / inv 1.6%, val stabilized (0.64-0.72); persistence on identical splits 0.5440 C — model beats persistence on all metrics (+6% temp, +11% sal) but far from 0.23. Per-float val diagnosis: 1902442 (Southern Ocean) 3.09 and 6903063 (Arctic, 7 windows) 3.20 dominate val error; in-scope floats 0.53-0.63. 3 floats (2901431/2901447/2901466) yield zero profiles — PSAL_QC=4 fleet-wide, correctly excluded per data_spec. Env torch is CPU-only (`2.14.0+cpu`) despite RTX 3060 Ti (fix: `pip install torch==2.14.0+cu126 --index-url https://download.pytorch.org/whl/cu126`); GPU speeds iteration, not the math. TASK-303 (≤0.23 C) left open.
- `[ASSUMPTION-001]`: The 16 canonical standard pressure levels ($5, 20, 50, 75, 100, 150, 200, 250, 300, 400, 500, 600, 700, 800, 900, 1000\text{ dbar}$) are universally sufficient for Arabian Sea upper-ocean and thermocline characterization, matching the operational frontend visualization and deep abyssal layer.
- `[ASSUMPTION-002]`: A 3-cycle historical window ($t-3, t-2, t-1$, representing a 30-day temporal lag) provides optimal balance between capturing seasonal drift and minimizing missing-cycle gaps in float lifetime.
- `[ASSUMPTION-003]`: In offline environments or when `GEMINI_API_KEY` is not provided, users prefer an instant, high-fidelity analytical response rather than an error modal.

### Resolved Decisions:
- `[RESOLVED-009]` (2026-10-15): **10-model benchmark with extended metrics (isolated).** `backend/experiments/benchmark_10_models/` trains 10 comparison models plus the MSE-only and physics Bi-LSTM on the Arabian Sea 2000-2026 splits under one shared protocol (GPU for neural models). FloatChat is retrained there from `app/ml/model.py` + `app/ml/loss.py`, not loaded from the deployed checkpoint. `metrics_extra.py` adds metric groups 1-7 (error size, ACC, skill, depth/season errors, MLD / D20 / OHC errors, inversion strength, MC-dropout coverage and CRPS) and a separate classification view (direction-of-change and stability-detection precision/recall/F1), computed for every seed. Climatology is reported twice: regional monthly, and location-by-month (1/2/3 deg cell chosen on validation, >= 10 profiles per cell, regional fallback); the local one is the ACC / skill reference. MLD is limited by the 16-level grid. No changes to `app/` or `src/`.
- `[RESOLVED-008]` (2026-10-13): **Extended evaluation experiment (isolated).** Added `backend/experiments/arabian_sea_2000_2026/` (stand-alone scripts, no changes to `app/` or `src/`). It downloads every float in 10-25N/50-75E for 2000-2026 via per-float `_prof.nc`, keeps D+A modes (adjusted variables) so 2023-2026 is covered (`--modes D` restores D-only), limits end-of-profile fill to 50 dbar, and requires 3-20 day steps inside a window. Two leakage-safe splits: chronological with purge (primary) and unseen floats (70/15/15, seed 42). Compares FloatChat against persistence, climatology, ridge, GB and an MSE-only Bi-LSTM ablation, and compares LIME vs KernelSHAP vs FloatChat IG. Design in `EXPERIMENT_PROTOCOL.md`; results auto-written to `outputs/RESULTS_ArabianSea_2000_2026.md`.
- `[RESOLVED-007]` (2026-09-25): **Forecastable-cycle gating.** The Bi-LSTM needs 3 prior cycles, so base cycles at index <3 always 400. Frontend now hides the first 3 cycles per float (`ControlBar` slices `cycles.slice(3)`); backend 400 message names the earliest valid cycle; `apiService` forwards backend `detail` text so users see actionable guidance instead of bare "Bad Request".
- `[RESOLVED-006]` (2026-09-24): **MC Dropout UQ operational.** `backend/app/core/uq.py` runs 50 stochastic forward passes with active dropout (p=0.2), computes mean/std/90%/95% CI per depth, outputs `UncertaintyBound[]` matching `src/types.ts`. Validated: upper CI > mean, thermocline variance check in place. Returns structured `UncertaintyBound[]` list for API consumption.
- `[RESOLVED-005]` (2026-09-24): **Measured baselines replace reference-table fiction.** TASK-304 acceptance says metrics should "match Table in ml_spec.md", but on our 47-float clean CSV the measured values differ (persistence 0.544 vs 0.48; GB 0.507 vs 0.341). Per `rules.md` Data Authenticity (no simulated numbers in production pathways), `baseline_metrics.json` ships measured values on identical test windows; the qualitative pattern matches the spec table (GB beats persistence on RMSE but violates physics at 23.9%; LSTM wins physics at 0.30% and thermocline). GB uses per-depth HistGradientBoostingRegressor via MultiOutputRegressor (installed sklearn HGB is single-output only).
- `[RESOLVED-004]` (2026-09-24): **Upstream NetCDF cleaning + stratified splits.** `process_argo_netcdf.py` now masks fill sentinels pre-interpolation, enforces hard physical bounds per measurement, discards cycles with any out-of-bounds/NaN interpolated level, and flags (not drops) density-unstable profiles. `train_model.py` uses stratified domain-aware float-level splits (Arctic 69030xx distributed train/val, all-Arabian test, seed 42) — compliant with `rules.md` split rule as written, so no spec change needed. Clean CSV regenerated at 50 floats: 9,680 profiles × 47 floats (3 floats excluded for fleet-wide PSAL_QC=4), 100% train-time retention, 83.1% stable. Raw NetCDF files untouched per user direction.
- `[RESOLVED-001]`: **Forecast Persistence:** Persist all run forecasts into the `forecast_logs` table via `POST /api/forecast`. Provide `GET /api/history` for audit and retrospective visualization.
- `[RESOLVED-002]`: **Forecast Export Format:** Standard JSON export (`GET /api/forecast/{id}/export-json` or client-side download) is implemented now. CF-compliant NetCDF (`.nc`) export is logged under Future Scope.
- `[RESOLVED-003]`: **Runtime Architecture:** Dual-mode in development (FastAPI on 8000, Vite on 3000 proxying `/api`). In production Docker, single FastAPI service serving the static `dist/` bundle on port 3000.

---

## 3. Gotchas & Domain Insights
1. **KaTeX CSS Requirement:** The KaTeX mathematical stylesheet must be explicitly included in `index.html` (`<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/katex.min.css">`). Without this, mathematical equations like $\frac{\partial\sigma_\theta}{\partial z}$ render as unstyled raw text.
2. **Pressure vs. Depth:** In oceanography, pressure in decibars (`dbar`) is approximately equal to depth in meters (1 dbar $\approx$ 0.993 m at low latitudes). Vertical profile plots must invert the vertical axis (0 dbar at top, 1000 dbar at bottom).
3. **Delayed-Mode vs. Real-Time QC:** Only use delayed-mode (`DATA_MODE = 'D'`) or QC flags 1/2 for model training and evidence citations. Real-time (`'R'`) uncorrected data often contain sensor drift spikes near the surface.
