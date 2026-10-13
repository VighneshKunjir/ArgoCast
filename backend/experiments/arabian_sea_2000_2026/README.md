# Arabian Sea 2000-2026 experiment - how to run

This folder retrains and re-evaluates the FloatChat forecaster on every Argo float that profiled the Arabian Sea (10-25 N, 50-75 E) between 2000 and 2026. It also compares three explanation methods: LIME, SHAP and FloatChat IG. The design is recorded in `EXPERIMENT_PROTOCOL.md`. The numbers are written by the pipeline to `outputs\RESULTS_ArabianSea_2000_2026.md`.

The scripts import `app.ml.model` and `app.ml.loss` from `backend\`, so keep this folder at `backend\experiments\arabian_sea_2000_2026\`.

## 1. Install (once)

Open a terminal in this folder, inside the same Python environment you use for the backend:

```
cd E:\Projects\FloatChat_v6\FloatChat_v6\backend\experiments\arabian_sea_2000_2026
pip install -r requirements_experiment.txt
```

## 2. Smoke test (about 5-15 minutes)

```
python s01_download.py --max-floats 20
python run_all.py --quick --skip-download
```

If this finishes and writes `outputs\RESULTS_ArabianSea_2000_2026.md`, everything is wired correctly. Before the full run, delete the smoke-test files:

- `data\raw\*`
- `data\processed\*`
- `outputs\`

Keep `data\ar_index_global_prof.txt.gz`; it is reused.

## 3. Full run

```
python run_all.py
```

Each stage can be restarted on its own. If a stage fails, `run_all.py` prints the `--from N` option to restart from there.

| Stage | Script | What it does | Rough cost |
|---|---|---|---|
| 1 | `s01_download.py` | Index plus one `*_prof.nc` per float. Resumable: rerun to retry failures | Index about 50 MB; float files typically a few GB; depends on your connection |
| 2 | `s02_build_dataset.py` | QC, interpolation to 16 levels, TEOS-10 flags | Minutes; uses all CPU cores |
| 3 | `s03_split.py` | Windows plus chronological and unseen-float splits | Seconds |
| 4 | `s04_train_evaluate.py` | 6 models x 2 splits, 3 seeds for the neural models, MC dropout, bootstrap | Longest stage; hours on CPU for tens of thousands of windows; much faster with a CUDA GPU |
| 5 | `s05_xai_compare.py` | LIME vs SHAP vs IG on 100 test windows, 2 targets | Tens of minutes; SHAP dominates |
| 6 | `s06_report.py` | Writes the results Markdown | Seconds |

### Cutting stage 4 time on a CPU laptop

Any extra options given to `run_all.py` are passed to stage 4:

```
python run_all.py --from 4 --seeds 42
python run_all.py --from 4 --seeds 42 --splits chronological --batch-size 128
```

### Useful options

- `python s02_build_dataset.py --modes D` keeps delayed-mode profiles only, as in the earlier runs. The default `DA` adds adjusted real-time profiles so that 2023-2026 is covered.
- `python s03_split.py --t1 2019-01-01 --t2 2022-01-01` fixes the chronological boundaries instead of the automatic 70 / 85 % placement.
- `python s05_xai_compare.py --targets temp:4 sal:4 temp:1` adds temperature at 20 dbar. Depth index 0-15 maps to 5 ... 1000 dbar.

## 4. What to add to the project afterwards

- `outputs\RESULTS_ArabianSea_2000_2026.md`: the record with all tables.
- `outputs\figures\*.png` (300 dpi) and `*.pdf` (vector, for the paper).
- `outputs\tables\*.json` and `*.csv`: the raw numbers behind every table.
