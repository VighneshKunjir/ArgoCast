# FloatChat 10-model benchmark - how to run (GPU)

Trains and evaluates 10 comparison forecasters plus the two FloatChat Bi-LSTM rows on the
405-float Arabian Sea 2000-2026 dataset, on both splits (chronological and unseen floats).

| # | Model | Runs on |
|---|---|---|
| 1 | Persistence | - |
| 2 | Climatology | - |
| 3 | Ridge | CPU |
| 4 | LightGBM | CPU |
| 5 | GRU | GPU |
| 6 | Seq2Seq LSTM + attention | GPU |
| 7 | TCN | GPU |
| 8 | DLinear | GPU |
| 9 | N-HiTS | GPU |
| 10 | TFT | GPU |
| ref | Bi-LSTM (MSE only), FloatChat Bi-LSTM | GPU |

## Folder layout

Put this folder at `backend\experiments\benchmark_10_models\` (it imports `app.ml.model` and
`app.ml.loss`). It reads the stage-3 outputs of the Arabian Sea experiment:

```
backend\experiments\arabian_sea_2000_2026\data\processed\windows.npz
backend\experiments\arabian_sea_2000_2026\data\processed\split_chronological.npz
backend\experiments\arabian_sea_2000_2026\data\processed\split_unseen_floats.npz
```

If these files are missing, rebuild them with
`python s02_build_dataset.py` then `python s03_split.py` in `arabian_sea_2000_2026\`.

## 1. Install (once)

```
pip install lightgbm
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

The second line must print `True` and your GPU name. The script stops if no GPU is visible.

## 2. Smoke test (a few minutes)

```
cd E:\Projects\FloatChat_v6\FloatChat_v6\backend\experiments\benchmark_10_models
python run_benchmark.py --dataset arabian2000 --quick
```

Writes to `outputs\_quick\`. Those numbers are not results; delete the folder afterwards.

## 3. Full run

```
python run_benchmark.py --dataset arabian2000
```

This runs both splits with 3 seeds. To halve the time, use one seed:

```
python run_benchmark.py --dataset arabian2000 --seeds 42
```

Optional: also rerun the 109-float set used in the current paper:

```
python run_benchmark.py --dataset canonical
```

## 4. Results

| File | Contents |
|---|---|
| `outputs\BENCHMARK_10_MODELS_RESULTS.md` | Combined ranking across all completed runs |
| `outputs\arabian2000_chronological\RESULTS.md` | Full record: data, models, main table with 95% CIs, seed stability, paired comparison vs FloatChat, per-depth errors, cost, interpretability signals |
| `outputs\arabian2000_unseen_floats\RESULTS.md` | Same, for the unseen-float split |
| `outputs\<run>\metrics.json`, `metrics.csv`, `per_depth_*.csv` | Raw numbers |
| `outputs\<run>\figures\*.png / *.pdf` | Paper figures (300 dpi PNG and vector PDF) |

Add the `.md` files and figures to the project so the paper can be updated from them.

## Fairness rules built into the script

- Same data, same standardisation (fitted on training targets only), same test windows for every model.
- All neural models: AdamW (lr 1e-3, wd 1e-4), cosine schedule, gradient clipping 5, batch 64,
  up to 60 epochs, early stopping (patience 10) on validation temperature RMSE.
- Same data loss for all (MSE_T + 2.5 MSE_S); only FloatChat Bi-LSTM adds the stability (10.0)
  and thermocline (1.5) penalties.
- Ridge alpha and LightGBM tree count are chosen on validation only.
- Confidence intervals use a float-level block bootstrap with identical resamples for every model,
  so paired differences against FloatChat and persistence are valid.
