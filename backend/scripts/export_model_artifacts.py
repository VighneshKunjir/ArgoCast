#!/usr/bin/env python
"""Export a benchmark-trained ArgoCast PI-BiLSTM checkpoint as the app's serving model.

The benchmark (research/benchmark/run_benchmark.py) trains the model on
the Arabian Sea 2000-2026 windows but stores only the network weights. This script
rebuilds everything else the API needs from the same split, so the served model is
exactly the one evaluated in the benchmark:

  model_weights.pt        network weights (copied from the checkpoint)
  preprocessor.joblib     per-depth StandardScalers fitted on the split's training targets
  ridge_baseline.joblib   Ridge baseline (alpha chosen on validation), shown next to each forecast
  uq_calibration.json     MC-dropout spread factors fitted on the validation windows
  metadata.json           provenance and test-set scores of the exported model

Usage (from backend/):
  python scripts/export_model_artifacts.py \
      --checkpoint research/benchmark/outputs/arabian2000_chronological/models/ArgoCast_PI-BiLSTM_seed42.pt \
      --proc-dir research/pipeline/data/processed --split chronological
"""
import argparse
import json
import sys
from datetime import date
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))

from app.ml.model import PhysicsInformedBiLSTM  # noqa: E402

DEPTHS = [5, 20, 50, 75, 100, 150, 200, 250, 300, 400, 500, 600, 700, 800, 900, 1000]
N = len(DEPTHS)
ARTIFACTS = BACKEND / "app" / "ml" / "artifacts"


def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def scale_x(X, ts, ss):
    out = np.empty_like(X, dtype=np.float32)
    for k in range(X.shape[1]):
        out[:, k, :N] = ts.transform(X[:, k, :N])
        out[:, k, N:] = ss.transform(X[:, k, N:])
    return out


def inversion_rate(t, s, lat, lon, thr=0.03):
    """% of profiles with a TEOS-10 potential-density decrease larger than thr between adjacent levels."""
    import gsw
    p = np.broadcast_to(np.array(DEPTHS, float), t.shape)
    sa = gsw.SA_from_SP(s, p, lon[:, None], lat[:, None])
    sig = gsw.sigma0(sa, gsw.CT_from_t(sa, t, p))
    return float((np.diff(sig, axis=1) < -thr).any(axis=1).mean() * 100)


@torch.no_grad()
def mc_stats(model, X, passes, seed=0):
    torch.manual_seed(seed)
    model.train()  # dropout on
    s1 = s2 = 0
    for _ in range(passes):
        a, b = model(X)
        z = torch.cat([a, b], 1).numpy().astype(np.float64)
        s1 = s1 + z
        s2 = s2 + z ** 2
    model.eval()
    mu = s1 / passes
    return mu, np.sqrt(np.maximum(s2 / passes - mu ** 2, 0))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--proc-dir", type=Path, required=True, help="folder with windows.npz and split_<name>.npz")
    ap.add_argument("--split", default="chronological", choices=["chronological", "unseen_floats"])
    ap.add_argument("--mc-passes", type=int, default=50)
    ap.add_argument("--out", type=Path, default=ARTIFACTS)
    args = ap.parse_args()

    w = np.load(args.proc_dir / "windows.npz")
    sp = np.load(args.proc_dir / f"split_{args.split}.npz")
    tr, va, te = sp["train"], sp["val"], sp["test"]
    X, yT, yS = w["X"].astype(np.float32), w["yT"].astype(np.float64), w["yS"].astype(np.float64)

    ts = StandardScaler().fit(yT[tr])
    ss = StandardScaler().fit(yS[tr])
    Xs = {k: scale_x(X[idx], ts, ss) for k, idx in (("train", tr), ("val", va), ("test", te))}

    model = PhysicsInformedBiLSTM()
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
    model.eval()
    with torch.no_grad():
        a, b = model(torch.from_numpy(Xs["test"]))
    pT, pS = ts.inverse_transform(a.numpy()), ss.inverse_transform(b.numpy())
    persT = X[te][:, -1, :N]
    inv = inversion_rate(pT, pS, w["lat"][te], w["lon"][te])

    # MC-dropout spread factor: 95th percentile of |error| / sd on validation, divided by 1.96
    mu_z, sd_z = mc_stats(model, torch.from_numpy(Xs["val"]), args.mc_passes)
    calib = {}
    for name, sl, scaler, y in (("temp", slice(0, N), ts, yT[va]), ("sal", slice(N, 2 * N), ss, yS[va])):
        mu = scaler.inverse_transform(mu_z[:, sl])
        sd = np.maximum(sd_z[:, sl] * scaler.scale_, 1e-9)
        calib[name] = float(np.quantile(np.abs(y - mu) / sd, 0.95) / 1.96)

    # Ridge baseline on the flattened, standardised history (alpha chosen on validation)
    Ytr = np.concatenate([ts.transform(yT[tr]), ss.transform(yS[tr])], 1)
    flat = {k: v.reshape(len(v), -1) for k, v in Xs.items()}
    scores = {}
    for alpha in (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0):
        p = Ridge(alpha=alpha).fit(flat["train"], Ytr).predict(flat["val"])
        scores[alpha] = rmse(ts.inverse_transform(p[:, :N]), yT[va])
    alpha = min(scores, key=scores.get)
    ridge = Ridge(alpha=alpha).fit(flat["train"], Ytr)
    rp = ridge.predict(flat["test"])
    ridge_T = rmse(ts.inverse_transform(rp[:, :N]), yT[te])

    args.out.mkdir(parents=True, exist_ok=True)
    # saved as CPU tensors so the API loads it on machines without a GPU
    torch.save({k: v.cpu() for k, v in model.state_dict().items()}, args.out / "model_weights.pt")
    joblib.dump({"temp_scaler": ts, "sal_scaler": ss, "standard_depths": DEPTHS}, args.out / "preprocessor.joblib")
    joblib.dump({"model": ridge, "alpha": alpha, "input": "standardised (3 x 32) history, flattened"},
                args.out / "ridge_baseline.joblib")
    (args.out / "uq_calibration.json").write_text(json.dumps({
        **calib,
        "method": "95th percentile of |y-mu|/sd on the validation windows divided by 1.96",
        "fitted_on": f"validation windows of the {args.split} split ({len(va)} windows)",
        "mc_passes": args.mc_passes,
        "note": "Multiplies the MC-dropout std; the forecast mean is unchanged",
    }, indent=2))
    meta = {
        "model_name": "ArgoCast PI-BiLSTM",
        "model_version": "v2.0.0-pibilstm-arabian2000",
        "architecture": "2-layer bidirectional LSTM, 128 units per direction, dropout 0.2, separate T and S heads",
        "loss": "MSE(T) + 2.5 MSE(S) + 10 stability hinge (TEOS-10 consistent) + 1.5 thermocline gradient",
        "export_date": date.today().isoformat(),
        "checkpoint": args.checkpoint.name,
        "dataset": "Arabian Sea (10-25N, 50-75E) Argo profiles 2000-2026, 16 levels 5-1000 dbar",
        "split": args.split,
        "n_windows": {"train": int(len(tr)), "val": int(len(va)), "test": int(len(te))},
        "n_floats": {k: int(np.unique(w["wmo"][idx]).size) for k, idx in (("train", tr), ("val", va), ("test", te))},
        "train_target_dates": [str(np.datetime64(int(w["target_date"][tr].min()), "s"))[:10],
                               str(np.datetime64(int(w["target_date"][tr].max()), "s"))[:10]],
        "test_target_dates": [str(np.datetime64(int(w["target_date"][te].min()), "s"))[:10],
                              str(np.datetime64(int(w["target_date"][te].max()), "s"))[:10]],
        "test_rmse_temp": round(rmse(pT, yT[te]), 4),
        "test_rmse_sal": round(rmse(pS, yS[te]), 4),
        "test_rmse_temp_persistence": round(rmse(persT, yT[te]), 4),
        "test_rmse_temp_ridge": round(ridge_T, 4),
        "test_inversion_rate_pct_teos10_gt_0.03": round(float(inv), 3),
        "uq_scale": calib,
        "ridge_alpha": alpha,
    }
    (args.out / "metadata.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
