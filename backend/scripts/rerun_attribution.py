#!/usr/bin/env python
"""
Re-run Integrated Gradients temporal attribution for the FINAL model over ALL test windows
(the earlier numbers came from a single window, Xte[:1], and an older 47-float model).

  python backend/scripts/rerun_attribution.py                 # defaults: depths 20,100,300,1000 dbar
  python backend/scripts/rerun_attribution.py --depth-idx 4 --max-windows 200   # quick check

Outputs (backend/app/ml/artifacts/experiments/): attribution_rerun.json, attribution_table.tex,
and figs/attribution.pdf (written next to the script's --fig-dir; default ./attribution.pdf).
Captum is called directly (not compute_integrated_gradients) because that helper squeezes
the batch dimension and calls delta.item(), which only works for a single window.
For each output we report two definitions of "temporal share":
  same_var : |attr| summed over the 16 inputs of the SAME variable (what the app shows)
  all_vars : |attr| summed over all 32 inputs
Shares are computed per window, then averaged; SD is across windows, and a second
SD is across floats (mean per float first) because windows within a float are correlated.
"""
import argparse, json, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
from experiment_uq_and_splits import load_df, windows, scale_X, pick_device, DEPTHS, ART, OUT  # noqa: E402


def main():
    import joblib, torch
    from captum.attr import IntegratedGradients
    from app.ml.model import PhysicsInformedBiLSTM
    from app.ml.xai import SingleOutputModelWrapper

    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, default=HERE.parent / "data" / "processed" / "argo_floats_canonical.csv")
    ap.add_argument("--depth-idx", type=int, nargs="+", default=[1, 4, 8, 15])
    ap.add_argument("--n-steps", type=int, default=50)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--max-windows", type=int, default=0, help="0 = all test windows")
    ap.add_argument("--fig-dir", type=Path, default=Path("."))
    ap.add_argument("--device", default="auto", help="auto | cuda | cpu")
    a = ap.parse_args()

    meta = json.loads((ART / "metadata.json").read_text())
    pre = joblib.load(ART / "preprocessor.joblib")
    ts, ss = pre["temp_scaler"], pre["sal_scaler"]
    model = PhysicsInformedBiLSTM()
    model.load_state_dict(torch.load(ART / "model_weights.pt", map_location="cpu"))
    dev = pick_device(a.device)
    model.to(dev).eval()

    df = load_df(a.csv)
    X, _, _, w = windows(df[df["wmo"].isin(meta["test_floats"])])
    Z = scale_X(X, ts, ss)
    if a.max_windows:
        Z, w = Z[:a.max_windows], w[:a.max_windows]
    print(f"{len(Z)} test windows from {len(set(w))} floats")

    res = {"n_windows": int(len(Z)), "n_floats": int(len(set(w))), "n_steps": a.n_steps, "outputs": {}}
    for var in ("temp", "sal"):
        for di in a.depth_idx:
            wrapper = SingleOutputModelWrapper(model, var, di).to(dev).eval()
            ig = IntegratedGradients(wrapper)
            shares = {"same_var": [], "all_vars": []}
            deltas = []
            for i in range(0, len(Z), a.batch):
                x = torch.from_numpy(Z[i:i + a.batch]).to(dev).requires_grad_(True)
                # cuDNN's RNN backward only works in train mode, which would switch dropout on.
                # Disabling cuDNN keeps eval() semantics (dropout off) with identical maths.
                with torch.backends.cudnn.flags(enabled=False):
                    attr, delta = ig.attribute(x, baselines=torch.zeros_like(x), n_steps=a.n_steps,
                                               return_convergence_delta=True)
                at = attr.detach().abs().cpu().numpy()               # (B,3,32)
                sl = slice(0, 16) if var == "temp" else slice(16, 32)
                for key, arr in (("same_var", at[:, :, sl]), ("all_vars", at)):
                    c = arr.sum(axis=2)                              # (B,3) per cycle
                    shares[key].append(c / np.maximum(c.sum(axis=1, keepdims=True), 1e-12))
                deltas.append(delta.detach().abs().cpu().numpy())
            deltas = np.concatenate(deltas)
            entry = {"depth_dbar": DEPTHS[di], "max_abs_delta": float(deltas.max()),
                     "mean_abs_delta": float(deltas.mean())}
            for key in shares:
                S = np.concatenate(shares[key])                      # (n,3) cycles t-3,t-2,t-1
                per_float = np.stack([S[w == f].mean(0) for f in sorted(set(w))])
                entry[key] = {"mean_pct": (100 * S.mean(0)).tolist(),
                              "sd_windows_pct": (100 * S.std(0, ddof=1)).tolist(),
                              "sd_floats_pct": (100 * per_float.std(0, ddof=1)).tolist(),
                              "per_float_pct": {str(f): (100 * per_float[j]).round(1).tolist()
                                                for j, f in enumerate(sorted(set(w)))}}
            res["outputs"][f"{var}_{DEPTHS[di]}"] = entry
            m = entry["same_var"]["mean_pct"]
            print(f"{var:4s} {DEPTHS[di]:5d} dbar  t-3/t-2/t-1 = {m[0]:.1f}/{m[1]:.1f}/{m[2]:.1f}  max|delta|={entry['max_abs_delta']:.2e}")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "attribution_rerun.json").write_text(json.dumps(res, indent=2))

    L = ["\\begin{table}[t]\\centering\\small",
         f"\\caption{{Temporal attribution shares (\\%, mean $\\pm$ SD across floats) over {res['n_windows']} test windows from {res['n_floats']} floats; same-variable inputs.}}\\label{{tab:attr}}",
         "\\begin{tabular}{llccc}\\toprule", "Output & Depth (dbar) & $t{-}3$ & $t{-}2$ & $t{-}1$\\\\\\midrule"]
    for k, e in res["outputs"].items():
        v = k.split("_")[0]; s = e["same_var"]
        L.append(f"{'Temperature' if v == 'temp' else 'Salinity'} & {e['depth_dbar']} & " +
                 " & ".join(f"{s['mean_pct'][j]:.1f}$\\pm${s['sd_floats_pct'][j]:.1f}" for j in range(3)) + "\\\\")
    L += ["\\bottomrule\\end{tabular}\\end{table}"]
    (OUT / "attribution_table.tex").write_text("\n".join(L))

    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "font.family": "serif", "axes.spines.top": False, "axes.spines.right": False})
    key100 = lambda v: f"{v}_100" if f"{v}_100" in res["outputs"] else None
    fig, ax = plt.subplots(1, 2, figsize=(6.4, 2.4), sharey=True)
    for k, (v, col, tt) in enumerate((("temp", "#c0392b", "Temperature, 100 dbar"), ("sal", "#2c6fbb", "Salinity, 100 dbar"))):
        if key100(v):
            s = res["outputs"][key100(v)]["same_var"]
            ax[k].bar(["t-3", "t-2", "t-1"], s["mean_pct"], yerr=s["sd_floats_pct"], color=col, capsize=3)
            ax[k].set_title(tt)
    ax[0].set_ylabel("Attribution share (%)")
    plt.tight_layout(); plt.savefig(a.fig_dir / "attribution.pdf")
    print("wrote", OUT / "attribution_rerun.json", OUT / "attribution_table.tex", a.fig_dir / "attribution.pdf")


if __name__ == "__main__":
    main()
