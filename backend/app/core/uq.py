"""
Monte Carlo Dropout Uncertainty Quantification Service (calibrated)

- Keeps dropout active during evaluation and runs N stochastic forward passes
- Computes mean, std, 90% CI, 95% CI per depth level
- Multiplies the raw dropout std by a per-variable scale factor read from
  app/ml/artifacts/uq_calibration.json (fitted on the validation windows only).
  On the 2019-2026 test windows raw 95% intervals covered about 47% (temp) / 49% (sal)
  of observations; the scaled intervals cover about 95%.
  If the file is missing, factors default to 1.0 (raw behaviour).
- The model is returned to eval() mode afterwards.
"""
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from app.ml.model import PhysicsInformedBiLSTM


@dataclass
class UncertaintyBound:
    """Uncertainty bounds for a single depth level (std_* are CALIBRATED stds)."""
    depth_dbar: int
    mean_temp: float
    std_temp: float
    ci90_temp_lower: float
    ci90_temp_upper: float
    ci95_temp_lower: float
    ci95_temp_upper: float
    mean_sal: float
    std_sal: float
    ci90_sal_lower: float
    ci90_sal_upper: float
    ci95_sal_lower: float
    ci95_sal_upper: float


STANDARD_DEPTHS = [5, 20, 50, 75, 100, 150, 200, 250, 300, 400, 500, 600, 700, 800, 900, 1000]
Z_90 = 1.645
Z_95 = 1.960
MC_PASSES = 50

CALIB_PATH = Path(__file__).resolve().parent.parent / "ml" / "artifacts" / "uq_calibration.json"


def load_uq_calibration(path: Path = CALIB_PATH) -> Dict[str, float]:
    """Return {'temp': s_T, 'sal': s_S}; 1.0 each if the artifact is absent or invalid."""
    try:
        d = json.loads(Path(path).read_text())
        s = {"temp": float(d["temp"]), "sal": float(d["sal"])}
        if s["temp"] > 0 and s["sal"] > 0:
            return s
    except Exception:
        pass
    return {"temp": 1.0, "sal": 1.0}


def run_mc_dropout_inference(
    model: PhysicsInformedBiLSTM,
    x_scaled: torch.Tensor,
    temp_scaler,
    sal_scaler,
    n_passes: int = MC_PASSES,
    std_scale: Optional[Dict[str, float]] = None,
) -> Tuple[List[UncertaintyBound], Dict[str, np.ndarray]]:
    """
    Run Monte Carlo Dropout inference and return calibrated bounds.

    std_scale: {'temp': s_T, 'sal': s_S}; None -> read uq_calibration.json (default 1.0).
    The forecast mean is unchanged by calibration; only the spread is rescaled.
    """
    scale = std_scale if std_scale is not None else load_uq_calibration()

    if x_scaled.dim() == 2:
        x_scaled = x_scaled.unsqueeze(0)
    device = next(model.parameters()).device
    x_scaled = x_scaled.to(device)

    all_temp_preds, all_sal_preds = [], []
    model.train()  # dropout ON
    try:
        with torch.no_grad():
            for _ in range(n_passes):
                temp_scaled, sal_scaled = model(x_scaled)
                all_temp_preds.append(temp_scaler.inverse_transform(temp_scaled.cpu().numpy()))
                all_sal_preds.append(sal_scaler.inverse_transform(sal_scaled.cpu().numpy()))
    finally:
        model.eval()

    temp_preds = np.stack(all_temp_preds, axis=0)  # (passes, n, 16)
    sal_preds = np.stack(all_sal_preds, axis=0)

    mean_temp = np.mean(temp_preds, axis=0).squeeze(0)
    mean_sal = np.mean(sal_preds, axis=0).squeeze(0)
    raw_std_temp = np.std(temp_preds, axis=0, ddof=1).squeeze(0)
    raw_std_sal = np.std(sal_preds, axis=0, ddof=1).squeeze(0)
    std_temp = raw_std_temp * scale["temp"]
    std_sal = raw_std_sal * scale["sal"]

    bounds = []
    for i, depth in enumerate(STANDARD_DEPTHS):
        bounds.append(UncertaintyBound(
            depth_dbar=depth,
            mean_temp=float(mean_temp[i]),
            std_temp=float(std_temp[i]),
            ci90_temp_lower=float(mean_temp[i] - Z_90 * std_temp[i]),
            ci90_temp_upper=float(mean_temp[i] + Z_90 * std_temp[i]),
            ci95_temp_lower=float(mean_temp[i] - Z_95 * std_temp[i]),
            ci95_temp_upper=float(mean_temp[i] + Z_95 * std_temp[i]),
            mean_sal=float(mean_sal[i]),
            std_sal=float(std_sal[i]),
            ci90_sal_lower=float(mean_sal[i] - Z_90 * std_sal[i]),
            ci90_sal_upper=float(mean_sal[i] + Z_90 * std_sal[i]),
            ci95_sal_lower=float(mean_sal[i] - Z_95 * std_sal[i]),
            ci95_sal_upper=float(mean_sal[i] + Z_95 * std_sal[i]),
        ))

    raw_data = {
        "temp_preds": temp_preds, "sal_preds": sal_preds,
        "mean_temp": mean_temp, "mean_sal": mean_sal,
        "std_temp": std_temp, "std_sal": std_sal,
        "std_temp_raw": raw_std_temp, "std_sal_raw": raw_std_sal,
        "scale": scale,
    }
    return bounds, raw_data


def validate_uncertainty_bounds(bounds: List[UncertaintyBound]) -> Dict[str, bool]:
    """Acceptance checks: upper CI > mean everywhere; thermocline spread > abyssal spread."""
    results = {"upper_gt_mean_temp": True, "upper_gt_mean_sal": True,
               "thermocline_var_gt_abyssal": False}
    for b in bounds:
        if not (b.ci95_temp_upper > b.mean_temp):
            results["upper_gt_mean_temp"] = False
        if not (b.ci95_sal_upper > b.mean_sal):
            results["upper_gt_mean_sal"] = False
    thermo_std_temp = np.array([b.std_temp for b in bounds[2:6]])
    if np.mean(thermo_std_temp) > bounds[15].std_temp * 2:
        results["thermocline_var_gt_abyssal"] = True
    return results


def uncertainty_bounds_to_dict(bounds: List[UncertaintyBound]) -> List[dict]:
    """Convert UncertaintyBound list to JSON-serializable dict list."""
    return [b.__dict__.copy() for b in bounds]
