"""
Integrated Gradients explanations for the FloatChat PI-BiLSTM.

For one forecast this module computes, with Captum's IntegratedGradients:

- Temporal attribution: the share of the attribution for one target output
  (default: temperature at 100 dbar) carried by each input cycle (t-3, t-2, t-1).
- Depth-to-depth attribution matrix: for every forecast depth z_out, the share of its
  attribution that comes from each input depth z_in of the same variable (inputs of
  all three cycles summed). One IG run per output depth, so this is a direct
  measurement, not an estimate.
- Key depth influences: for four depth zones of the forecast, the input zone that
  contributes most, read off the matrix.

Integrated Gradients needs a reference input ("baseline"). The baseline is the zero
vector in standardised units, i.e. the training-mean profile at every depth, so each
attribution answers: "how much does this input's departure from the mean profile move
the forecast?". By construction the attributions of one output add up to
f(input) - f(baseline); the convergence delta reports how far the numerical integral
is from that identity.
"""
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

try:
    from captum.attr import IntegratedGradients
    CAPTUM_AVAILABLE = True
except ImportError:  # pragma: no cover
    CAPTUM_AVAILABLE = False

from app.ml.model import PhysicsInformedBiLSTM, SingleOutputModelWrapper

DEPTHS = np.array([5, 20, 50, 75, 100, 150, 200, 250, 300, 400, 500, 600, 700, 800, 900, 1000], dtype=float)
N_LEV = len(DEPTHS)
VAR_LABEL = {"temp": "temperature", "sal": "salinity"}
ZONES = [  # (label, first index, last index) on the 16-level grid
    ("Upper layer (5-50 dbar)", 0, 2),
    ("Thermocline (75-200 dbar)", 3, 6),
    ("Intermediate (250-500 dbar)", 7, 10),
    ("Deep (600-1000 dbar)", 11, 15),
]


def compute_integrated_gradients(
    model: PhysicsInformedBiLSTM,
    input_tensor: torch.Tensor,
    baseline_tensor: Optional[torch.Tensor] = None,
    target_variable: str = "temp",
    target_depth_idx: int = 4,
    n_steps: int = 50,
) -> Tuple[np.ndarray, float]:
    """Integrated Gradients for one output (variable, depth). Returns (attributions (1,3,32), delta)."""
    if not CAPTUM_AVAILABLE:
        raise RuntimeError("captum is required for Integrated Gradients: pip install captum")
    model.eval()
    wrapper = SingleOutputModelWrapper(model, target_variable=target_variable, target_depth_idx=target_depth_idx)
    ig = IntegratedGradients(wrapper)
    x = input_tensor.detach().clone().requires_grad_(True)
    base = torch.zeros_like(x) if baseline_tensor is None else baseline_tensor.detach()
    attr, delta = ig.attribute(x, baselines=base, n_steps=n_steps, return_convergence_delta=True)
    return attr.detach().cpu().numpy(), float(delta.abs().max().item())


def compute_temporal_attribution(
    attributions: np.ndarray,
    target_variable: str = "temp",
    input_cycles: Optional[List[int]] = None,
    target_depth_idx: int = 4,
) -> List[Dict]:
    """Share of |attribution| (all 32 input features) carried by each input cycle."""
    attr = np.abs(attributions.squeeze(0))          # (3, 32)
    per_cycle = attr.sum(axis=1)
    total = per_cycle.sum()
    shares = per_cycle / total if total > 0 else np.full(3, 1 / 3)
    target = f"{VAR_LABEL[target_variable]} at {int(DEPTHS[target_depth_idx])} dbar"
    out = []
    for i, share in enumerate(shares):
        offset = -(3 - i)
        cyc = f" (cycle {input_cycles[i]})" if input_cycles and i < len(input_cycles) else ""
        out.append({
            "cycle_offset": offset,
            "cycle_label": f"t{offset}{cyc}",
            "importance_score": float(share),
            "interpretation": f"{share * 100:.1f}% of the attribution for the forecast {target}.",
        })
    return out


def compute_depth_attribution_matrix(
    model: PhysicsInformedBiLSTM,
    input_tensor: torch.Tensor,
    target_variable: str = "temp",
    n_steps: int = 32,
) -> Tuple[np.ndarray, float, float]:
    """
    Measured depth-to-depth attribution for one variable.

    Returns (matrix, cross_variable_share, max_delta) where matrix[i, j] is the share of
    the attribution of forecast depth j that comes from input depth i of the same
    variable (columns sum to 1), and cross_variable_share is the mean share that comes
    from the other variable's inputs.
    """
    own = slice(0, N_LEV) if target_variable == "temp" else slice(N_LEV, 2 * N_LEV)
    other = slice(N_LEV, 2 * N_LEV) if target_variable == "temp" else slice(0, N_LEV)
    matrix = np.zeros((N_LEV, N_LEV))
    cross, max_delta = [], 0.0
    for j in range(N_LEV):
        attr, delta = compute_integrated_gradients(model, input_tensor, None, target_variable, j, n_steps)
        a = np.abs(attr.squeeze(0))                 # (3, 32)
        own_by_depth = a[:, own].sum(axis=0)        # (16,)
        total = a.sum()
        matrix[:, j] = own_by_depth / own_by_depth.sum() if own_by_depth.sum() > 0 else 0.0
        cross.append(a[:, other].sum() / total if total > 0 else 0.0)
        max_delta = max(max_delta, delta)
    return matrix, float(np.mean(cross)), max_delta


def compute_key_depth_influences(matrix: np.ndarray, target_variable: str = "temp") -> List[Dict]:
    """For each forecast depth zone, the input zone with the largest share (from the measured matrix)."""
    var = VAR_LABEL[target_variable]
    out = []
    for label, j0, j1 in ZONES:
        col = matrix[:, j0:j1 + 1].mean(axis=1)    # share by input depth for this output zone
        zone_shares = [(lab, float(col[i0:i1 + 1].sum())) for lab, i0, i1 in ZONES]
        best_label, best_share = max(zone_shares, key=lambda t: t[1])
        top_depth = int(DEPTHS[int(np.argmax(col))])
        same = best_label == label
        out.append({
            "target_zone": label,
            "dominant_input_depth": f"{best_label} (largest single level: {top_depth} dbar)",
            "attribution_percentage": round(best_share * 100, 1),
            "scientific_driver": (
                f"The forecast {var} in this zone is explained mostly by the recent {var} in the same zone."
                if same else
                f"The forecast {var} in this zone draws most on {var} from the {best_label.split(' (')[0].lower()} inputs."
            ),
        })
    return out


def compute_xai_attribution(
    model: PhysicsInformedBiLSTM,
    input_tensor: torch.Tensor,
    target_variable: str = "temp",
    target_depth_idx: int = 4,
    baseline_tensor: Optional[torch.Tensor] = None,
    n_steps: int = 50,
    input_cycles: Optional[List[int]] = None,
    **_unused,
) -> Dict:
    """
    Full explanation of one forecast.

    Returns temporal_attribution (for target_variable at target_depth_idx),
    depth_attribution_matrix (16 x 16 entries, same variable), key_depth_influences,
    convergence_delta (for the temporal target), plus metadata describing the target,
    the baseline and the share of attribution coming from the other variable.
    """
    if not CAPTUM_AVAILABLE:
        raise RuntimeError("captum is required for Integrated Gradients: pip install captum")

    attributions, delta = compute_integrated_gradients(
        model, input_tensor, baseline_tensor, target_variable, target_depth_idx, n_steps)
    temporal = compute_temporal_attribution(attributions, target_variable, input_cycles, target_depth_idx)

    a = np.abs(attributions.squeeze(0))
    k = int(np.argmax(a))
    cyc_i, feat = divmod(k, 2 * N_LEV)
    top_input = {
        "cycle_offset": -(3 - cyc_i),
        "cycle": input_cycles[cyc_i] if input_cycles and cyc_i < len(input_cycles) else None,
        "variable": "temperature" if feat < N_LEV else "salinity",
        "depth_dbar": int(DEPTHS[feat % N_LEV]),
        "share": float(a.flat[k] / a.sum()) if a.sum() > 0 else 0.0,
    }

    matrix, cross_share, matrix_delta = compute_depth_attribution_matrix(model, input_tensor, target_variable)
    depth_matrix = [
        {"input_depth": int(DEPTHS[i]), "output_depth": int(DEPTHS[j]), "saliency_weight": float(matrix[i, j])}
        for i in range(N_LEV) for j in range(N_LEV)
    ]

    return {
        "temporal_attribution": temporal,
        "depth_attribution_matrix": depth_matrix,
        "key_depth_influences": compute_key_depth_influences(matrix, target_variable),
        "convergence_delta": delta,
        "target": {"variable": VAR_LABEL[target_variable], "depth_dbar": int(DEPTHS[target_depth_idx])},
        "baseline": "training-mean profile (zero in standardised units)",
        "top_input": top_input,
        "cross_variable_share": cross_share,
        "matrix_variable": VAR_LABEL[target_variable],
        "matrix_convergence_delta": matrix_delta,
    }
