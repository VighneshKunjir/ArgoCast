"""Shared configuration and helpers for the Arabian Sea 2000-2026 experiment.

Everything in this folder is self-contained except the FloatChat model and
loss, which are imported from backend/app/ml so the experiment trains the
exact architecture used by the application.
"""
from __future__ import annotations

import json
import os
import platform
import random
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

EXP_DIR = Path(__file__).resolve().parent
BACKEND_DIR = EXP_DIR.parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

DATA_DIR = Path(os.environ.get("FC_DATA_DIR", EXP_DIR / "data"))
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
OUT_DIR = Path(os.environ.get("FC_OUT_DIR", EXP_DIR / "outputs"))
FIG_DIR = OUT_DIR / "figures"
TAB_DIR = OUT_DIR / "tables"
MODEL_DIR = OUT_DIR / "models"
PRED_DIR = OUT_DIR / "predictions"

for _d in (RAW_DIR, PROCESSED_DIR, FIG_DIR, TAB_DIR, MODEL_DIR, PRED_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# Region and period (same box as scripts/discover_arabian_sea_floats.py)
MIN_LAT, MAX_LAT = 10.0, 25.0
MIN_LON, MAX_LON = 50.0, 75.0
START_YEAR, END_YEAR = 2000, 2026

STANDARD_DEPTHS = np.array(
    [5, 20, 50, 75, 100, 150, 200, 250, 300, 400, 500, 600, 700, 800, 900, 1000],
    dtype=float,
)
N_LEV = len(STANDARD_DEPTHS)
TEMP_COLS = [f"temp_{int(d)}" for d in STANDARD_DEPTHS]
SAL_COLS = [f"sal_{int(d)}" for d in STANDARD_DEPTHS]
THERM_IDX = [2, 3, 4, 5]          # 50-150 dbar
DEEP_IDX = [10, 11, 12, 13, 14, 15]  # 500-1000 dbar

# Physical bounds (same as scripts/process_argo_netcdf.py)
TEMP_BOUNDS = (-2.0, 35.0)
SAL_BOUNDS = (30.0, 42.0)

# Mirrors of the Argo GDAC (IFREMER primary, US GODAE fallback)
MIRRORS = [
    "https://data-argo.ifremer.fr",
    "https://usgodae.org/pub/outgoing/argo",
]

SEASONS = {  # Arabian Sea monsoon calendar
    "Winter monsoon (DJF)": [12, 1, 2],
    "Spring inter-monsoon (MAM)": [3, 4, 5],
    "Summer monsoon (JJAS)": [6, 7, 8, 9],
    "Fall inter-monsoon (ON)": [10, 11],
}


def season_of(month: int) -> str:
    for name, months in SEASONS.items():
        if month in months:
            return name
    return "unknown"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass


def sigma_poly(t: np.ndarray, s: np.ndarray) -> np.ndarray:
    """Polynomial potential density used by the FloatChat loss (physical units)."""
    return 28.14 - 0.0735 * t - 0.00469 * t ** 2 + 0.802 * (s - 35.0)


def inversion_rate_poly(t: np.ndarray, s: np.ndarray, tol: float = 1e-6) -> float:
    """Percent of profiles with any density inversion (polynomial sigma, as in the repo)."""
    d = np.diff(sigma_poly(t, s), axis=1) / np.diff(STANDARD_DEPTHS)
    return float((d < -tol).any(axis=1).mean() * 100.0)


def inversion_rate_teos10(t: np.ndarray, s: np.ndarray, lat: np.ndarray, lon: np.ndarray,
                          tol: float = 1e-6) -> float:
    """Percent of profiles with any inversion using exact TEOS-10 sigma0 (gsw)."""
    import gsw
    p = np.broadcast_to(STANDARD_DEPTHS, t.shape)
    la = np.broadcast_to(lat[:, None], t.shape)
    lo = np.broadcast_to(lon[:, None], t.shape)
    sa = gsw.SA_from_SP(s, p, lo, la)
    ct = gsw.CT_from_t(sa, t, p)
    sig = gsw.sigma0(sa, ct)
    d = np.diff(sig, axis=1) / np.diff(STANDARD_DEPTHS)
    return float((d < -tol).any(axis=1).mean() * 100.0)


def rmse(a, b) -> float:
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def mae(a, b) -> float:
    return float(np.mean(np.abs(np.asarray(a) - np.asarray(b))))


def r2(pred, true) -> float:
    true = np.asarray(true)
    ss_res = np.sum((true - pred) ** 2)
    ss_tot = np.sum((true - true.mean()) ** 2)
    return float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")


def write_json(path: Path, obj) -> None:
    def _default(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (Path, datetime)):
            return str(o)
        raise TypeError(type(o))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=_default))


def read_json(path: Path):
    return json.loads(Path(path).read_text())


def environment_info() -> dict:
    info = {"python": sys.version.split()[0], "platform": platform.platform(),
            "processor": platform.processor() or platform.machine()}
    for mod in ("numpy", "pandas", "torch", "sklearn", "xarray", "gsw", "shap", "lime", "captum"):
        try:
            m = __import__(mod)
            info[mod] = getattr(m, "__version__", "installed")
        except ImportError:
            info[mod] = "missing"
    try:
        import torch
        info["device"] = ("cuda:" + torch.cuda.get_device_name(0)) if torch.cuda.is_available() else "cpu"
        info["torch_threads"] = torch.get_num_threads()
    except ImportError:
        pass
    return info


# ---------------------------------------------------------------- plotting
def setup_matplotlib():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 300, "savefig.bbox": "tight",
        "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.25, "legend.frameon": False,
        "font.family": "DejaVu Sans",
    })
    return plt


# Colour-blind safe palette (Okabe-Ito)
PALETTE = {
    "Persistence": "#999999",
    "Climatology": "#CC79A7",
    "Ridge": "#E69F00",
    "Gradient Boosting": "#56B4E9",
    "Bi-LSTM (MSE only)": "#009E73",
    "FloatChat Bi-LSTM": "#0072B2",
    "Observed": "#000000",
    "LIME": "#E69F00",
    "SHAP": "#009E73",
    "FloatChat IG": "#0072B2",
    "Random": "#999999",
}


def savefig(fig, name: str) -> Path:
    path = FIG_DIR / f"{name}.png"
    fig.savefig(path)
    fig.savefig(FIG_DIR / f"{name}.pdf")
    import matplotlib.pyplot as plt
    plt.close(fig)
    return path


def depth_labels():
    return [f"{int(d)}" for d in STANDARD_DEPTHS]
