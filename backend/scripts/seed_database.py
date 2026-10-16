#!/usr/bin/env python
"""Load the Arabian Sea 2000-2026 Argo dataset into the FloatChat database.

Source: backend/data/processed/argo_arabian_2000_2026.csv.gz
  one row per quality-controlled profile (321 floats, 36,164 profiles), already
  interpolated to the 16 standard levels (5-1000 dbar). It is the table written by
  research/pipeline/s02_build_dataset.py (plus the GDAC data centre of each float),
  the same data the forecasting model was trained and evaluated on.

The script is idempotent: floats and profiles already in the database are skipped.
Use --reset to drop and rebuild the Argo tables (forecast logs are kept).

Usage (from backend/):
  python scripts/seed_database.py            # full dataset
  python scripts/seed_database.py --reset
  python scripts/seed_database.py --floats 6  # small subset (used by the tests)
"""
import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import gsw
import numpy as np
import pandas as pd

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))

from app.db.session import SessionLocal, engine, init_db  # noqa: E402
from app.models.schema import ArgoFloat, ArgoProfile, ProfileLevel  # noqa: E402

DEPTHS = [5, 20, 50, 75, 100, 150, 200, 250, 300, 400, 500, 600, 700, 800, 900, 1000]
DATA_FILE = BACKEND / "data" / "processed" / "argo_arabian_2000_2026.csv.gz"


def profile_id(wmo: int, cycle: int) -> str:
    return f"ARGO_{wmo}_CYC{cycle:03d}"


def potential_density(df: pd.DataFrame) -> np.ndarray:
    """TEOS-10 potential density anomaly sigma0 (kg m-3) for every profile and level."""
    t = df[[f"temp_{d}" for d in DEPTHS]].to_numpy(float)
    s = df[[f"sal_{d}" for d in DEPTHS]].to_numpy(float)
    p = np.broadcast_to(np.array(DEPTHS, float), t.shape)
    lon = df["longitude"].to_numpy(float)[:, None]
    lat = df["latitude"].to_numpy(float)[:, None]
    sa = gsw.SA_from_SP(s, p, lon, lat)
    return gsw.sigma0(sa, gsw.CT_from_t(sa, t, p))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", type=Path, default=DATA_FILE)
    ap.add_argument("--reset", action="store_true", help="drop and recreate the float/profile/level tables first")
    ap.add_argument("--floats", type=int, default=None,
                    help="only load the N floats with the most profiles (quick test database)")
    args = ap.parse_args()

    if not args.csv.exists():
        sys.exit(f"Dataset not found: {args.csv}")
    t0 = time.time()
    df = pd.read_csv(args.csv)
    df["date"] = pd.to_datetime(df["date"])
    if "dac" not in df.columns:
        # Raw pipeline output (research/pipeline/s02_build_dataset.py) has no DAC column;
        # take it from the download manifest when available.
        manifest = next((p for p in (args.csv.parent / "float_manifest.csv",
                                     args.csv.parent.parent / "float_manifest.csv") if p.exists()), None)
        dac = pd.read_csv(manifest, usecols=["wmo", "dac"]) if manifest else pd.DataFrame(columns=["wmo", "dac"])
        df = df.merge(dac, on="wmo", how="left")
        df["dac"] = df["dac"].fillna("unknown")
    if args.floats:
        keep = df["wmo"].value_counts().head(args.floats).index
        df = df[df["wmo"].isin(keep)]
    df = df.sort_values(["wmo", "cycle"]).reset_index(drop=True)

    if args.reset:
        for table in (ProfileLevel.__table__, ArgoProfile.__table__, ArgoFloat.__table__):
            table.drop(engine, checkfirst=True)
    init_db()

    db = SessionLocal()
    try:
        have_floats = {w for (w,) in db.query(ArgoFloat.wmo_id).all()}
        have_profiles = {p for (p,) in db.query(ArgoProfile.profile_id).all()}
        df["profile_id"] = [profile_id(w, c) for w, c in zip(df["wmo"], df["cycle"])]
        new = df[~df["profile_id"].isin(have_profiles)].reset_index(drop=True)
        now = datetime.now(timezone.utc).replace(tzinfo=None)

        floats = []
        for wmo, g in df.groupby("wmo"):
            if str(wmo) in have_floats:
                continue
            floats.append({
                "wmo_id": str(wmo),
                "name": f"Float {wmo} (Arabian Sea)",
                "base_lat": float(g["latitude"].iloc[-1]),   # latest known position
                "base_lon": float(g["longitude"].iloc[-1]),
                "institution": f"Argo GDAC ({str(g['dac'].iloc[0]).upper()})",
                "status": "ACTIVE" if g["date"].max() >= pd.Timestamp("2025-01-01") else "INACTIVE",
                "created_at": now,
            })
        if floats:
            db.execute(ArgoFloat.__table__.insert(), floats)

        if len(new):
            sig = potential_density(new)
            profiles, levels = [], []
            for i, r in enumerate(new.itertuples(index=False)):
                prefix = "D" if r.data_mode == "D" else "R"  # adjusted real-time files keep the R prefix
                fname = f"{prefix}{r.wmo}_{r.cycle:03d}.nc"
                profiles.append({
                    "profile_id": r.profile_id,
                    "wmo_id": str(r.wmo),
                    "cycle_number": int(r.cycle),
                    "date": r.date.date(),
                    "latitude": float(r.latitude),
                    "longitude": float(r.longitude),
                    "qc_status": "QC_PASS_FLAG_1" if r.data_mode == "D" else "QC_PASS_FLAG_2",
                    "raw_netcdf_source": fname,
                    "gdac_archive_path": f"dac/{r.dac}/{r.wmo}/profiles/{fname}",
                    "created_at": now,
                })
                for k, d in enumerate(DEPTHS):
                    levels.append({
                        "profile_id": r.profile_id,
                        "depth_dbar": d,
                        "temperature": float(getattr(r, f"temp_{d}")),
                        "salinity": float(getattr(r, f"sal_{d}")),
                        "potential_density": float(sig[i, k]),
                    })
            for start in range(0, len(profiles), 5000):
                db.execute(ArgoProfile.__table__.insert(), profiles[start:start + 5000])
            for start in range(0, len(levels), 50000):
                db.execute(ProfileLevel.__table__.insert(), levels[start:start + 50000])
        db.commit()

        print(f"Added {len(floats)} floats and {len(new)} profiles ({len(new) * len(DEPTHS)} levels) "
              f"in {time.time() - t0:.1f} s")
        print(f"Database totals: {db.query(ArgoFloat).count()} floats, "
              f"{db.query(ArgoProfile).count()} profiles, {db.query(ProfileLevel).count()} levels")
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


if __name__ == "__main__":
    main()
