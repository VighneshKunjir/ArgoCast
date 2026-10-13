#!/usr/bin/env python
"""Stage 2 - turn the raw {WMO}_prof.nc files into one clean profile table.

Cleaning rules (in order; the counts dropped at each rule are logged):
 1. Keep profiles inside the box (10-25 N, 50-75 E) and period (2000-2026).
 2. Keep data modes D (delayed) and A (adjusted real-time) by default and use
    the *_ADJUSTED variables for both. R (raw real-time) is excluded unless
    --modes DAR is given.  Ascending profiles only (DIRECTION = 'A').
 3. Position and date QC must be 1 or 2.
 4. Mask fill values, keep level QC 1/2 for PRES, TEMP and PSAL.
 5. Hard physical bounds per level: T in [-2, 35] C, S in [30, 42] PSU.
 6. At least 10 good levels, and pressure coverage from <= 10 dbar to
    >= 950 dbar, so the 16 standard levels are interpolated, not invented.
 7. PCHIP interpolation to the 16 standard levels. A standard level that lies
    just outside the measured range (at most 50 dbar beyond the shallowest or
    deepest good level) takes the nearest measured value; anything further
    would be extrapolation and the profile is dropped. All 16 values must be
    finite and in bounds.
 8. Duplicate (WMO, cycle) pairs: keep the delayed-mode one.

The TEOS-10 static-stability flag is computed with gsw at the true position
and kept as a column (profiles are never removed for instability).

Output: data/processed/profiles_arabian_2000_2026.csv.gz
"""
from __future__ import annotations

import argparse
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from common import (END_YEAR, MAX_LAT, MAX_LON, MIN_LAT, MIN_LON, PROCESSED_DIR,
                    RAW_DIR, SAL_BOUNDS, STANDARD_DEPTHS, START_YEAR, TEMP_BOUNDS,
                    write_json)

OUT_CSV = PROCESSED_DIR / "profiles_arabian_2000_2026.csv.gz"
GOOD = {1, 2}


def _decode(arr) -> np.ndarray:
    a = np.asarray(arr)
    if a.dtype.kind == "S":
        return np.char.decode(a, "latin-1")
    if a.dtype.kind == "O":
        return np.array([x.decode("latin-1") if isinstance(x, bytes) else str(x) for x in a.ravel()]).reshape(a.shape)
    return a.astype(str)


def _qc_int(arr) -> np.ndarray:
    s = _decode(arr)
    out = np.full(s.shape, 9, dtype=np.int8)
    for v in "0123456789":
        out[s == v] = int(v)
    return out


def _interp(pres, temp, sal):
    from scipy.interpolate import PchipInterpolator
    order = np.argsort(pres)
    pres, temp, sal = pres[order], temp[order], sal[order]
    pres, uidx = np.unique(pres, return_index=True)
    temp, sal = temp[uidx], sal[uidx]
    if len(pres) < 4:
        return None, None
    ti = PchipInterpolator(pres, temp, extrapolate=False)(STANDARD_DEPTHS)
    si = PchipInterpolator(pres, sal, extrapolate=False)(STANDARD_DEPTHS)
    # Standard levels just outside the measured range (<= 50 dbar away, e.g. 5 dbar
    # when the top good level is 8 dbar) take the nearest measured value.
    for arr in (ti, si):
        top, bot = STANDARD_DEPTHS < pres[0], STANDARD_DEPTHS > pres[-1]
        arr[top & (pres[0] - STANDARD_DEPTHS <= 50)] = (temp if arr is ti else sal)[0]
        arr[bot & (STANDARD_DEPTHS - pres[-1] <= 50)] = (temp if arr is ti else sal)[-1]
    return ti, si


def process_file(path: str, modes: str, y0: int, y1: int, min_top: float, min_bottom: float):
    import gsw
    import xarray as xr
    drops = Counter()
    rows = []
    try:
        ds = xr.open_dataset(path, decode_times=True, mask_and_scale=True)
    except Exception as e:  # unreadable file
        return [], {"unreadable_file": 1}, str(e)
    try:
        n_prof = ds.sizes.get("N_PROF", 0)
        lat = ds["LATITUDE"].values.astype(float)
        lon = ds["LONGITUDE"].values.astype(float)
        juld = pd.to_datetime(ds["JULD"].values, errors="coerce")
        mode = _decode(ds["DATA_MODE"].values) if "DATA_MODE" in ds else np.array(["R"] * n_prof)
        direc = _decode(ds["DIRECTION"].values) if "DIRECTION" in ds else np.array(["A"] * n_prof)
        pos_qc = _qc_int(ds["POSITION_QC"].values) if "POSITION_QC" in ds else np.ones(n_prof, int)
        juld_qc = _qc_int(ds["JULD_QC"].values) if "JULD_QC" in ds else np.ones(n_prof, int)
        cyc = ds["CYCLE_NUMBER"].values
        wmo_arr = _decode(ds["PLATFORM_NUMBER"].values)

        def var(name):
            return ds[name].values if name in ds else None

        P, T, S = var("PRES"), var("TEMP"), var("PSAL")
        Pa, Ta, Sa = var("PRES_ADJUSTED"), var("TEMP_ADJUSTED"), var("PSAL_ADJUSTED")
        Pq, Tq, Sq = (_qc_int(var(n)) if var(n) is not None else None for n in ("PRES_QC", "TEMP_QC", "PSAL_QC"))
        Paq, Taq, Saq = (_qc_int(var(n)) if var(n) is not None else None
                         for n in ("PRES_ADJUSTED_QC", "TEMP_ADJUSTED_QC", "PSAL_ADJUSTED_QC"))

        for i in range(n_prof):
            drops["seen"] += 1
            if not (np.isfinite(lat[i]) and np.isfinite(lon[i])) or pd.isna(juld[i]):
                drops["no_position_or_date"] += 1; continue
            if not (MIN_LAT <= lat[i] <= MAX_LAT and MIN_LON <= lon[i] <= MAX_LON):
                drops["outside_box"] += 1; continue
            if not (y0 <= juld[i].year <= y1):
                drops["outside_period"] += 1; continue
            m = mode[i].strip() or "R"
            if m not in modes:
                drops[f"mode_{m}_excluded"] += 1; continue
            if direc[i].strip() == "D":
                drops["descending_profile"] += 1; continue
            if pos_qc[i] not in GOOD or juld_qc[i] not in GOOD:
                drops["bad_position_or_date_qc"] += 1; continue
            if m in "DA" and Ta is not None:
                p, t, s, pq, tq, sq = Pa[i], Ta[i], Sa[i], Paq[i], Taq[i], Saq[i]
            else:
                p, t, s, pq, tq, sq = P[i], T[i], S[i], Pq[i], Tq[i], Sq[i]
            p, t, s = (np.asarray(x, dtype=float) for x in (p, t, s))
            ok = (np.isfinite(p) & np.isfinite(t) & np.isfinite(s)
                  & np.isin(pq, list(GOOD)) & np.isin(tq, list(GOOD)) & np.isin(sq, list(GOOD))
                  & (np.abs(p) < 9000) & (np.abs(t) < 900) & (np.abs(s) < 900))
            ok &= (t >= TEMP_BOUNDS[0]) & (t <= TEMP_BOUNDS[1]) & (s >= SAL_BOUNDS[0]) & (s <= SAL_BOUNDS[1]) & (p > 0)
            p, t, s = p[ok], t[ok], s[ok]
            if len(p) < 10:
                drops["fewer_than_10_good_levels"] += 1; continue
            if p.min() > min_top or p.max() < min_bottom:
                drops["insufficient_depth_coverage"] += 1; continue
            ti, si = _interp(p, t, s)
            if ti is None or not (np.isfinite(ti).all() and np.isfinite(si).all()):
                drops["interpolation_failed"] += 1; continue
            if (ti < TEMP_BOUNDS[0]).any() or (ti > TEMP_BOUNDS[1]).any() or (si < SAL_BOUNDS[0]).any() or (si > SAL_BOUNDS[1]).any():
                drops["interpolated_out_of_bounds"] += 1; continue
            sa = gsw.SA_from_SP(si, STANDARD_DEPTHS, lon[i], lat[i])
            ct = gsw.CT_from_t(sa, ti, STANDARD_DEPTHS)
            sig = gsw.sigma0(sa, ct)
            dsig = np.diff(sig) / np.diff(STANDARD_DEPTHS)
            mld_idx = np.where(sig - sig[0] >= 0.03)[0]
            row = {
                "wmo": int(str(wmo_arr[i]).strip()), "cycle": int(cyc[i]),
                "date": juld[i].strftime("%Y-%m-%dT%H:%M:%S"),
                "latitude": float(lat[i]), "longitude": float(lon[i]),
                "data_mode": m, "n_levels_raw": int(len(p)),
                "pres_min": float(p.min()), "pres_max": float(p.max()),
                "is_stable": bool((dsig >= -1e-6).all()),
                "n_inversions": int((dsig < -1e-6).sum()),
                "mld_dbar": float(STANDARD_DEPTHS[mld_idx[0]]) if len(mld_idx) else np.nan,
            }
            for k, d in enumerate(STANDARD_DEPTHS):
                row[f"temp_{int(d)}"] = float(ti[k])
                row[f"sal_{int(d)}"] = float(si[k])
            rows.append(row)
            drops["kept"] += 1
    finally:
        ds.close()
    return rows, dict(drops), None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--modes", default="DA", help="data modes to keep: DA (default), D, or DAR")
    ap.add_argument("--start-year", type=int, default=START_YEAR)
    ap.add_argument("--end-year", type=int, default=END_YEAR)
    ap.add_argument("--min-top", type=float, default=10.0, help="shallowest good level must be <= this (dbar)")
    ap.add_argument("--min-bottom", type=float, default=950.0, help="deepest good level must be >= this (dbar)")
    ap.add_argument("--jobs", type=int, default=None, help="parallel worker processes")
    args = ap.parse_args()

    files = sorted(str(p) for p in Path(RAW_DIR).glob("*_prof.nc"))
    print(f"Raw float files: {len(files)} in {RAW_DIR}")
    if not files:
        raise SystemExit("No *_prof.nc files found. Run s01_download.py first.")
    t0 = time.time()
    all_rows, drops, errors = [], Counter(), {}
    with ProcessPoolExecutor(max_workers=args.jobs) as ex:
        futs = {ex.submit(process_file, f, args.modes, args.start_year, args.end_year,
                          args.min_top, args.min_bottom): f for f in files}
        for k, fut in enumerate(as_completed(futs), 1):
            rows, d, err = fut.result()
            all_rows.extend(rows)
            drops.update(d)
            if err:
                errors[Path(futs[fut]).name] = err
            if k % 25 == 0 or k == len(files):
                print(f"  [{k}/{len(files)}] profiles kept so far: {len(all_rows):,}")
    if not all_rows:
        raise SystemExit("No profiles survived cleaning; check the raw files.")

    df = pd.DataFrame(all_rows)
    n_before = len(df)
    df["mode_rank"] = df["data_mode"].map({"D": 0, "A": 1, "R": 2})
    df = (df.sort_values(["wmo", "cycle", "mode_rank"])
            .drop_duplicates(["wmo", "cycle"], keep="first")
            .drop(columns="mode_rank").sort_values(["wmo", "date"]).reset_index(drop=True))
    drops["duplicate_cycle_removed"] = n_before - len(df)
    drops["kept"] = len(df)
    df.to_csv(OUT_CSV, index=False, compression="gzip")

    dates = pd.to_datetime(df["date"])
    summary = {
        "n_profiles": int(len(df)), "n_floats": int(df["wmo"].nunique()),
        "date_min": str(dates.min().date()), "date_max": str(dates.max().date()),
        "modes_kept": args.modes, "mode_counts": df["data_mode"].value_counts().to_dict(),
        "stable_fraction_teos10": float(df["is_stable"].mean()),
        "profiles_per_year": dates.dt.year.value_counts().sort_index().to_dict(),
        "cleaning_counts": dict(drops), "unreadable_files": errors,
        "coverage_rule_dbar": [args.min_top, args.min_bottom],
        "elapsed_s": round(time.time() - t0, 1),
    }
    write_json(PROCESSED_DIR / "dataset_summary.json", summary)
    print(f"Saved {OUT_CSV}  ({len(df):,} profiles, {df['wmo'].nunique()} floats, "
          f"{summary['date_min']} to {summary['date_max']})")
    print("Cleaning counts:", dict(drops))


if __name__ == "__main__":
    main()
