#!/usr/bin/env python
"""Stage 1 - download every Argo float that profiled the Arabian Sea, 2000-2026.

How it works
1. Fetch the GDAC profile index (ar_index_global_prof.txt.gz) and keep every
   profile whose position lies inside the box and whose date is in the period.
2. Collect the set of floats behind those profiles (any data mode).
3. Download one multi-profile file per float ({WMO}_prof.nc). This is far
   faster than fetching thousands of single-cycle files; profiles outside the
   box/period are removed later in stage 2.

Downloads are resumable (finished files are skipped, partial files are
written as .part and renamed only when complete). Each file is tried on the
IFREMER mirror first, then US GODAE.

Usage
    python s01_download.py                      # all floats, 2000-2026
    python s01_download.py --workers 6
    python s01_download.py --mirror https://usgodae.org/pub/outgoing/argo
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import gzip
import os
import time
from collections import defaultdict
from pathlib import Path

import httpx

from common import (MAX_LAT, MAX_LON, MIN_LAT, MIN_LON, MIRRORS, RAW_DIR,
                    START_YEAR, END_YEAR, DATA_DIR, write_json)

INDEX_NAME = "ar_index_global_prof.txt.gz"


def fetch_index(mirrors: list[str], dest: Path, refresh: bool) -> Path:
    if dest.exists() and dest.stat().st_size > 0 and not refresh:
        print(f"Using cached index {dest} ({dest.stat().st_size/1e6:.1f} MB)")
        return dest
    for m in mirrors:
        url = f"{m}/{INDEX_NAME}"
        print(f"Downloading index from {url}")
        try:
            tmp = dest.with_suffix(".part")
            with httpx.stream("GET", url, timeout=httpx.Timeout(600, connect=30),
                              follow_redirects=True) as r:
                r.raise_for_status()
                with open(tmp, "wb") as f:
                    for chunk in r.iter_bytes(1 << 20):
                        f.write(chunk)
            tmp.replace(dest)
            print(f"  saved {dest.stat().st_size/1e6:.1f} MB")
            return dest
        except Exception as e:  # try next mirror
            print(f"  failed: {e}")
    raise SystemExit("Could not download the Argo index from any mirror.")


def parse_index(path: Path, y0: int, y1: int):
    """Return {wmo: info} for floats with >=1 profile in the box and period."""
    floats: dict = {}
    n_lines = n_in = 0
    with gzip.open(path, "rt", encoding="latin-1") as fh:
        rows = (ln for ln in fh if not ln.startswith("#"))
        reader = csv.reader(rows)
        header = next(reader)
        col = {name: i for i, name in enumerate(header)}
        for row in reader:
            n_lines += 1
            try:
                fname, date = row[col["file"]], row[col["date"]]
                lat, lon = float(row[col["latitude"]]), float(row[col["longitude"]])
            except (ValueError, IndexError, KeyError):
                continue
            if not (MIN_LAT <= lat <= MAX_LAT and MIN_LON <= lon <= MAX_LON):
                continue
            if len(date) < 8 or not (y0 <= int(date[:4]) <= y1):
                continue
            parts = fname.split("/")
            if len(parts) < 4:
                continue
            dac, wmo, base = parts[0], parts[1], parts[-1]
            n_in += 1
            f = floats.setdefault(wmo, {"dac": dac, "n_profiles": 0, "n_D": 0, "n_R": 0,
                                        "first": date[:8], "last": date[:8],
                                        "lat_sum": 0.0, "lon_sum": 0.0})
            f["n_profiles"] += 1
            f["n_D" if base.startswith("D") else "n_R"] += 1
            f["first"] = min(f["first"], date[:8])
            f["last"] = max(f["last"], date[:8])
            f["lat_sum"] += lat
            f["lon_sum"] += lon
    print(f"Index lines: {n_lines:,} | profiles in box & period: {n_in:,} | floats: {len(floats)}")
    return floats


def _safe_unlink(p: Path) -> None:
    """Delete a file, ignoring errors (Windows may hold a lock for a moment)."""
    try:
        p.unlink()
    except OSError:
        pass


async def _safe_replace(src: Path, dst: Path, tries: int = 20) -> bool:
    """Rename src -> dst, retrying while antivirus / indexer briefly locks the file."""
    for i in range(tries):
        try:
            os.replace(src, dst)
            return True
        except PermissionError:
            await asyncio.sleep(0.5 * (i + 1))
    return False


async def download_one(client, sem, wmo, dac, mirrors, out_dir, stats):
    dest = out_dir / f"{wmo}_prof.nc"
    if dest.exists() and dest.stat().st_size > 0:
        stats["skipped"] += 1
        return wmo, "cached"
    async with sem:
        for attempt in range(3):
            for m in mirrors:
                url = f"{m}/dac/{dac}/{wmo}/{wmo}_prof.nc"
                tmp = out_dir / f"{wmo}_prof.nc.part{attempt}"  # new name per attempt avoids a locked leftover
                try:
                    async with client.stream("GET", url) as r:
                        if r.status_code == 404:
                            continue
                        r.raise_for_status()
                        with open(tmp, "wb") as f:
                            async for chunk in r.aiter_bytes(1 << 16):
                                f.write(chunk)
                    if await _safe_replace(tmp, dest):
                        stats["ok"] += 1
                        return wmo, "ok"
                    _safe_unlink(tmp)
                except Exception:
                    _safe_unlink(tmp)
            await asyncio.sleep(2 * (attempt + 1))
    stats["failed"] += 1
    return wmo, "failed"


async def download_all(floats: dict, mirrors, out_dir: Path, workers: int):
    sem = asyncio.Semaphore(workers)
    stats = defaultdict(int)
    limits = httpx.Limits(max_connections=workers * 2)
    timeout = httpx.Timeout(300, connect=30)
    results = {}
    async with httpx.AsyncClient(timeout=timeout, limits=limits, follow_redirects=True) as client:
        tasks = [download_one(client, sem, w, info["dac"], mirrors, out_dir, stats)
                 for w, info in floats.items()]
        t0 = time.time()
        for i, coro in enumerate(asyncio.as_completed(tasks), 1):
            try:
                wmo, status = await coro
                results[wmo] = status
            except Exception as e:  # never let one file stop the whole download
                stats["failed"] += 1
                print(f"  unexpected error: {e}")
            if i % 10 == 0 or i == len(tasks):
                print(f"  [{i}/{len(tasks)}] ok={stats['ok']} cached={stats['skipped']} "
                      f"failed={stats['failed']}  {time.time()-t0:.0f}s")
    return results


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start-year", type=int, default=START_YEAR)
    ap.add_argument("--end-year", type=int, default=END_YEAR)
    ap.add_argument("--workers", type=int, default=4, help="parallel downloads (be polite: 4-8)")
    ap.add_argument("--mirror", action="append", help="override mirror list (repeatable)")
    ap.add_argument("--refresh-index", action="store_true")
    ap.add_argument("--max-floats", type=int, default=None, help="debug: limit number of floats")
    args = ap.parse_args()

    mirrors = args.mirror or MIRRORS
    idx = fetch_index(mirrors, DATA_DIR / INDEX_NAME, args.refresh_index)
    floats = parse_index(idx, args.start_year, args.end_year)
    if args.max_floats:
        floats = dict(sorted(floats.items(), key=lambda kv: -kv[1]["n_profiles"])[: args.max_floats])

    manifest = DATA_DIR / "float_manifest.csv"
    with open(manifest, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["wmo", "dac", "n_profiles_in_box", "n_delayed", "n_realtime",
                    "first_date", "last_date", "mean_lat", "mean_lon"])
        for wmo, i in sorted(floats.items()):
            w.writerow([wmo, i["dac"], i["n_profiles"], i["n_D"], i["n_R"], i["first"], i["last"],
                        round(i["lat_sum"] / i["n_profiles"], 3), round(i["lon_sum"] / i["n_profiles"], 3)])
    print(f"Manifest: {manifest}")

    for leftover in RAW_DIR.glob("*.part*"):  # partial files from an interrupted run
        _safe_unlink(leftover)
    print(f"Downloading {len(floats)} float files to {RAW_DIR} with {args.workers} workers")
    status = asyncio.run(download_all(floats, mirrors, RAW_DIR, args.workers))
    failed = sorted(w for w in floats if status.get(w) not in ("ok", "cached"))
    write_json(DATA_DIR / "download_log.json", {
        "box": [MIN_LAT, MAX_LAT, MIN_LON, MAX_LON],
        "years": [args.start_year, args.end_year],
        "mirrors": mirrors,
        "n_floats_in_index": len(floats),
        "n_profiles_in_index": sum(i["n_profiles"] for i in floats.values()),
        "n_failed": len(failed), "failed_wmos": failed,
        "finished": time.strftime("%Y-%m-%d %H:%M:%S"),
    })
    print(f"Done. Failed: {len(failed)}" + (f" -> rerun the script to retry {failed[:10]}" if failed else ""))


if __name__ == "__main__":
    main()
