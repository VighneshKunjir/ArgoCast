"""Benchmark study results (13 models, two test sets) for the Benchmarks tab."""

import json
from functools import lru_cache
from pathlib import Path

from fastapi import APIRouter, HTTPException

router = APIRouter()

RESULTS = Path(__file__).resolve().parents[1] / "ml" / "artifacts" / "benchmark_results.json"


@lru_cache(maxsize=1)
def _load(mtime: float) -> dict:
    return json.loads(RESULTS.read_text(encoding="utf-8"))


@router.get("/benchmark", response_model=dict)
async def get_benchmark():
    """Return the condensed benchmark record built by scripts/build_benchmark_summary.py."""
    if not RESULTS.exists():
        raise HTTPException(status_code=404, detail="benchmark_results.json not found; run scripts/build_benchmark_summary.py")
    return _load(RESULTS.stat().st_mtime)
