#!/usr/bin/env python
"""Run the data pipeline and evaluation: download -> clean -> split -> train/evaluate -> XAI comparison.

    python run_all.py                 # full run (default settings)
    python run_all.py --skip-download # data already in data/raw
    python run_all.py --quick         # fast smoke test of every stage (1 seed, few epochs)
    python run_all.py --from 4        # restart from stage 4
    python run_all.py --from 4 --seeds 42 --splits chronological
                                      # any extra options are passed to stage 4 (training)

Stage numbers: 1 download, 2 build dataset, 3 split, 4 train/evaluate, 5 XAI comparison.
The 13-model benchmark (research/benchmark/run_benchmark.py) runs on the output of stage 3.
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent

FULL = {
    1: ["s01_download.py"],
    2: ["s02_build_dataset.py"],
    3: ["s03_split.py"],
    4: ["s04_train_evaluate.py"],
    5: ["s05_xai_compare.py"],
}
QUICK_EXTRA = {
    4: ["--seeds", "42", "--epochs", "3", "--patience", "2", "--mc-passes", "5", "--bootstrap", "200", "--gb-iter", "20"],
    5: ["--n", "8", "--stab-n", "3", "--matrix-n", "2", "--shap-nsamples", "200", "--lime-samples", "300", "--ig-steps", "16"],
}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--skip-download", action="store_true")
    ap.add_argument("--from", dest="start", type=int, default=1)
    ap.add_argument("--to", dest="end", type=int, default=5)
    args, passthrough = ap.parse_known_args()
    t0 = time.time()
    for stage in range(args.start, args.end + 1):
        if stage == 1 and args.skip_download:
            continue
        cmd = [sys.executable, str(HERE / FULL[stage][0])] + FULL[stage][1:]
        if args.quick:
            cmd += QUICK_EXTRA.get(stage, [])
        if stage == 4:
            cmd += passthrough
        print(f"\n{'=' * 70}\nSTAGE {stage}: {' '.join(cmd[1:])}\n{'=' * 70}", flush=True)
        r = subprocess.run(cmd, cwd=HERE)
        if r.returncode != 0:
            sys.exit(f"Stage {stage} failed (exit {r.returncode}). Fix the error and rerun with --from {stage}.")
    print(f"\nFinished stages {args.start}-{args.end} in {(time.time() - t0) / 60:.1f} min.")
    sys.path.insert(0, str(HERE))
    from common import OUT_DIR
    print(f"Results: {OUT_DIR / 'tables'}")


if __name__ == "__main__":
    main()
