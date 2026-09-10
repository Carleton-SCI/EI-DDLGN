#!/usr/bin/env python3
"""Run the six-configuration, ten-seed QAT-FCNN paper protocol."""

import argparse
import subprocess
import sys
from pathlib import Path


PAPER_SPARSITIES = (4, 6, 8, 10, 11, 12)
ROOT = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        required=True,
        help="Exactly ten independent training seeds used for the aggregate experiment.",
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--fhe-samples", type=int, default=5)
    parser.add_argument("--reports-dir", type=Path, default=ROOT / "experiment_reports")
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without running the expensive training/FHE sweep.",
    )
    args = parser.parse_args()
    if len(args.seeds) != 10 or len(set(args.seeds)) != 10:
        parser.error("--seeds must contain exactly ten distinct integers")
    return args


def main():
    args = parse_args()
    for sparsity in PAPER_SPARSITIES:
        for seed in args.seeds:
            command = [
                sys.executable,
                str(ROOT / "mnist_in_fhe.py"),
                "--sparsity",
                str(sparsity),
                "--seed",
                str(seed),
                "--epochs",
                str(args.epochs),
                "--fhe-samples",
                str(args.fhe_samples),
                "--reports-dir",
                str(args.reports_dir),
            ]
            if args.cuda:
                command.append("--cuda")
            print(" ".join(command), flush=True)
            if not args.dry_run:
                subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
