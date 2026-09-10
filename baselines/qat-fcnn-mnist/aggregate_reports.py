#!/usr/bin/env python3
"""Aggregate QAT-FCNN JSON reports into the format used by paper Table 5."""

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports_dir", type=Path)
    parser.add_argument("--output", type=Path, default=Path("qat_fcnn_aggregated.csv"))
    parser.add_argument("--expected-runs", type=int, default=10)
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Write groups that do not contain the expected number of reports.",
    )
    return parser.parse_args()


def load_reports(reports_dir):
    groups = defaultdict(list)
    for path in sorted(reports_dir.glob("*.json")):
        report = json.loads(path.read_text(encoding="utf-8"))
        settings = report["settings"]
        results = report["results"]
        sparsity = int(settings["sparsity"])
        groups[sparsity].append(
            {
                "accuracy_percent": 100.0 * float(results["accuracy"]["VL full"]),
                "time_seconds": float(
                    results["timing"]["fhe_encrypt_run_decrypt_avg_seconds"]
                ),
                "bit_width": int(results["bit_widths"]["FHE short"]),
                "seed": int(settings["seed"]),
                "path": path,
            }
        )
    return groups


def bit_width_summary(runs):
    counts = Counter(run["bit_width"] for run in runs)
    total = len(runs)
    return "; ".join(
        f"{width}-bit: {100.0 * count / total:g}%" for width, count in sorted(counts.items())
    )


def main():
    args = parse_args()
    groups = load_reports(args.reports_dir)
    if not groups:
        raise SystemExit(f"No JSON reports found in {args.reports_dir}")

    rows = []
    for sparsity, runs in sorted(groups.items()):
        seeds = {run["seed"] for run in runs}
        if len(seeds) != len(runs):
            raise SystemExit(f"QAT-FCNN-{sparsity} contains duplicate seed reports")
        if len(runs) != args.expected_runs and not args.allow_incomplete:
            raise SystemExit(
                f"QAT-FCNN-{sparsity} has {len(runs)} reports; "
                f"expected {args.expected_runs}"
            )
        rows.append(
            {
                "model": f"QAT-FCNN-{sparsity}",
                "active_connections": 14 * sparsity,
                "accuracy_percent": f"{mean(run['accuracy_percent'] for run in runs):.2f}",
                "time_per_image_seconds": f"{mean(run['time_seconds'] for run in runs):.2f}",
                "circuit_bit_width": bit_width_summary(runs),
                "runs": len(runs),
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} QAT-FCNN aggregate rows to {args.output}")


if __name__ == "__main__":
    main()
