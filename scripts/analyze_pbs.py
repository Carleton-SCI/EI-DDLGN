#!/usr/bin/env python
"""CLI for static PBS analysis of exported LGN/DDLGN models."""

from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ei_ddlgn.pbs_analyzer import (  # noqa: E402
    DEFAULT_OUTPUT_DIR,
    analyze_many_pbs_counts,
    analyze_pbs_count,
    save_analysis_outputs,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Analyze baseline and public-constant-propagated PBS counts."
    )
    parser.add_argument(
        "--checkpoint",
        "--model",
        dest="model",
        default=None,
        help=(
            "Model directory, best_lgn_gates.csv, or checkpoint file with a "
            "sibling best_lgn_gates.csv export."
        ),
    )
    parser.add_argument(
        "--checkpoints",
        nargs="+",
        default=None,
        help=(
            "One or more model directories, CSVs, checkpoints, or glob patterns "
            "to analyze into combined CSV/JSON outputs."
        ),
    )
    parser.add_argument(
        "--metadata",
        default=None,
        help="Optional metadata JSON path. Defaults to sibling best_lgn_metadata.json.",
    )
    parser.add_argument(
        "--model-name",
        default=None,
        help="Optional model name for single-checkpoint analysis.",
    )
    parser.add_argument(
        "--dataset-name",
        default=None,
        help="Dataset name to store in the CSV/JSON outputs.",
    )
    parser.add_argument(
        "--input-status",
        default="encrypted",
        choices=["encrypted", "PUBLIC_CONST_0", "PUBLIC_CONST_1", "0", "1"],
        help="Status assigned to every input wire.",
    )
    parser.add_argument(
        "--pbs-latency-ms",
        type=float,
        default=None,
        help="Optional per-PBS latency used for before/after estimates.",
    )
    parser.add_argument(
        "--save-csv",
        default=None,
        help="Optional additional per-layer CSV path for single-checkpoint analysis.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory for combined per-layer CSV, summary CSV, and JSON results.",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append to existing CSV/JSON files instead of replacing this run's outputs.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Do not print the analysis table.",
    )
    args = parser.parse_args()

    if args.checkpoints:
        checkpoints = _expand_checkpoint_patterns(args.checkpoints)
        if not checkpoints:
            raise SystemExit("No checkpoints matched --checkpoints patterns.")
        analyze_many_pbs_counts(
            checkpoints,
            input_status=args.input_status,
            pbs_latency_ms=args.pbs_latency_ms,
            verbose=not args.quiet,
            metadata=args.metadata,
            dataset_name=args.dataset_name,
            output_dir=args.output_dir,
            append=args.append,
        )
    else:
        if args.model is None:
            raise SystemExit("Provide --checkpoint/--model or --checkpoints.")
        result = analyze_pbs_count(
            args.model,
            input_status=args.input_status,
            pbs_latency_ms=args.pbs_latency_ms,
            save_csv=args.save_csv,
            verbose=not args.quiet,
            metadata=args.metadata,
            model_name=args.model_name,
            dataset_name=args.dataset_name,
        )
        save_analysis_outputs([result], output_dir=args.output_dir, append=args.append)
    return 0


def _expand_checkpoint_patterns(patterns: list[str]) -> list[str]:
    matches: list[str] = []
    for pattern in patterns:
        expanded = glob.glob(pattern)
        matches.extend(expanded if expanded else [pattern])
    return sorted(dict.fromkeys(matches))


if __name__ == "__main__":
    raise SystemExit(main())
