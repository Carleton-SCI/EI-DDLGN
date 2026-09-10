"""Static PBS-cost analyzer for discretized 2-input logic-gate networks.

The exported LGN/DDLGN model format used in this repository stores each
discretized gate as a 4-bit truth-table id. The bit order is:

    (a, b) = 00, 01, 10, 11

For example, gate id 1 is ``0001`` (AND), gate id 3 is ``0011`` (a), and
gate id 15 is ``1111`` (constant one).
"""

from __future__ import annotations

import csv
import json
import math
import re
import warnings
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


class SignalStatus(str, Enum):
    PUBLIC_CONST_0 = "PUBLIC_CONST_0"
    PUBLIC_CONST_1 = "PUBLIC_CONST_1"
    ENCRYPTED = "ENCRYPTED"


@dataclass(frozen=True)
class CostModel:
    encrypted_encrypted_non_unary_pbs: float = 1
    mixed_public_encrypted_pbs: float = 0
    public_public_pbs: float = 0
    unary_or_constant_pbs: float = 0


@dataclass(frozen=True)
class GateSpec:
    layer: int
    neuron: int
    input_a: int
    input_b: int
    gate: int | Sequence[int] | str
    gate_name: str | None = None


@dataclass(frozen=True)
class GateAnalysis:
    pbs: float
    output_status: SignalStatus
    truth_table: tuple[int, int, int, int]


GATE_NAME_TO_ID = {
    "zero": 0,
    "0": 0,
    "false": 0,
    "and": 1,
    "notimplies": 2,
    "not_implies": 2,
    "aandnotb": 2,
    "a": 3,
    "notimpliedby": 4,
    "not_implied_by": 4,
    "notaandb": 4,
    "b": 5,
    "xor": 6,
    "or": 7,
    "notor": 8,
    "not_or": 8,
    "nor": 8,
    "notxor": 9,
    "not_xor": 9,
    "xnor": 9,
    "notb": 10,
    "not_b": 10,
    "impliedby": 11,
    "implied_by": 11,
    "not_a": 12,
    "nota": 12,
    "implies": 13,
    "notand": 14,
    "not_and": 14,
    "nand": 14,
    "one": 15,
    "1": 15,
    "true": 15,
}

GATE_ID_TO_NAME = {
    0: "zero",
    1: "and",
    2: "not_implies",
    3: "a",
    4: "not_implied_by",
    5: "b",
    6: "xor",
    7: "or",
    8: "not_or",
    9: "not_xor",
    10: "not_b",
    11: "implied_by",
    12: "not_a",
    13: "implies",
    14: "not_and",
    15: "one",
}

CHEAP_ENCRYPTED_ENCRYPTED_GATE_IDS = {0, 3, 5, 10, 12, 15}

DEFAULT_OUTPUT_DIR = Path("artifacts/analysis-generated")
DEFAULT_PER_LAYER_CSV = "pbs_analysis_per_layer.csv"
DEFAULT_SUMMARY_CSV = "pbs_analysis_summary.csv"
DEFAULT_RESULTS_JSON = "pbs_analysis_results.json"

PER_LAYER_FIELDNAMES = [
    "model_name",
    "checkpoint_path",
    "dataset_name",
    "depth",
    "width",
    "layer_index",
    "num_gates",
    "baseline_pbs",
    "optimized_pbs",
    "pbs_saved",
    "pbs_reduction_percent",
    "const0_outputs",
    "const1_outputs",
    "public_const_outputs",
    "encrypted_outputs",
    "public_const_output_percent",
    "encrypted_output_percent",
    "estimated_baseline_latency_ms",
    "estimated_optimized_latency_ms",
    "estimated_latency_saved_ms",
    "estimated_latency_reduction_percent",
]

SUMMARY_FIELDNAMES = [
    "model_name",
    "checkpoint_path",
    "dataset_name",
    "depth",
    "width",
    "best_epoch",
    "best_val_accuracy",
    "test_accuracy_python_discrete",
    "test_accuracy_percent",
    "num_layers",
    "total_gates",
    "total_baseline_pbs",
    "total_optimized_pbs",
    "total_pbs_saved",
    "total_pbs_reduction_percent",
    "total_const0_outputs",
    "total_const1_outputs",
    "total_public_const_outputs",
    "total_encrypted_outputs",
    "final_layer_const0_outputs",
    "final_layer_const1_outputs",
    "final_layer_public_const_outputs",
    "final_layer_encrypted_outputs",
    "estimated_total_baseline_latency_ms",
    "estimated_total_optimized_latency_ms",
    "estimated_total_latency_saved_ms",
    "estimated_total_latency_reduction_percent",
]


def normalize_gate_name(name: str) -> str:
    normalized = str(name).strip().lower()
    for ch in (" ", "-", "(", ")", ".", ","):
        normalized = normalized.replace(ch, "")
    return normalized


def truth_table_from_gate(gate: int | str | Sequence[int]) -> tuple[int, int, int, int]:
    """Return a truth table in [f00, f01, f10, f11] order."""
    if isinstance(gate, bool):
        gate = int(gate)

    if isinstance(gate, int):
        if gate < 0 or gate > 15:
            raise ValueError(f"Gate integer must be in [0, 15], got {gate}")
        return tuple(int(bit) for bit in f"{gate:04b}")  # type: ignore[return-value]

    if isinstance(gate, str):
        key = normalize_gate_name(gate)
        if key not in GATE_NAME_TO_ID:
            raise ValueError(f"Unknown gate name: {gate!r}")
        return truth_table_from_gate(GATE_NAME_TO_ID[key])

    values = tuple(int(v) for v in gate)
    if len(values) != 4 or any(v not in (0, 1) for v in values):
        raise ValueError(
            "Gate truth-table vectors must contain exactly four 0/1 values "
            "in [f00, f01, f10, f11] order"
        )
    return values


def truth_table_to_gate_id(truth_table: Sequence[int]) -> int:
    tt = truth_table_from_gate(truth_table)
    return int("".join(str(v) for v in tt), 2)


def coerce_signal_status(value: Any) -> SignalStatus:
    if isinstance(value, SignalStatus):
        return value
    if isinstance(value, bool):
        return SignalStatus.PUBLIC_CONST_1 if value else SignalStatus.PUBLIC_CONST_0
    if isinstance(value, int) and value in (0, 1):
        return SignalStatus.PUBLIC_CONST_1 if value else SignalStatus.PUBLIC_CONST_0

    text = str(value).strip().upper()
    aliases = {
        "0": SignalStatus.PUBLIC_CONST_0,
        "FALSE": SignalStatus.PUBLIC_CONST_0,
        "CONST0": SignalStatus.PUBLIC_CONST_0,
        "PUBLIC_CONST_0": SignalStatus.PUBLIC_CONST_0,
        "PUBLIC0": SignalStatus.PUBLIC_CONST_0,
        "1": SignalStatus.PUBLIC_CONST_1,
        "TRUE": SignalStatus.PUBLIC_CONST_1,
        "CONST1": SignalStatus.PUBLIC_CONST_1,
        "PUBLIC_CONST_1": SignalStatus.PUBLIC_CONST_1,
        "PUBLIC1": SignalStatus.PUBLIC_CONST_1,
        "ENCRYPTED": SignalStatus.ENCRYPTED,
        "ENC": SignalStatus.ENCRYPTED,
    }
    if text not in aliases:
        raise ValueError(f"Unknown signal status: {value!r}")
    return aliases[text]


def public_const_value(status: SignalStatus) -> int | None:
    if status == SignalStatus.PUBLIC_CONST_0:
        return 0
    if status == SignalStatus.PUBLIC_CONST_1:
        return 1
    return None


def classify_unary_outputs(g0: int, g1: int) -> SignalStatus:
    """Classify a unary function represented by [g(0), g(1)]."""
    if (g0, g1) == (0, 0):
        return SignalStatus.PUBLIC_CONST_0
    if (g0, g1) == (1, 1):
        return SignalStatus.PUBLIC_CONST_1
    if (g0, g1) in ((0, 1), (1, 0)):
        return SignalStatus.ENCRYPTED
    raise ValueError(f"Unary outputs must be bits, got {(g0, g1)!r}")


def analyze_lut2_gate(
    gate: int | str | Sequence[int],
    input_a_status: SignalStatus | str | bool | int,
    input_b_status: SignalStatus | str | bool | int,
    cost_model: CostModel | None = None,
) -> GateAnalysis:
    """Analyze one LUT2 gate under the public-constant propagation model."""
    cost_model = cost_model or CostModel()
    tt = truth_table_from_gate(gate)
    a_status = coerce_signal_status(input_a_status)
    b_status = coerce_signal_status(input_b_status)
    ca = public_const_value(a_status)
    cb = public_const_value(b_status)

    if ca is not None and cb is not None:
        out = tt[2 * ca + cb]
        status = SignalStatus.PUBLIC_CONST_1 if out else SignalStatus.PUBLIC_CONST_0
        return GateAnalysis(cost_model.public_public_pbs, status, tt)

    if ca is None and cb is not None:
        # With b public, the LUT2 collapses to the unary function g(a):
        # g(0) = f(0, cb), g(1) = f(1, cb). Under the default Boolean FHE
        # model, constants, identity, and NOT need no PBS.
        g0 = tt[cb]
        g1 = tt[2 + cb]
        return GateAnalysis(
            cost_model.mixed_public_encrypted_pbs,
            classify_unary_outputs(g0, g1),
            tt,
        )

    if ca is not None and cb is None:
        # Symmetric mixed case: with a public, the LUT2 collapses to g(b).
        g0 = tt[2 * ca]
        g1 = tt[2 * ca + 1]
        return GateAnalysis(
            cost_model.mixed_public_encrypted_pbs,
            classify_unary_outputs(g0, g1),
            tt,
        )

    gate_id = truth_table_to_gate_id(tt)
    if gate_id in CHEAP_ENCRYPTED_ENCRYPTED_GATE_IDS:
        if gate_id == 0:
            status = SignalStatus.PUBLIC_CONST_0
        elif gate_id == 15:
            status = SignalStatus.PUBLIC_CONST_1
        else:
            status = SignalStatus.ENCRYPTED
        return GateAnalysis(cost_model.unary_or_constant_pbs, status, tt)

    return GateAnalysis(
        cost_model.encrypted_encrypted_non_unary_pbs,
        SignalStatus.ENCRYPTED,
        tt,
    )


def baseline_gate_pbs(
    gate: int | str | Sequence[int], cost_model: CostModel | None = None
) -> float:
    analysis = analyze_lut2_gate(
        gate,
        SignalStatus.ENCRYPTED,
        SignalStatus.ENCRYPTED,
        cost_model=cost_model,
    )
    return analysis.pbs


def analyze_pbs_count(
    model: Any,
    input_status: str | SignalStatus | Sequence[str | SignalStatus | bool | int] = "encrypted",
    pbs_latency_ms: float | None = None,
    save_csv: str | Path | None = None,
    verbose: bool = True,
    cost_model: CostModel | None = None,
    metadata: str | Path | Mapping[str, Any] | None = None,
    model_name: str | None = None,
    checkpoint_path: str | Path | None = None,
    dataset_name: str | None = None,
    output_dir: str | Path | None = None,
    append: bool = False,
) -> dict[str, Any]:
    """Analyze PBS counts for an exported or in-memory LGN/DDLGN model.

    ``model`` may be one of:
    - a model directory containing ``best_lgn_gates.csv`` and metadata,
    - a path to a gates CSV,
    - a sequence of layers containing ``GateSpec`` objects, mappings, or tuples,
    - a PyTorch ``nn.Sequential``/iterable containing ``LogicLayer`` objects
      with ``weights`` and ``indices`` attributes.
    """
    cost_model = cost_model or CostModel()
    layers, source_info = load_gate_layers(model, metadata=metadata)

    if not layers:
        raise ValueError("No logic-gate layers were found to analyze")

    source_info = dict(source_info)
    resolved_checkpoint_path = _resolve_checkpoint_path(
        checkpoint_path=checkpoint_path,
        model=model,
        source_info=source_info,
    )
    source_info["checkpoint_path"] = resolved_checkpoint_path

    metadata_obj = source_info.get("metadata")
    resolved_model_name = model_name or _infer_model_name(model, source_info)
    resolved_dataset_name = dataset_name or _infer_dataset_name(
        metadata_obj,
        resolved_checkpoint_path,
    )
    model_metadata = _extract_model_metadata(metadata_obj, resolved_checkpoint_path)

    input_count = _infer_input_count(layers, source_info.get("metadata"))
    previous_statuses = _expand_input_status(input_status, input_count)

    layer_stats: list[dict[str, Any]] = []
    total_gates = 0
    total_baseline = 0.0
    total_optimized = 0.0
    total_const0_outputs = 0
    total_const1_outputs = 0
    total_encrypted_outputs = 0

    for layer in layers:
        output_statuses: list[SignalStatus] = []
        layer_baseline = 0.0
        layer_optimized = 0.0

        for gate in layer["gates"]:
            if gate.input_a >= len(previous_statuses) or gate.input_b >= len(previous_statuses):
                raise IndexError(
                    f"Layer {gate.layer} neuron {gate.neuron} references inputs "
                    f"({gate.input_a}, {gate.input_b}), but previous layer has "
                    f"{len(previous_statuses)} wires"
                )

            gate_ref = gate.gate
            layer_baseline += baseline_gate_pbs(gate_ref, cost_model=cost_model)
            optimized = analyze_lut2_gate(
                gate_ref,
                previous_statuses[gate.input_a],
                previous_statuses[gate.input_b],
                cost_model=cost_model,
            )
            layer_optimized += optimized.pbs
            output_statuses.append(optimized.output_status)

        const0_count = output_statuses.count(SignalStatus.PUBLIC_CONST_0)
        const1_count = output_statuses.count(SignalStatus.PUBLIC_CONST_1)
        encrypted_count = output_statuses.count(SignalStatus.ENCRYPTED)
        public_count = const0_count + const1_count
        saved = layer_baseline - layer_optimized

        stat = {
            "layer": layer["label"],
            "gates": len(layer["gates"]),
            "baseline_pbs": _number_for_display(layer_baseline),
            "optimized_pbs": _number_for_display(layer_optimized),
            "saved": _number_for_display(saved),
            "const0_out": const0_count,
            "const1_out": const1_count,
            "public_const_out": public_count,
            "encrypted_out": encrypted_count,
            "public_const_output_percent": _percent(public_count, len(layer["gates"])),
            "encrypted_output_percent": _percent(encrypted_count, len(layer["gates"])),
            "pbs_reduction_percent": _percent(saved, layer_baseline),
        }
        if pbs_latency_ms is not None:
            stat.update(
                {
                    "estimated_baseline_latency_ms": layer_baseline * pbs_latency_ms,
                    "estimated_optimized_latency_ms": layer_optimized * pbs_latency_ms,
                    "estimated_latency_saved_ms": saved * pbs_latency_ms,
                    "estimated_latency_reduction_percent": _percent(saved, layer_baseline),
                }
            )
        layer_stats.append(stat)

        total_gates += len(layer["gates"])
        total_baseline += layer_baseline
        total_optimized += layer_optimized
        total_const0_outputs += const0_count
        total_const1_outputs += const1_count
        total_encrypted_outputs += encrypted_count
        previous_statuses = output_statuses

    total_saved = total_baseline - total_optimized
    total_public_outputs = total_const0_outputs + total_const1_outputs
    final_const0 = previous_statuses.count(SignalStatus.PUBLIC_CONST_0)
    final_const1 = previous_statuses.count(SignalStatus.PUBLIC_CONST_1)
    final_public = final_const0 + final_const1
    final_outputs = len(previous_statuses)
    totals = {
        "total_gates": total_gates,
        "total_baseline_pbs": _number_for_display(total_baseline),
        "total_optimized_pbs": _number_for_display(total_optimized),
        "total_saved": _number_for_display(total_saved),
        "pbs_saving_percent": _percent(total_saved, total_baseline),
        "total_const0_outputs": total_const0_outputs,
        "total_const1_outputs": total_const1_outputs,
        "total_public_const_outputs": total_public_outputs,
        "total_encrypted_outputs": total_encrypted_outputs,
        "final_const0_outputs": final_const0,
        "final_const1_outputs": final_const1,
        "final_public_const_outputs": final_public,
        "final_public_const_percent": _percent(final_public, final_outputs),
        "final_encrypted_outputs": previous_statuses.count(SignalStatus.ENCRYPTED),
    }

    if pbs_latency_ms is not None:
        totals.update(
            {
                "pbs_latency_ms": pbs_latency_ms,
                "baseline_latency_ms": total_baseline * pbs_latency_ms,
                "optimized_latency_ms": total_optimized * pbs_latency_ms,
                "latency_saved_ms": total_saved * pbs_latency_ms,
                "latency_reduction_percent": _percent(total_saved, total_baseline),
            }
        )

    result = {
        "model_name": resolved_model_name,
        "checkpoint_path": resolved_checkpoint_path,
        "dataset_name": resolved_dataset_name,
        "model_metadata": model_metadata,
        "source": source_info,
        "cost_model": {
            "encrypted_encrypted_non_unary_pbs": cost_model.encrypted_encrypted_non_unary_pbs,
            "mixed_public_encrypted_pbs": cost_model.mixed_public_encrypted_pbs,
            "public_public_pbs": cost_model.public_public_pbs,
            "unary_or_constant_pbs": cost_model.unary_or_constant_pbs,
        },
        "layer_stats": layer_stats,
        "totals": totals,
    }

    if save_csv is not None:
        save_analysis_csv(result, save_csv)
    if output_dir is not None:
        save_analysis_outputs([result], output_dir=output_dir, append=append)
    if verbose:
        print(format_analysis_table(result))

    return result


def load_gate_layers(
    model: Any, metadata: str | Path | Mapping[str, Any] | None = None
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if isinstance(model, (str, Path)):
        return _load_gate_layers_from_path(Path(model), metadata=metadata)

    if _looks_like_layer_sequence(model):
        return _load_gate_layers_from_sequence(model), {"kind": "sequence", "metadata": metadata}

    extracted = _try_extract_logic_layers_from_model(model)
    if extracted is not None:
        layers, source_info = extracted
        if metadata is not None:
            source_info = dict(source_info)
            source_info["metadata"] = metadata
        return layers, source_info

    raise TypeError(
        "Unsupported model source. Provide a model directory, gates CSV, layer "
        "sequence, or Python model containing LogicLayer-like objects."
    )


def analyze_many_pbs_counts(
    models: Sequence[Any],
    input_status: str | SignalStatus | Sequence[str | SignalStatus | bool | int] = "encrypted",
    pbs_latency_ms: float | None = None,
    verbose: bool = True,
    cost_model: CostModel | None = None,
    metadata: str | Path | Mapping[str, Any] | None = None,
    dataset_name: str | None = None,
    output_dir: str | Path | None = DEFAULT_OUTPUT_DIR,
    append: bool = False,
) -> list[dict[str, Any]]:
    """Analyze several checkpoints and write combined CSV/JSON outputs."""
    results = []
    for model in models:
        result = analyze_pbs_count(
            model,
            input_status=input_status,
            pbs_latency_ms=pbs_latency_ms,
            verbose=verbose,
            cost_model=cost_model,
            metadata=metadata,
            dataset_name=dataset_name,
        )
        results.append(result)

    if output_dir is not None:
        save_analysis_outputs(results, output_dir=output_dir, append=append)

    return results


def save_analysis_csv(result: Mapping[str, Any], save_csv: str | Path) -> None:
    """Save the paper-ready per-layer CSV for one analysis result."""
    path = Path(save_csv)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=PER_LAYER_FIELDNAMES)
        writer.writeheader()
        writer.writerows(build_per_layer_rows(result))


def save_analysis_outputs(
    results: Sequence[Mapping[str, Any]],
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    append: bool = False,
    per_layer_csv: str | Path | None = None,
    summary_csv: str | Path | None = None,
    json_path: str | Path | None = None,
) -> dict[str, Path]:
    """Save combined per-layer CSV, model-summary CSV, and JSON results."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    per_layer_path = Path(per_layer_csv) if per_layer_csv else output_path / DEFAULT_PER_LAYER_CSV
    summary_path = Path(summary_csv) if summary_csv else output_path / DEFAULT_SUMMARY_CSV
    results_json_path = Path(json_path) if json_path else output_path / DEFAULT_RESULTS_JSON

    per_layer_rows = []
    summary_rows = []
    for result in results:
        per_layer_rows.extend(build_per_layer_rows(result))
        summary_rows.append(build_summary_row(result))

    if append:
        per_layer_rows = _read_existing_csv_rows(per_layer_path) + per_layer_rows
        summary_rows = _read_existing_csv_rows(summary_path) + summary_rows

    _write_csv(per_layer_path, PER_LAYER_FIELDNAMES, per_layer_rows)
    _write_csv(summary_path, SUMMARY_FIELDNAMES, summary_rows)

    json_payload = {"results": [make_json_safe(result) for result in results]}
    if append and results_json_path.exists():
        try:
            existing = json.loads(results_json_path.read_text(encoding="utf-8"))
            if isinstance(existing, Mapping) and isinstance(existing.get("results"), list):
                json_payload["results"] = existing["results"] + json_payload["results"]
        except json.JSONDecodeError:
            pass

    results_json_path.parent.mkdir(parents=True, exist_ok=True)
    results_json_path.write_text(
        json.dumps(json_payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    return {
        "per_layer_csv": per_layer_path,
        "summary_csv": summary_path,
        "results_json": results_json_path,
    }


def build_per_layer_rows(result: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    model_metadata = result.get("model_metadata", {})
    for row in result["layer_stats"]:
        public_outputs = row["const0_out"] + row["const1_out"]
        rows.append(
            {
                "model_name": result.get("model_name"),
                "checkpoint_path": result.get("checkpoint_path"),
                "dataset_name": result.get("dataset_name"),
                "depth": model_metadata.get("depth") if isinstance(model_metadata, Mapping) else None,
                "width": model_metadata.get("width") if isinstance(model_metadata, Mapping) else None,
                "layer_index": row["layer"],
                "num_gates": row["gates"],
                "baseline_pbs": row["baseline_pbs"],
                "optimized_pbs": row["optimized_pbs"],
                "pbs_saved": row["saved"],
                "pbs_reduction_percent": row["pbs_reduction_percent"],
                "const0_outputs": row["const0_out"],
                "const1_outputs": row["const1_out"],
                "public_const_outputs": public_outputs,
                "encrypted_outputs": row["encrypted_out"],
                "public_const_output_percent": row["public_const_output_percent"],
                "encrypted_output_percent": row["encrypted_output_percent"],
                "estimated_baseline_latency_ms": row.get("estimated_baseline_latency_ms"),
                "estimated_optimized_latency_ms": row.get("estimated_optimized_latency_ms"),
                "estimated_latency_saved_ms": row.get("estimated_latency_saved_ms"),
                "estimated_latency_reduction_percent": row.get(
                    "estimated_latency_reduction_percent"
                ),
            }
        )
    return rows


def build_summary_row(result: Mapping[str, Any]) -> dict[str, Any]:
    totals = result["totals"]
    model_metadata = result.get("model_metadata", {})
    if not isinstance(model_metadata, Mapping):
        model_metadata = {}
    return {
        "model_name": result.get("model_name"),
        "checkpoint_path": result.get("checkpoint_path"),
        "dataset_name": result.get("dataset_name"),
        "depth": model_metadata.get("depth"),
        "width": model_metadata.get("width"),
        "best_epoch": model_metadata.get("best_epoch"),
        "best_val_accuracy": model_metadata.get("best_val_accuracy"),
        "test_accuracy_python_discrete": model_metadata.get("test_accuracy_python_discrete"),
        "test_accuracy_percent": model_metadata.get("test_accuracy_percent"),
        "num_layers": len(result["layer_stats"]),
        "total_gates": totals["total_gates"],
        "total_baseline_pbs": totals["total_baseline_pbs"],
        "total_optimized_pbs": totals["total_optimized_pbs"],
        "total_pbs_saved": totals["total_saved"],
        "total_pbs_reduction_percent": totals["pbs_saving_percent"],
        "total_const0_outputs": totals["total_const0_outputs"],
        "total_const1_outputs": totals["total_const1_outputs"],
        "total_public_const_outputs": totals["total_public_const_outputs"],
        "total_encrypted_outputs": totals["total_encrypted_outputs"],
        "final_layer_const0_outputs": totals["final_const0_outputs"],
        "final_layer_const1_outputs": totals["final_const1_outputs"],
        "final_layer_public_const_outputs": totals["final_public_const_outputs"],
        "final_layer_encrypted_outputs": totals["final_encrypted_outputs"],
        "estimated_total_baseline_latency_ms": totals.get("baseline_latency_ms"),
        "estimated_total_optimized_latency_ms": totals.get("optimized_latency_ms"),
        "estimated_total_latency_saved_ms": totals.get("latency_saved_ms"),
        "estimated_total_latency_reduction_percent": totals.get("latency_reduction_percent"),
    }


def format_analysis_table(result: Mapping[str, Any]) -> str:
    rows = list(result["layer_stats"])
    totals = result["totals"]
    total_row = {
        "layer": "Total",
        "gates": totals["total_gates"],
        "baseline_pbs": totals["total_baseline_pbs"],
        "optimized_pbs": totals["total_optimized_pbs"],
        "saved": totals["total_saved"],
        "const0_out": totals["final_const0_outputs"],
        "const1_out": totals["final_const1_outputs"],
        "encrypted_out": totals["final_encrypted_outputs"],
        "pbs_reduction_percent": totals["pbs_saving_percent"],
    }
    all_rows = rows + [total_row]

    headers = [
        ("layer", "Layer"),
        ("gates", "Gates"),
        ("baseline_pbs", "Baseline PBS"),
        ("optimized_pbs", "Optimized PBS"),
        ("saved", "Saved"),
        ("const0_out", "Const0 out"),
        ("const1_out", "Const1 out"),
        ("encrypted_out", "Encrypted out"),
        ("pbs_reduction_percent", "PBS reduction %"),
    ]
    formatted_rows = []
    for row in all_rows:
        formatted_rows.append(
            {
                key: (
                    f"{row[key]:.2f}"
                    if key == "pbs_reduction_percent"
                    else str(row[key])
                )
                for key, _ in headers
            }
        )

    widths = {
        key: max(len(title), *(len(row[key]) for row in formatted_rows))
        for key, title in headers
    }

    header_line = " | ".join(title.ljust(widths[key]) for key, title in headers)
    sep_line = "-+-".join("-" * widths[key] for key, _ in headers)
    body = [
        " | ".join(row[key].rjust(widths[key]) for key, _ in headers)
        for row in formatted_rows
    ]

    lines = [header_line, sep_line, *body]
    if "baseline_latency_ms" in totals:
        lines.extend(
            [
                "",
                "Latency estimate:",
                f"  per PBS: {totals['pbs_latency_ms']} ms",
                f"  baseline: {totals['baseline_latency_ms']:.3f} ms",
                f"  optimized: {totals['optimized_latency_ms']:.3f} ms",
                f"  saved: {totals['latency_saved_ms']:.3f} ms",
            ]
        )
    return "\n".join(lines)


def analysis_to_dataframe(result: Mapping[str, Any]) -> Any:
    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError("pandas is required for analysis_to_dataframe") from exc
    return pd.DataFrame(result["layer_stats"])


def make_json_safe(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(k): make_json_safe(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return [make_json_safe(v) for v in value]
    if isinstance(value, list):
        return [make_json_safe(v) for v in value]
    return value


def _write_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _read_existing_csv_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _resolve_checkpoint_path(
    checkpoint_path: str | Path | None,
    model: Any,
    source_info: Mapping[str, Any],
) -> str:
    if checkpoint_path is not None:
        return str(Path(checkpoint_path))
    if source_info.get("checkpoint"):
        return str(source_info["checkpoint"])
    if source_info.get("path"):
        return str(source_info["path"])
    if isinstance(model, (str, Path)):
        return str(model)
    return "<in-memory>"


def _infer_model_name(model: Any, source_info: Mapping[str, Any]) -> str:
    metadata = source_info.get("metadata")
    if isinstance(metadata, Mapping):
        for key in ("model_name", "model_file"):
            value = metadata.get(key)
            if value:
                return str(value)
        sweep = metadata.get("sweep")
        if isinstance(sweep, Mapping):
            depth = sweep.get("depth")
            width = sweep.get("width")
            if depth is not None and width is not None:
                return f"depth_{int(depth):02d}_width_{int(width):05d}"

    path_text = source_info.get("checkpoint") or source_info.get("path")
    if path_text:
        path = Path(str(path_text))
        if path.name == "best_lgn_gates.csv":
            return path.parent.name
        return path.name
    if isinstance(model, (str, Path)):
        path = Path(model)
        if path.name == "best_lgn_gates.csv":
            return path.parent.name
        return path.name
    return "in_memory_model"


def _extract_model_metadata(metadata: Any, checkpoint_path: str | Path | None) -> dict[str, Any]:
    depth = None
    width = None
    best_epoch = None
    best_val_accuracy = None
    test_accuracy = None

    if isinstance(metadata, Mapping):
        sweep = metadata.get("sweep")
        if isinstance(sweep, Mapping):
            depth = _to_int_or_none(sweep.get("depth"))
            width = _to_int_or_none(sweep.get("width"))

        depth = depth if depth is not None else _to_int_or_none(metadata.get("depth"))
        depth = depth if depth is not None else _to_int_or_none(metadata.get("num_layers"))
        width = width if width is not None else _to_int_or_none(metadata.get("width"))
        width = width if width is not None else _to_int_or_none(metadata.get("num_neurons"))

        best_epoch = _to_int_or_none(metadata.get("best_epoch"))
        best_val_accuracy = _to_float_or_none(metadata.get("best_val_accuracy"))
        test_accuracy = _to_float_or_none(
            metadata.get(
                "test_accuracy_python_discrete",
                metadata.get("test_accuracy"),
            )
        )

    if checkpoint_path is not None:
        path_text = str(checkpoint_path).replace("\\", "/")
        match = re.search(r"depth_(\d+)_width_(\d+)", path_text)
        if match:
            depth = depth if depth is not None else int(match.group(1))
            width = width if width is not None else int(match.group(2))

    return {
        "depth": depth,
        "width": width,
        "best_epoch": best_epoch,
        "best_val_accuracy": best_val_accuracy,
        "test_accuracy_python_discrete": test_accuracy,
        "test_accuracy_percent": _accuracy_to_percent(test_accuracy),
    }


def _to_int_or_none(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _to_float_or_none(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _accuracy_to_percent(value: float | None) -> float | None:
    if value is None:
        return None
    return value * 100.0 if value <= 1.0 else value


def _infer_dataset_name(metadata: Any, checkpoint_path: str | Path | None = None) -> str:
    path_text = str(checkpoint_path).lower().replace("\\", "/") if checkpoint_path is not None else ""
    if "trained_models_mnist_depth_vs_width" in path_text:
        return "MNIST Depth vs Width"

    if isinstance(metadata, Mapping):
        for key in ("dataset_name", "dataset"):
            value = metadata.get(key)
            if value:
                return str(value)
    if checkpoint_path is not None:
        text = path_text
        if "fashionmnist" in text or "fashion_mnist" in text or "fashion-mnist" in text:
            return "FashionMNIST"
        if "uci_phishing" in text or ("uci" in text and "phishing" in text):
            return "UCI Phishing Websites"
        if "mnist" in text:
            return "MNIST"
    return "unknown"


def _load_gate_layers_from_path(
    path: Path, metadata: str | Path | Mapping[str, Any] | None = None
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    source_path = path
    if path.is_dir():
        gates_csv = path / "best_lgn_gates.csv"
        metadata_path = path / "best_lgn_metadata.json"
    elif path.name == "best_lgn_gates.csv" or path.suffix.lower() == ".csv":
        gates_csv = path
        metadata_path = path.with_name("best_lgn_metadata.json")
    elif path.parent.joinpath("best_lgn_gates.csv").exists():
        gates_csv = path.parent / "best_lgn_gates.csv"
        metadata_path = path.parent / "best_lgn_metadata.json"
    else:
        loaded_model = _try_torch_load(path)
        if loaded_model is None:
            raise FileNotFoundError(
                f"Could not find best_lgn_gates.csv for {path}. Provide a model "
                "directory, gates CSV, or checkpoint with a sibling export."
            )
        layers, info = load_gate_layers(loaded_model, metadata=metadata)
        info["checkpoint"] = str(path)
        return layers, info

    if metadata is None and metadata_path.exists():
        metadata_obj: Mapping[str, Any] | None = json.loads(metadata_path.read_text(encoding="utf-8"))
    elif metadata is None:
        metadata_obj = None
    elif isinstance(metadata, Mapping):
        metadata_obj = metadata
    else:
        metadata_obj = json.loads(Path(metadata).read_text(encoding="utf-8"))

    layers = _read_gates_csv(gates_csv)
    return layers, {
        "kind": "csv",
        "path": str(source_path),
        "gates_csv": str(gates_csv),
        "metadata_path": str(metadata_path) if metadata_path.exists() else None,
        "metadata": metadata_obj,
    }


def _read_gates_csv(path: Path) -> list[dict[str, Any]]:
    rows_by_layer: dict[int, list[GateSpec]] = {}
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"layer", "neuron", "input_a", "input_b"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(
                f"{path} must contain columns {sorted(required)} plus gate_id or gate_name"
            )

        for row in reader:
            gate_ref: int | str
            if row.get("gate_id") not in (None, ""):
                gate_ref = int(row["gate_id"])
            elif row.get("gate_name") not in (None, ""):
                gate_ref = str(row["gate_name"])
            else:
                raise ValueError(f"CSV row lacks gate_id/gate_name: {row}")

            gate = GateSpec(
                layer=int(row["layer"]),
                neuron=int(row["neuron"]),
                input_a=int(row["input_a"]),
                input_b=int(row["input_b"]),
                gate=gate_ref,
                gate_name=row.get("gate_name") or None,
            )
            rows_by_layer.setdefault(gate.layer, []).append(gate)

    return [
        {"label": layer_label, "gates": sorted(gates, key=lambda g: g.neuron)}
        for layer_label, gates in sorted(rows_by_layer.items())
    ]


def _load_gate_layers_from_sequence(model: Any) -> list[dict[str, Any]]:
    layers: list[dict[str, Any]] = []
    for layer_idx, raw_layer in enumerate(model):
        raw_gates = list(raw_layer)
        gates = [_coerce_gate_spec(g, layer_idx, neuron_idx) for neuron_idx, g in enumerate(raw_gates)]
        layers.append({"label": layer_idx, "gates": gates})
    return layers


def _coerce_gate_spec(raw: Any, layer_idx: int, neuron_idx: int) -> GateSpec:
    if isinstance(raw, GateSpec):
        return raw

    if isinstance(raw, Mapping):
        gate = raw.get("gate", raw.get("gate_id", raw.get("gate_name")))
        if gate is None:
            raise ValueError(f"Gate mapping lacks gate/gate_id/gate_name: {raw}")
        return GateSpec(
            layer=int(raw.get("layer", layer_idx)),
            neuron=int(raw.get("neuron", neuron_idx)),
            input_a=int(raw["input_a"]),
            input_b=int(raw["input_b"]),
            gate=gate,
            gate_name=raw.get("gate_name"),
        )

    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
        if len(raw) < 3:
            raise ValueError(f"Gate tuple must be (input_a, input_b, gate), got {raw!r}")
        return GateSpec(
            layer=layer_idx,
            neuron=neuron_idx,
            input_a=int(raw[0]),
            input_b=int(raw[1]),
            gate=raw[2],
        )

    raise TypeError(f"Unsupported gate spec: {raw!r}")


def _looks_like_layer_sequence(model: Any) -> bool:
    if isinstance(model, (str, bytes, Path, Mapping)):
        return False
    try:
        layers = list(model)
    except TypeError:
        return False
    if not layers:
        return False
    return all(
        isinstance(layer, Iterable)
        and not hasattr(layer, "weights")
        and not isinstance(layer, (str, bytes, Mapping))
        for layer in layers
    )


def _try_extract_logic_layers_from_model(model: Any) -> tuple[list[dict[str, Any]], dict[str, Any]] | None:
    try:
        iterable = list(model)
    except TypeError:
        iterable = [model]

    layers: list[dict[str, Any]] = []
    for model_layer_idx, layer in enumerate(iterable):
        if not (hasattr(layer, "weights") and hasattr(layer, "indices")):
            continue

        weights = getattr(layer, "weights")
        indices = getattr(layer, "indices")
        gate_ids = _tensor_to_list(weights.argmax(1))
        input_a = _tensor_to_list(indices[0])
        input_b = _tensor_to_list(indices[1])

        if not (len(gate_ids) == len(input_a) == len(input_b)):
            raise ValueError(
                f"Logic layer {model_layer_idx} has mismatched gate/input lengths"
            )

        gates = [
            GateSpec(
                layer=model_layer_idx,
                neuron=neuron_idx,
                input_a=int(a),
                input_b=int(b),
                gate=int(gate_id),
                gate_name=GATE_ID_TO_NAME.get(int(gate_id)),
            )
            for neuron_idx, (a, b, gate_id) in enumerate(zip(input_a, input_b, gate_ids))
        ]
        layers.append({"label": model_layer_idx, "gates": gates})

    if not layers:
        return None

    warnings.warn(
        "No finalized discrete gate export was provided; PBS count is based on "
        "argmax discretization of layer.weights.",
        RuntimeWarning,
        stacklevel=2,
    )
    return layers, {"kind": "python_model_argmax", "metadata": None}


def _tensor_to_list(value: Any) -> list[Any]:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "tolist"):
        return value.tolist()
    return list(value)


def _try_torch_load(path: Path) -> Any | None:
    try:
        import torch
    except ImportError:
        return None
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except Exception:
        return None


def _infer_input_count(layers: Sequence[Mapping[str, Any]], metadata: Mapping[str, Any] | None) -> int:
    if metadata:
        for key in ("in_dim", "input_dim"):
            if key in metadata:
                return int(metadata[key])
    first_layer = layers[0]["gates"]
    max_index = max(max(g.input_a, g.input_b) for g in first_layer)
    return max_index + 1


def _expand_input_status(
    input_status: str | SignalStatus | Sequence[str | SignalStatus | bool | int],
    input_count: int,
) -> list[SignalStatus]:
    if isinstance(input_status, Sequence) and not isinstance(input_status, (str, bytes)):
        statuses = [coerce_signal_status(status) for status in input_status]
        if len(statuses) != input_count:
            raise ValueError(
                f"input_status length is {len(statuses)}, expected {input_count}"
            )
        return statuses
    return [coerce_signal_status(input_status)] * input_count


def _percent(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return 100.0 * numerator / denominator


def _number_for_display(value: float) -> int | float:
    if math.isclose(value, round(value)):
        return int(round(value))
    return value
