from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ei_ddlgn.pbs_analyzer import (  # noqa: E402
    SignalStatus,
    analyze_lut2_gate,
    analyze_pbs_count,
    save_analysis_outputs,
    truth_table_from_gate,
)


CHEAP_IDS = {0, 3, 5, 10, 12, 15}


def expected_unary_status(values):
    if values == [0, 0]:
        return SignalStatus.PUBLIC_CONST_0
    if values == [1, 1]:
        return SignalStatus.PUBLIC_CONST_1
    return SignalStatus.ENCRYPTED


def test_both_encrypted_inputs_cost_model():
    for gate_id in range(16):
        result = analyze_lut2_gate(
            gate_id,
            SignalStatus.ENCRYPTED,
            SignalStatus.ENCRYPTED,
        )
        expected_pbs = 0 if gate_id in CHEAP_IDS else 1
        assert result.pbs == expected_pbs


def test_encrypted_first_public_second_input():
    for gate_id in range(16):
        tt = truth_table_from_gate(gate_id)
        for cb in (0, 1):
            result = analyze_lut2_gate(
                gate_id,
                SignalStatus.ENCRYPTED,
                SignalStatus.PUBLIC_CONST_1 if cb else SignalStatus.PUBLIC_CONST_0,
            )
            expected = expected_unary_status([tt[cb], tt[2 + cb]])
            assert result.pbs == 0
            assert result.output_status == expected


def test_public_first_encrypted_second_input():
    for gate_id in range(16):
        tt = truth_table_from_gate(gate_id)
        for ca in (0, 1):
            result = analyze_lut2_gate(
                gate_id,
                SignalStatus.PUBLIC_CONST_1 if ca else SignalStatus.PUBLIC_CONST_0,
                SignalStatus.ENCRYPTED,
            )
            expected = expected_unary_status([tt[2 * ca], tt[2 * ca + 1]])
            assert result.pbs == 0
            assert result.output_status == expected


def test_both_public_inputs():
    for gate_id in range(16):
        tt = truth_table_from_gate(gate_id)
        for ca in (0, 1):
            for cb in (0, 1):
                result = analyze_lut2_gate(
                    gate_id,
                    SignalStatus.PUBLIC_CONST_1 if ca else SignalStatus.PUBLIC_CONST_0,
                    SignalStatus.PUBLIC_CONST_1 if cb else SignalStatus.PUBLIC_CONST_0,
                )
                expected_bit = tt[2 * ca + cb]
                expected_status = (
                    SignalStatus.PUBLIC_CONST_1
                    if expected_bit
                    else SignalStatus.PUBLIC_CONST_0
                )
                assert result.pbs == 0
                assert result.output_status == expected_status


def test_constant_propagation_across_layers_saves_pbs():
    # Layer 0 emits one public zero and one encrypted identity.
    # Layer 1 uses XOR, which costs PBS for encrypted/encrypted inputs, but with
    # the public zero input it collapses to identity and costs no PBS.
    toy_layers = [
        [
            (0, 1, 0),  # constant 0
            (0, 1, 3),  # a
        ],
        [
            (0, 1, 6),  # xor
        ],
    ]
    result = analyze_pbs_count(toy_layers, verbose=False)

    assert result["layer_stats"][0]["baseline_pbs"] == 0
    assert result["layer_stats"][0]["optimized_pbs"] == 0
    assert result["layer_stats"][1]["baseline_pbs"] == 1
    assert result["layer_stats"][1]["optimized_pbs"] == 0
    assert result["totals"]["total_saved"] == 1


def test_paper_ready_csv_and_json_exports(tmp_path):
    toy_layers = [
        [
            (0, 1, 0),
            (0, 1, 3),
        ],
        [
            (0, 1, 6),
        ],
    ]
    result = analyze_pbs_count(
        toy_layers,
        pbs_latency_ms=8.5,
        verbose=False,
        model_name="toy",
        checkpoint_path="toy_checkpoint",
        dataset_name="toy_dataset",
    )
    paths = save_analysis_outputs([result], output_dir=tmp_path)

    per_layer = paths["per_layer_csv"].read_text(encoding="utf-8")
    summary = paths["summary_csv"].read_text(encoding="utf-8")
    payload = paths["results_json"].read_text(encoding="utf-8")

    assert "model_name,checkpoint_path,dataset_name,depth,width,layer_index" in per_layer
    assert "toy,toy_checkpoint,toy_dataset,,,1,1,1,0,1" in per_layer
    assert "total_pbs_saved" in summary
    assert "estimated_total_latency_saved_ms" in summary
    assert '"model_name": "toy"' in payload
