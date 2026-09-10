import hashlib
import importlib.util
import json
from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "baselines"
    / "qat-fcnn-mnist"
    / "aggregate_reports.py"
)
SPEC = importlib.util.spec_from_file_location("qat_aggregate", SCRIPT)
qat_aggregate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(qat_aggregate)


def write_report(path, seed, accuracy, latency, bit_width):
    path.write_text(
        json.dumps(
            {
                "settings": {"sparsity": 4, "seed": seed},
                "results": {
                    "accuracy": {"VL full": accuracy},
                    "timing": {"fhe_encrypt_run_decrypt_avg_seconds": latency},
                    "bit_widths": {"FHE short": bit_width},
                },
            }
        ),
        encoding="utf-8",
    )


def test_qat_report_loading_and_bit_width_summary(tmp_path):
    write_report(tmp_path / "seed_1.json", 1, 0.91, 80.0, 6)
    write_report(tmp_path / "seed_2.json", 2, 0.93, 90.0, 7)

    groups = qat_aggregate.load_reports(tmp_path)

    assert sorted(groups) == [4]
    assert [run["seed"] for run in groups[4]] == [1, 2]
    assert qat_aggregate.bit_width_summary(groups[4]) == "6-bit: 50%; 7-bit: 50%"


def test_qat_reference_output_checksums():
    baseline_dir = SCRIPT.parent
    manifest = baseline_dir / "REFERENCE_OUTPUTS.sha256"
    for line in manifest.read_text(encoding="utf-8").splitlines():
        expected, relative_path = line.split(maxsplit=1)
        contents = (baseline_dir / relative_path).read_bytes()
        assert hashlib.sha256(contents).hexdigest() == expected
