import ast
import inspect
import json
from pathlib import Path

import pytest

import src.hayflow_model.hines_regenerative_fresh_test as model_module
import src.hayflow_teacher.regenerative_fresh_test as teacher_module
from src.hayflow_model.hines_regenerative_fresh_test import (
    HinesRegenerativeFreshTestConfig,
)
from src.hayflow_teacher.regenerative_fresh_test import (
    RegenerativeFreshTestConfig,
    protocol_from_frozen_row,
)
from src.hayflow_teacher.regenerative_training_support import _plan_rows


ROOT = Path(__file__).resolve().parents[2]


def test_05jo_configs_freeze_cardinality_seeds_and_robust_gate():
    teacher = RegenerativeFreshTestConfig()
    teacher.validate()
    assert (teacher.pair_count, teacher.episode_count, teacher.transition_count) == (
        32,
        64,
        768,
    )
    assert list(range(teacher.seed_start, teacher.seed_start + teacher.pair_count)) == list(
        range(1_100_001, 1_100_033)
    )
    model = HinesRegenerativeFreshTestConfig()
    model.validate()
    assert model.seeds == (17, 29, 43)
    assert model.minimum_passing_seeds == 2
    assert model.minimum_improvement_vs_best_baseline_fraction == pytest.approx(0.05)
    with pytest.raises(ValueError, match="preregistered"):
        RegenerativeFreshTestConfig(pair_count=31).validate()
    with pytest.raises(ValueError, match="registered"):
        HinesRegenerativeFreshTestConfig(seeds=(17, 29)).validate()


def test_05jo_protocol_roundtrip_preserves_frozen_plan_row():
    row = {
        "trajectory_id": "05jm-fresh_test-pair-0000-low",
        "category": "dendritic_events",
        "protocol": "regenerative_fresh_test",
        "protocol_id": "05jm-fresh-test",
        "protocol_variant": "low",
        "seed": 1_100_001,
        "duration_ms": 12,
        "split": "test",
        "stimulus_onset_step": 4,
        "snapshot_source": "05jm-fresh_test-snapshot-0000",
        "metadata": {
            "branch_arm": "low",
            "snapshot_id": "05jm-fresh_test-snapshot-0000",
        },
        "actions": {
            "4": [
                {
                    "kind": "synaptic_event",
                    "offset_ms": 0.2,
                    "synapse_id": 72,
                    "weight_multiplier": 1.0,
                    "duration_ms": None,
                    "amplitude_na": None,
                    "release_observed": None,
                    "rng_sequence_before": None,
                    "metadata": {},
                }
            ]
        },
    }
    assert _plan_rows([protocol_from_frozen_row(row)]) == [row]


def test_05jo_registered_05jn_result_matches_both_provenance_contracts():
    result = json.loads(
        (ROOT / "experiments/hayflow/05j_n_regenerative_decoder_refit/result.json")
        .read_text(encoding="utf-8")
    )
    archive = result["archive"]
    assert archive["sha256"] == teacher_module.EXPECTED_05JN_ARCHIVE_SHA256
    assert archive["sha256"] == model_module.EXPECTED_05JN_ARCHIVE_SHA256
    assert archive["artifact_index_sha256"] == teacher_module.EXPECTED_05JN_INDEX_SHA256
    assert archive["artifact_index_sha256"] == model_module.EXPECTED_05JN_INDEX_SHA256
    assert archive["final_report_sha256"] == teacher_module.EXPECTED_05JN_FINAL_SHA256
    assert archive["final_report_sha256"] == model_module.EXPECTED_05JN_FINAL_SHA256
    assert result["passing_seed_count"] == 3
    assert result["fresh_test_generation_authorized"] is True
    assert result["candidate_model_authorized"] is False
    assert result["full_training_authorized"] is False


def test_05jo_verifies_05jn_before_opening_the_sealed_05jm_plan():
    source = inspect.getsource(teacher_module.RegenerativeFreshTestSession.open_authorized_plan)
    verification_n = source.index("contract_n, payload_n = _verify_artifact")
    parse_n = source.index('report_n = json.loads(payload_n["final_report.json"])')
    verification_m = source.index("contract_m, payload_m = _verify_artifact")
    parse_sealed = source.index('sealed = json.loads(payload_m["sealed_fresh_test_plan.json"])')
    assert verification_n < parse_n < verification_m < parse_sealed


def test_05jo_preserves_all_inherited_metric_and_reconstruction_contracts():
    cls = model_module.HinesRegenerativeFreshTestEvaluation
    assert "_metrics" not in cls.__dict__
    assert "fit_fixed_tree_ridge_baseline" not in cls.__dict__
    assert "reconstruct_frozen_checkpoints" not in cls.__dict__
    signature = inspect.signature(cls._metrics)
    assert list(signature.parameters)[:3] == ["self", "role", "residual"]
    evaluation_source = inspect.getsource(cls.evaluate_frozen_checkpoints)
    assert "torch.optim" not in evaluation_source
    assert "load_state_dict" in evaluation_source
    assert "checkpoint_selection_performed" in evaluation_source
    assert "retraining_performed" in evaluation_source


def test_05jo_notebook_has_exact_fresh_test_and_browser_blob_download_contract():
    notebook = json.loads(
        (ROOT / "notebooks/05j_o_regenerative_fresh_test.ipynb").read_text(
            encoding="utf-8"
        )
    )
    source = "\n".join(
        "".join(cell.get("source", [])) for cell in notebook["cells"]
    )
    assert "torch.cuda.is_available" in source
    assert "open_authorized_plan" in source
    assert "generate_fresh_test_shard" in source
    assert "validate_fresh_test_shard" in source
    assert "evaluate_frozen_checkpoints" in source
    assert "fresh_manifest['trajectory_count']==64" in source
    assert "fresh_manifest['transition_count']==768" in source
    assert "run_decoder_refit" not in source
    assert "not evaluation_report['checkpoint_selection_performed']" in source
    assert "not evaluation_report['retraining_performed']" in source
    assert "not final_report['full_training_authorized']" in source
    assert "base64.b64encode" in source and "application/zip" in source
    assert "FileLink" not in source and "rglob('/kaggle" not in source
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            ast.parse("".join(cell.get("source", [])))
