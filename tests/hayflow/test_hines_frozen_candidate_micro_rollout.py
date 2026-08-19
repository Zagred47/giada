import ast
import inspect
import json
from pathlib import Path

import numpy as np
import pytest

import src.hayflow_model.hines_frozen_candidate_micro_rollout as rollout_module
from src.hayflow_model.hines_frozen_candidate_micro_rollout import (
    HinesFrozenCandidateMicroRolloutConfig,
    rollout_voltage_metrics,
)


ROOT = Path(__file__).resolve().parents[2]


def test_05k_registered_config_freezes_all_seeds_horizons_and_pair_count():
    config = HinesFrozenCandidateMicroRolloutConfig()
    config.validate()
    assert config.seeds == (17, 29, 43)
    assert config.horizons_ms == (2, 4, 8)
    assert config.branch_step == 4 and config.pair_count == 32
    assert config.minimum_passing_seeds == 2
    with pytest.raises(ValueError, match="2/4/8"):
        HinesFrozenCandidateMicroRolloutConfig(horizons_ms=(2, 4)).validate()
    with pytest.raises(ValueError, match="registered seeds"):
        HinesFrozenCandidateMicroRolloutConfig(seeds=(17, 29)).validate()


def test_rollout_voltage_metrics_measures_endpoint_trace_and_pair_retention():
    target = np.zeros((4, 2, 3), dtype=np.float64)
    target[1, -1] = 2.0
    target[3, -1] = 4.0
    prediction = target.copy()
    report = rollout_voltage_metrics(
        prediction, target, physical_min_mv=-150.0, physical_max_mv=100.0
    )
    assert report["episode_count"] == 4 and report["pair_count"] == 2
    assert report["horizon_ms"] == 2
    assert report["endpoint_voltage_rmse_mv"] == 0.0
    assert report["trace_voltage_rmse_mv"] == 0.0
    assert report["median_branching_retention"] == pytest.approx(1.0)
    assert report["physical_voltage_violation_count"] == 0
    assert report["numerically_finite"] is True


def test_05jo_registered_result_matches_05k_provenance_constants():
    result = json.loads(
        (ROOT / "experiments/hayflow/05j_o_regenerative_fresh_test/result.json")
        .read_text(encoding="utf-8")
    )
    archive = result["archive"]
    assert archive["sha256"] == rollout_module.EXPECTED_05JO_ARCHIVE_SHA256
    assert archive["artifact_index_sha256"] == rollout_module.EXPECTED_05JO_INDEX_SHA256
    assert archive["final_report_sha256"] == rollout_module.EXPECTED_05JO_FINAL_SHA256
    assert (
        result["teacher"]["transition_store_sha256"]
        == rollout_module.EXPECTED_05JO_TRANSITION_SHA256
    )
    assert result["frozen_model_evaluation"]["passing_seed_count"] == 3
    assert result["micro_rollout_authorized"] is True
    assert result["full_training_authorized"] is False


def test_05k_future_teacher_fields_are_explicitly_zeroed_after_initialization():
    source = inspect.getsource(
        rollout_module.HinesFrozenCandidateMicroRollout._hide_future_teacher_state
    )
    for name in (
        "teacher_state_t",
        "voltage_t",
        "calcium_t",
        "synapse_state_t",
        "anchor_voltage_t",
    ):
        assert name in source
    assert "torch.zeros_like" in source
    candidate = inspect.getsource(
        rollout_module.HinesFrozenCandidateMicroRollout._candidate_rollout
    )
    assert "recurrent=recurrent" in candidate
    assert 'decode_teacher=False' in candidate
    assert "_hide_future_teacher_state" in candidate
    assert "_causal_frontend_batch" in candidate


def test_05k_synapse_frontend_exposes_only_authentic_ab_state_and_realized_drive():
    state = inspect.getsource(
        rollout_module.HinesFrozenCandidateMicroRollout._authentic_synapse_frontend_state
    )
    assert "np.zeros_like" in state
    assert "a_index" in state and "b_index" in state
    drive = inspect.getsource(
        rollout_module.HinesFrozenCandidateMicroRollout._causal_frontend_batch
    )
    assert "encode_realized_synaptic_drive" in drive
    assert "_authentic_synapse_frontend_state" in drive


def test_05k_verifies_05jo_authorization_before_opening_fresh_store():
    source = inspect.getsource(
        rollout_module.HinesFrozenCandidateMicroRollout.prepare_frozen_candidate_micro_rollout
    )
    verify = source.index("verified_fresh_test_artifact_root")
    gate = source.index("05j-o did not authorize micro-rollout")
    open_store = source.index("self.prepare_fresh_test_evaluation()")
    assert verify < gate < open_store


def test_05k_preserves_inherited_metric_and_reconstruction_contracts():
    cls = rollout_module.HinesFrozenCandidateMicroRollout
    assert "_metrics" not in cls.__dict__
    assert "fit_fixed_tree_ridge_baseline" not in cls.__dict__
    assert "reconstruct_frozen_checkpoints" not in cls.__dict__
    assert list(inspect.signature(cls._metrics).parameters)[:3] == [
        "self",
        "role",
        "residual",
    ]


def test_05k_notebook_is_compact_frozen_and_uses_browser_blob_download():
    notebook = json.loads(
        (ROOT / "notebooks/05k_frozen_candidate_micro_rollout.ipynb").read_text(
            encoding="utf-8"
        )
    )
    source = "\n".join(
        "".join(cell.get("source", [])) for cell in notebook["cells"]
    )
    assert "torch.cuda.is_available" in source
    assert "evaluate_frozen_micro_rollout" in source
    assert "future_teacher_membrane_or_ion_states_injected" in source
    assert "run_decoder_refit" not in source
    assert "checkpoint_selection" not in source
    assert "base64.b64encode" in source and "application/zip" in source
    assert "FileLink" not in source and "rglob('/kaggle" not in source
    assert "display(rollout)" not in source
    assert "display(final_report)" not in source
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            ast.parse("".join(cell.get("source", [])))
