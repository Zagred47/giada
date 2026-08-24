import ast
import inspect
import json
from pathlib import Path

from src.hayflow_model.state_scheduled_sampling_confirmation import (
    EXPECTED_06BF_INDEX_SHA256,
    SCHEDULED_ARMS,
    StateScheduledSamplingConfig,
    StateScheduledSamplingConfirmation,
)


ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = (
    ROOT / "experiments" / "hayflow" / "06b_g_state_scheduled_sampling_confirmation"
)
NOTEBOOK = ROOT / "notebooks" / "06b_g_state_scheduled_sampling_confirmation.ipynb"
CONFIG = (
    ROOT / "configs" / "hayflow" / "hayflow_state_scheduled_sampling_confirmation.yml"
)


def test_06bg_preregisters_independent_confirmation_and_hierarchy():
    prereg = json.loads((EXPERIMENT / "preregistration.json").read_text())
    assert prereg["status"] == "preregistered_before_execution"
    assert prereg["independent_confirmation"]["train_split_only"]
    assert prereg["independent_confirmation"][
        "trajectory_disjoint_from_all_06b_f_roles"
    ]
    assert prereg["continuation_matrix"]["primary_arm"] == (
        "state_linear_curriculum"
    )
    assert prereg["continuation_matrix"]["fallback_arm"] == "scalar_continue"
    assert prereg["continuation_matrix"][
        "hierarchical_fallback_registered_before_execution"
    ]


def test_06bg_configuration_uses_direct_eight_ms_fixed_continuation():
    config = StateScheduledSamplingConfig()
    config.validate()
    text = CONFIG.read_text(encoding="utf-8")
    assert config.scheduled_unroll_horizon_ms == 8
    assert config.scheduled_checkpoints == (0, 100, 200, 400)
    assert config.curriculum_decay_steps == 300
    assert config.curriculum_initial_teacher_probability == 0.5
    assert "scheduled_unroll_horizon_ms: 8" in text


def test_06bg_has_five_matched_interventions_and_exact_source():
    assert list(SCHEDULED_ARMS) == [
        "scalar_continue",
        "state_linear_curriculum",
        "joint_linear_curriculum",
        "state_fixed_25",
        "shuffled_continue",
    ]
    assert EXPECTED_06BF_INDEX_SHA256 == (
        "0c0103784aa1da435e43252d29d5b10d65c7e6baf8e94075f4cc0babe03fb799"
    )


def test_06bg_confirmation_role_is_next_unused_component_prefix():
    source = inspect.getsource(
        StateScheduledSamplingConfirmation._build_independent_confirmation_role
    )
    assert "consumed =" in source
    assert "components[consumed:stop]" in source
    assert "previous & confirmation" in source
    assert 'row.get("split")' in source


def test_06bg_training_shares_batches_draws_and_never_reads_confirmation():
    source = inspect.getsource(
        StateScheduledSamplingConfirmation.train_synchronized_scheduled_matrix
    )
    row_position = source.index("rows = rng.choice")
    arm_position = source.index("for arm, pair in pairs.items()", row_position)
    assert row_position < arm_position
    assert "replace=False" in source
    assert "state_uniform = rng.random" in source
    assert "voltage_uniform = rng.random" in source
    assert source.index("state_uniform = rng.random") < arm_position
    assert 'self._batch_tensors("fit"' in source
    assert 'self._batch_tensors("confirmation"' not in source


def test_06bg_finalization_preserves_preregistered_primary_then_fallback():
    source = inspect.getsource(
        StateScheduledSamplingConfirmation.finalize_scheduled_sampling_confirmation
    )
    assert 'primary = "state_linear_curriculum"' in source
    assert 'fallback = "scalar_continue"' in source
    assert source.index("if source_confirmed and primary_pass") < source.index(
        "elif source_confirmed and fallback_pass"
    )
    assert '"coupled_06c_canary_authorized": authorize_06c' in source
    assert '"full_training_authorized": False' in source


def test_06bg_notebook_is_compact_and_uses_stable_blob_download():
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    assert len(notebook["cells"]) <= 12
    code = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    )
    for cell in notebook["cells"]:
        if cell.get("cell_type") == "code":
            ast.parse("".join(cell.get("source", [])))
            assert not cell.get("outputs")
    assert "EXPECTED_06BF_INDEX_SHA256" in code
    assert "train_synchronized_scheduled_matrix" in code
    assert "evaluate_independent_confirmation" in code
    assert "base64.b64encode" in code and "new Blob" in code
    assert "FileLink" not in code
    assert "display(confirmation_report)" not in code
