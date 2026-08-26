import ast
import inspect
import json
from pathlib import Path

import pytest

from src.hayflow_model import atomic_state_dynamics_playground as atomic
from src.hayflow_model.continuous_mixture_state_playground import (
    EXPECTED_06BL_ARCHIVE_SHA256,
    EXPECTED_06BL_FINAL_SHA256,
    EXPECTED_06BL_INDEX_SHA256,
    FULL_MATRIX_ARMS,
    LOCAL_RECURRENT,
    PHYSIOLOGY_INSTANTANEOUS,
    SCALING_ARMS,
    SHUFFLED_TREE_RECURRENT,
    TREE_RECURRENT,
    TREE_RECURRENT_NO_ORACLE_AUX,
    VOLTAGE_INSTANTANEOUS,
    ContinuousMixtureCell,
    ContinuousMixtureStateConfig,
    ContinuousMixtureStatePlayground,
)


ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_L = ROOT / "experiments" / "hayflow" / "06b_l_voltage_error_model_revision"
EXPERIMENT_M = ROOT / "experiments" / "hayflow" / "06b_m_continuous_mixture_state_playground"
NOTEBOOK = ROOT / "notebooks" / "06b_m_continuous_mixture_state_playground.ipynb"


def test_06bm_registers_exact_terminal_06bl_result():
    result = json.loads((EXPERIMENT_L / "result.json").read_text())
    assert result["archive_sha256"] == EXPECTED_06BL_ARCHIVE_SHA256
    assert result["artifact_index_sha256"] == EXPECTED_06BL_INDEX_SHA256
    assert result["final_report_sha256"] == EXPECTED_06BL_FINAL_SHA256
    assert result["formal_diagnosis"] == (
        "OPTIMAL_BLEND_ORACLE_WORKS_BUT_REGIME_GATE_FAILS"
    )
    assert result["decision"]["architecture_revision"] == "continuous_mixture_state"


def test_06bm_preregisters_small_orthogonal_factorial_matrix():
    prereg = json.loads((EXPERIMENT_M / "preregistration.json").read_text())
    assert prereg["status"] == "preregistered_before_execution"
    assert set(prereg["factorial_matrix"]) == set(FULL_MATRIX_ARMS)
    assert set(prereg["mini_scaling_law"]["scaling_arms"]) == set(SCALING_ARMS)
    assert prereg["common_controller"]["parameter_matched_within_width"]
    assert not prereg["training_contract"]["teacher_endpoint_used_as_input"]
    assert prereg["training_contract"][
        "teacher_optimal_blend_used_only_as_training_target"
    ]


def test_06bm_registers_verified_terminal_result():
    result = json.loads((EXPERIMENT_M / "result.json").read_text())
    assert result["archive_sha256"] == (
        "9b97c3b4a465f376f98b97a9408f7accd0268a0681b8be8c432f40327f97bfee"
    )
    assert result["artifact_index_sha256"] == (
        "319bacdeece4749407643816977550b3b326694f787503bb50dcc3d3e1ef73f6"
    )
    assert result["final_report_sha256"] == (
        "5f35dc5e4a4d3b308d8c1ccaeed2cba4da780ced201a42f9ee36640bb5c1628b"
    )
    assert result["formal_diagnosis"] == (
        "MIXTURE_TARGET_LEARNABLE_BUT_RECURSIVE_COMPOSITION_FAILS"
    )
    assert result["decision"]["architecture_revision"] == (
        "continuous_mixture_objective_and_coupling_revision"
    )
    assert not result["decision"]["fresh_train_support_confirmation_authorized"]


def test_06bm_configuration_and_run_specs_are_bounded():
    config = ContinuousMixtureStateConfig()
    config.validate()
    instance = object.__new__(ContinuousMixtureStatePlayground)
    instance.config = config
    specs = instance._run_specs()
    assert len(specs) == 12
    assert len(set(specs)) == len(specs)
    assert all((arm, 16) in specs for arm in FULL_MATRIX_ARMS)
    assert all((arm, width) in specs for arm in SCALING_ARMS for width in (8, 32))
    assert config.mixture_checkpoints == (0, 100, 200, 400)
    assert config.maximum_mixture_parameter_count == 20000


def test_06bm_arm_matrix_is_pairwise_interpretable():
    contract = ContinuousMixtureStatePlayground._arm_contract
    voltage = contract(VOLTAGE_INSTANTANEOUS)
    physiology = contract(PHYSIOLOGY_INSTANTANEOUS)
    local = contract(LOCAL_RECURRENT)
    tree = contract(TREE_RECURRENT)
    shuffled = contract(SHUFFLED_TREE_RECURRENT)
    no_aux = contract(TREE_RECURRENT_NO_ORACLE_AUX)
    assert voltage["input"] != physiology["input"]
    assert not physiology["recurrent"] and local["recurrent"]
    assert local["topology"] == "local" and tree["topology"] == "authentic_tree"
    assert shuffled["topology"] == "relabelled_tree"
    assert tree["oracle_auxiliary"] and not no_aux["oracle_auxiliary"]


@pytest.mark.skipif(atomic.torch is None, reason="PyTorch is optional locally")
def test_06bm_controllers_are_parameter_matched_within_width():
    models = [ContinuousMixtureCell(64, 11, 4, 16) for _ in FULL_MATRIX_ARMS]
    counts = [sum(parameter.numel() for parameter in model.parameters()) for model in models]
    assert len(set(counts)) == 1
    batch, segments = 2, 7
    features = atomic.torch.randn(batch, segments, 64)
    hidden = atomic.torch.zeros(batch, segments, 16)
    parent = atomic.torch.arange(segments)
    children = atomic.torch.arange(segments)[:, None]
    mask = atomic.torch.zeros(segments, 1)
    region = atomic.torch.zeros(segments, dtype=atomic.torch.long)
    alpha, next_hidden = models[0](
        features,
        region,
        hidden,
        parent,
        children,
        mask,
        recurrent=True,
        topology="local",
    )
    assert alpha.shape == (batch, segments)
    assert next_hidden.shape == hidden.shape
    assert atomic.torch.all((0 <= alpha) & (alpha <= 1))


def test_06bm_teacher_future_is_loss_only_and_updates_are_synchronized():
    unroll = inspect.getsource(ContinuousMixtureStatePlayground._mixture_unroll)
    train = inspect.getsource(
        ContinuousMixtureStatePlayground.train_synchronized_mixture_matrix
    )
    assert "target_voltage" in unroll and "oracle_alpha" in unroll
    assert "features = self._features" in unroll
    feature_prefix = unroll[: unroll.index("target_voltage =")]
    assert 'batch["voltage_t1"]' not in feature_prefix
    synchronized_steps = train[train.index("for step in range") :]
    assert "for (arm, width), model in models.items()" in synchronized_steps
    assert synchronized_steps.index("rows = rng.choice") < synchronized_steps.index(
        "for (arm, width), model in models.items()"
    )
    assert '"development_used_during_training": False' in train


def test_06bm_finalization_has_hierarchical_non_test_decision():
    source = inspect.getsource(
        ContinuousMixtureStatePlayground.finalize_continuous_mixture_state
    )
    assert source.index("if primary_pass") < source.index("elif local_pass")
    assert source.index("elif local_pass") < source.index("elif instantaneous_pass")
    assert '"validation_state_accessed": False' in source
    assert '"test_state_accessed": False' in source
    assert '"coupled_06c_canary_authorized": False' in source
    assert '"full_training_authorized": False' in source


def test_06bm_notebook_is_compact_and_uses_stable_blob_download():
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
    assert "EXPECTED_06BL_INDEX_SHA256" in code
    assert "train_synchronized_mixture_matrix" in code
    assert "evaluate_mixture_matrix" in code
    assert "base64.b64encode" in code and "new Blob" in code
    assert "FileLink" not in code
