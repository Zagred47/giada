import ast
import inspect
import json
from pathlib import Path

import pytest

from src.hayflow_model import atomic_state_dynamics_playground as atomic
from src.hayflow_model.structure_preserving_coupling_forensic import (
    ENDPOINT_MIXED_STATE,
    EXPECTED_06BM_ARCHIVE_SHA256,
    EXPECTED_06BM_FINAL_SHA256,
    EXPECTED_06BM_INDEX_SHA256,
    GENERIC_STATE,
    PERSISTENCE_REGRET,
    PRE_MIXED_STATE,
    RELAXATION_STATE,
    STANDARD_ROLLOUT,
    BoundedRelaxationStateUpdater,
    StructurePreservingCouplingConfig,
    StructurePreservingCouplingForensic,
    audit_cnexp_teacher_contract,
)


ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_M = ROOT / "experiments" / "hayflow" / "06b_m_continuous_mixture_state_playground"
EXPERIMENT_N = ROOT / "experiments" / "hayflow" / "06b_n_structure_preserving_coupling_forensic"
NOTEBOOK = ROOT / "notebooks" / "06b_n_structure_preserving_coupling_forensic.ipynb"
TEACHER_MODS = ROOT.parent / "neuron_as_deep_net" / "L5PC_NEURON_simulation" / "mods"


def test_06bn_is_authorized_by_exact_registered_06bm_result():
    result = json.loads((EXPERIMENT_M / "result.json").read_text())
    assert result["archive_sha256"] == EXPECTED_06BM_ARCHIVE_SHA256
    assert result["artifact_index_sha256"] == EXPECTED_06BM_INDEX_SHA256
    assert result["final_report_sha256"] == EXPECTED_06BM_FINAL_SHA256
    assert result["formal_diagnosis"] == (
        "MIXTURE_TARGET_LEARNABLE_BUT_RECURSIVE_COMPOSITION_FAILS"
    )


def test_06bn_preregisters_bounded_four_axis_matrix():
    prereg = json.loads((EXPERIMENT_N / "preregistration.json").read_text())
    matrix = prereg["stage_3_factorial_matrix"]
    assert prereg["status"] == "preregistered_before_execution"
    assert matrix["arm_count"] == 16
    assert matrix["same_minibatches_within_seed"]
    assert matrix["same_initialization_except_registered_gate_bias"]
    assert not prereg["stage_1_frozen_counterfactuals"]["selection_eligible"]
    assert not prereg["stage_2_structure_probe"][
        "teacher_microtrace_upper_bound_selection_eligible"
    ]


def test_06bn_config_and_specs_cover_exact_2x2x2x2_matrix():
    config = StructurePreservingCouplingConfig()
    config.validate()
    session = object.__new__(StructurePreservingCouplingForensic)
    session.config = config
    specs = session._factor_specs()
    assert len(specs) == 16 and len(set(specs)) == 16
    assert {row[0] for row in specs} == {0.02, 0.5}
    assert {row[1] for row in specs} == {STANDARD_ROLLOUT, PERSISTENCE_REGRET}
    assert {row[2] for row in specs} == {PRE_MIXED_STATE, ENDPOINT_MIXED_STATE}
    assert {row[3] for row in specs} == {GENERIC_STATE, RELAXATION_STATE}
    assert config.factor_checkpoints == (0, 50, 100, 200)


def test_06bn_endpoint_mixture_evolves_both_complete_state_paths():
    source = inspect.getsource(StructurePreservingCouplingForensic._coupled_state_delta)
    endpoint = source[source.index("if coupling == ENDPOINT_MIXED_STATE") :]
    assert "zeros_like(baseline)" in endpoint
    assert "dynamic = self._state_delta" in endpoint
    assert "carry + coordinate_alpha * (dynamic - carry)" in endpoint


def test_06bn_regret_is_measured_against_recursive_persistence():
    source = inspect.getsource(StructurePreservingCouplingForensic._factor_unroll)
    assert "model_sq - persistence_sq" in source
    assert "quiescent_regret_multiplier" in source
    assert "current_voltage - target_voltage" in source


def test_06bn_path_resolution_and_semantic_negative_controls_are_present():
    frozen = inspect.getsource(
        StructurePreservingCouplingForensic.run_frozen_counterfactual_matrix
    )
    relaxation = inspect.getsource(
        StructurePreservingCouplingForensic._evaluate_relaxation
    )
    for name in (
        "causal_endpoint_2_support",
        "causal_coarse_4_support",
        "causal_linear_8_support",
        "teacher_microtrace_upper_bound",
    ):
        assert name in frozen
    assert "permutation" in relaxation and "shuffled" in relaxation


def test_06bn_path_boundary_forces_checkpoint_dtype():
    source = inspect.getsource(StructurePreservingCouplingForensic._state_forward_path)
    assert "dtype=normalized_state.dtype" in source
    frozen = inspect.getsource(
        StructurePreservingCouplingForensic.run_frozen_counterfactual_matrix
    )
    assert "dtype=current_voltage.dtype" in frozen


def test_06bn_cnexp_audit_is_honest_about_exact_execution():
    report = audit_cnexp_teacher_contract(TEACHER_MODS, ())
    assert report["cnexp_file_count"] >= 10
    assert not report["exact_cnexp_replay_executed"]
    assert not report["exact_cnexp_replay_eligible"]
    assert report["blockers"]


@pytest.mark.skipif(atomic.torch is None, reason="PyTorch is optional locally")
def test_06bn_relaxation_model_has_bounded_equilibrium_and_positive_rate():
    model = BoundedRelaxationStateUpdater(
        mechanism_count=3,
        variable_count=4,
        kind_count=2,
        region_count=3,
        static_width=5,
        drive_width=6,
        path_width=8,
        hidden_width=12,
        embedding_width=3,
    )
    count = 7
    equilibrium, rate = model(
        atomic.torch.randn(count, 8),
        atomic.torch.randn(count, 6),
        atomic.torch.randn(count, 5),
        atomic.torch.zeros(count, dtype=atomic.torch.long),
        atomic.torch.zeros(count, dtype=atomic.torch.long),
        atomic.torch.zeros(count, dtype=atomic.torch.long),
        atomic.torch.zeros(count, dtype=atomic.torch.long),
    )
    assert atomic.torch.all((equilibrium > 0) & (equilibrium < 1))
    assert atomic.torch.all(rate > 0)


def test_06bn_notebook_is_compact_parseable_and_uses_blob_download():
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
    assert "EXPECTED_06BM_INDEX_SHA256" in code
    assert "run_frozen_counterfactual_matrix" in code
    assert "train_relaxation_updaters" in code
    assert "train_synchronized_factorial_matrix" in code
    assert "base64.b64encode" in code and "new Blob" in code
    assert "FileLink" not in code
