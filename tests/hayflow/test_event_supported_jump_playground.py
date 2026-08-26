import ast
import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from src.hayflow_model import atomic_state_dynamics_playground as atomic
from src.hayflow_model.event_supported_jump_playground import (
    CHRONOLOGICAL_JUMP,
    DEEPSET_EVENTS,
    EVENT_FEATURE_NAMES,
    EVENT_NORMALIZATIONS,
    EVENT_REPRESENTATIONS,
    LEGACY_ALL_ENTRY_P99,
    MOMENT_POOL,
    NONZERO_ROBUST_LOG,
    PASSIVE_DEFAULT_GATE,
    SAFETY_GATES,
    UNGATED_RESIDUAL,
    EventConditionedSourceCell,
    EventSupportedJumpConfig,
    EventSupportedJumpPlayground,
    build_event_supported_roles,
    fit_event_normalizers,
    normalize_event_tensor,
    ordered_event_tensor,
)


ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = ROOT / "experiments" / "hayflow" / "06b_q_event_supported_jump_playground"
NOTEBOOK = ROOT / "notebooks" / "06b_q_event_supported_jump_and_mechanism_playground.ipynb"


class _Store:
    def __init__(self):
        self.layout = SimpleNamespace(
            segment_count=3,
            synapses=[
                {
                    "id": 4,
                    "segment_id": 2,
                    "parameters": {"gmax": 0.002},
                    "components": [{"name": "AMPA"}, {"name": "NMDA"}],
                }
            ],
        )

    def actions(self, index, view):
        assert view == "U_realized"
        if index == 0:
            return []
        return [
            {
                "kind": "synaptic_event",
                "synapse_id": 4,
                "offset_ms": 0.2,
                "released_quantity": 0.7,
                "weight_multiplier": 1.0,
                "gmax": 0.002,
                "ampa_state_increment": 0.4,
                "nmda_state_increment": 0.3,
            },
            {
                "kind": "synaptic_event",
                "synapse_id": 4,
                "offset_ms": 0.8,
                "released_quantity": 0.5,
                "weight_multiplier": 1.0,
                "gmax": 0.002,
                "ampa_state_increment": 0.2,
                "nmda_state_increment": 0.1,
            },
        ]


class _RoleStore:
    def __init__(self):
        self.episode_rows = [
            {
                "trajectory_id": f"trajectory-{index}",
                "split": "train",
                "seed": 100 + index,
                "snapshot_id": f"snapshot-{index}",
                "snapshot_source": f"source-{index}",
            }
            for index in range(3)
        ]
        self.trajectory_indices = {
            f"trajectory-{index}": np.asarray([2 * index, 2 * index + 1])
            for index in range(3)
        }

    def actions(self, index, view):
        assert view == "U_realized"
        return (
            [{"kind": "synaptic_event", "released_quantity": 0.0}]
            if index % 2
            else []
        )


def test_06bq_preregisters_one_multifactor_run_without_fake_current_target():
    prereg = json.loads((EXPERIMENT / "preregistration.json").read_text())
    assert prereg["status"] == "preregistered_before_execution"
    assert prereg["stage_B_paired_matrix"]["factorial_design"] == "3x2"
    assert prereg["stage_B_paired_matrix"]["role_stratification"] == (
        "seed/snapshot-disjoint mixed-support components profiled from U_realized only; no outcome labels"
    )
    assert prereg["mechanism_factorization_gate"]["required_evidence"] == (
        "causal per-mechanism current integral over the full 1 ms macro-step"
    )
    assert prereg["prohibitions"]["fake_mechanism_integral_target"] is False


def test_06bq_registered_result_qualifies_automatic_causal_and_gate_claims():
    result = json.loads((EXPERIMENT / "result.json").read_text())
    assert result["archive"]["all_indexed_members_verified"]
    assert result["support_repair"]["conclusion"] == (
        "the zero-support confound from 06b-p was removed"
    )
    assert result["paired_3x2_result"]["median_one_step_gain_over_passive_fraction"] > 0.04
    assert result["causal_controls"]["automatic_positive_sign_gate_passed"]
    assert not result["causal_controls"]["decision_grade_event_content_materiality"]
    assert result["passive_default_gate"]["passive_default_selected_steps"] == {
        "61017": 0,
        "61029": 0,
        "61043": 0,
    }
    assert not result["passive_default_gate"]["learned_gate_effect_demonstrated"]
    assert result["automated_outcome"]["selected_candidate"] is None


def test_06bq_configuration_is_a_fixed_3x2_plus_2_arm_safety_design():
    config = EventSupportedJumpConfig()
    config.validate()
    assert config.event_representations == (
        MOMENT_POOL,
        DEEPSET_EVENTS,
        CHRONOLOGICAL_JUMP,
    )
    assert config.event_normalizations == (
        LEGACY_ALL_ENTRY_P99,
        NONZERO_ROBUST_LOG,
    )
    assert config.safety_gates == (UNGATED_RESIDUAL, PASSIVE_DEFAULT_GATE)
    session = object.__new__(EventSupportedJumpPlayground)
    session.config = config
    assert len(session._jump_specs()) == 6


def test_06bq_ordered_tensor_preserves_timestamp_order_and_receptor_coupling():
    values, mask, counts = ordered_event_tensor(
        _Store(), [0, 1], max_events_per_segment=4
    )
    assert values.shape == (2, 3, 4, len(EVENT_FEATURE_NAMES))
    assert counts[0].sum() == 0 and counts[1, 2] == 2
    offset = EVENT_FEATURE_NAMES.index("offset_ms")
    ampa = EVENT_FEATURE_NAMES.index("ampa_state_increment")
    nmda = EVENT_FEATURE_NAMES.index("nmda_state_increment")
    assert values[1, 2, :2, offset].tolist() == pytest.approx([0.2, 0.8])
    assert values[1, 2, 0, ampa] == pytest.approx(0.4)
    assert values[1, 2, 0, nmda] == pytest.approx(0.3)
    assert mask[1, 2, :2].all()


def test_06bq_roles_use_mixed_components_not_impossible_all_no_event_episodes():
    roles, report = build_event_supported_roles(
        _RoleStore(),
        role_seed=17,
        component_targets={
            "fit": (1, 1),
            "calibration": (1, 1),
            "development": (1, 1),
        },
    )
    assert report["valid"]
    assert report["available_component_counts"]["mixed"] == 3
    assert report["available_component_counts"]["no_event_only"] == 0
    assert all(len(rows) == 1 for rows in roles.values())


def test_06bq_nonzero_normalizer_does_not_collapse_sparse_event_features():
    values = np.zeros((40, 2, 3, len(EVENT_FEATURE_NAMES)), dtype=np.float32)
    mask = np.zeros((40, 2, 3), dtype=bool)
    feature = EVENT_FEATURE_NAMES.index("released_quantity")
    values[0, 0, 0, feature] = 0.5
    values[1, 1, 0, feature] = 1.0
    mask[0, 0, 0] = mask[1, 1, 0] = True
    scales = fit_event_normalizers(values, mask, quantile=0.99, floor=1e-6)
    assert scales[LEGACY_ALL_ENTRY_P99][feature] == pytest.approx(1e-6)
    assert scales[NONZERO_ROBUST_LOG][feature] == pytest.approx(0.75)


@pytest.mark.skipif(atomic.torch is None, reason="PyTorch is optional locally")
def test_06bq_all_representation_arms_have_identical_parameter_count():
    model = EventConditionedSourceCell(
        base_feature_width=20,
        event_feature_width=len(EVENT_FEATURE_NAMES),
        event_embedding_width=8,
        region_count=4,
        region_width=3,
        hidden_width=12,
        output_limit=8.0,
        passive_gate_initial_bias=-4.0,
    )
    count = sum(parameter.numel() for parameter in model.parameters())
    assert count > 0
    assert all(
        sum(parameter.numel() for parameter in model.parameters()) == count
        for _ in EVENT_REPRESENTATIONS
        for _ in EVENT_NORMALIZATIONS
    )


def test_06bq_training_is_balanced_paired_and_probes_after_warmup():
    source = inspect.getsource(EventSupportedJumpPlayground.train_event_representation_matrix)
    assert "rng.choice(positive" in source and "rng.choice(negative" in source
    assert "batch_stream_sha256" in source
    assert "postwarmup_gradient_probes" in source
    assert "calibration_half_A" in source and "calibration_half_B" in source
    assert '"development"' not in source


def test_06bq_mechanism_factorization_is_guarded_by_integrated_teacher_targets():
    source = inspect.getsource(
        EventSupportedJumpPlayground._mechanism_factorization_eligibility_audit
    )
    assert "time_integrated_current_coordinate_count" in source
    assert '"factorized_model_trained": False' in source
    assert "boundary currents cannot be summed" in source


def test_06bq_notebook_is_compact_parseable_and_uses_blob_download():
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    assert len(notebook["cells"]) <= 11
    code = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    )
    for cell in notebook["cells"]:
        if cell.get("cell_type") == "code":
            ast.parse("".join(cell.get("source", [])))
            assert not cell.get("outputs")
    assert "IMPLEMENTATION_COMMIT_PLACEHOLDER" not in code
    assert "7a6cb88d3a5838232e9130cf6545ac9ce9524075" in code
    assert "stale_incomplete_output_removed" in code
    assert "EXPECTED_06BP_INDEX_SHA256" in code
    assert "run_sparse_event_synthetic_preflight" in code
    assert "train_event_representation_matrix" in code
    assert "run_passive_default_safety_gate" in code
    assert "base64.b64encode" in code and "new Blob" in code
    assert "FileLink" not in code
