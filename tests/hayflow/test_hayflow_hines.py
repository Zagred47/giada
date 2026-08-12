from types import SimpleNamespace
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import src.hayflow_model.hines_isolation_experiment as isolation_module
import src.hayflow_model.hines_conditioning_experiment as conditioning_module

from src.hayflow_data.hines_inputs import (
    canonical_anchor_segment_ids,
    encode_realized_synaptic_drive,
    explicit_teacher_views,
)
from src.hayflow_model.hayflow_hines import (
    HINES_SYNAPTIC_FEATURE_NAMES,
    HayFlowHinesConfig,
    SYNAPTIC_STATISTICS,
)
from src.hayflow_model.hines_experiment import (
    HayFlowHinesExperiment,
    HinesPrototypeExperimentConfig,
)
from src.hayflow_model.hines_isolation_experiment import (
    EXPECTED_05B_ARCHIVE_SHA256,
    HinesCausalIsolationExperiment,
    HinesIsolationConfig,
)
from src.hayflow_model.hines_conditioning_experiment import (
    EXPECTED_05C_ARCHIVE_SHA256,
    HinesConditioningConfig,
    HinesResidualConditioningExperiment,
)


def test_05b_notebook_uses_composite_bundle_public_api():
    root = Path(__file__).resolve().parents[2]
    notebook = json.loads(
        (root / "notebooks/05b_hayflow_hines_canary_revision.ipynb").read_text(
            encoding="utf-8"
        )
    )
    source = "\n".join(
        "".join(cell.get("source", [])) for cell in notebook["cells"]
    )
    assert "bundle.report" not in source
    assert "bundle.manifest" in source
    assert "bundle.transition_count" in source


def test_05b_experimental_record_is_hashed_and_keeps_full_training_blocked():
    root = Path(__file__).resolve().parents[2]
    record = json.loads(
        (
            root
            / "experiments/hayflow/05b_hayflow_hines_canary_v2/result.json"
        ).read_text(encoding="utf-8")
    )
    assert record["decision"] == "NO_GO_FULL_TRAINING"
    assert not record["next_step"]["full_training_authorized"]
    assert record["provenance"]["dataset_transition_count"] == 29880
    assert len(record["artifact"]["sha256"]) == 64
    assert len(record["artifact_members"]["checkpoints/canary_models.pt"]["sha256"]) == 64
    assert not record["models"]["HayFlow-Hines-H2"]["passed"]


def test_05c_experimental_record_preserves_the_diagnostic_no_go():
    root = Path(__file__).resolve().parents[2]
    record = json.loads(
        (
            root
            / "experiments/hayflow/05c_hayflow_hines_causal_isolation/result.json"
        ).read_text(encoding="utf-8")
    )
    assert record["status"] == "complete"
    assert record["decision"] == "DIAGNOSTIC_ONLY_NO_FULL_TRAINING"
    assert record["diagnosis"] == "ENCODER_OR_OPTIMIZATION_BOTTLENECK"
    assert record["artifact"]["sha256"] == (
        "b6a2e222529fd293fd75602bdac3b0feca8729832f371fadd24c9aa3b96b0d70"
    )
    assert record["artifact_integrity"]["all_indexed_members_verified"]
    assert not record["progressive_micro_overfit"]["timed_one_transition_passed"]
    assert not record["progressive_micro_overfit"]["direct_one_transition_passed"]
    assert not record["next_step"]["full_training_authorized"]


def test_05d_experimental_record_scopes_the_representation_no_go():
    root = Path(__file__).resolve().parents[2]
    record = json.loads(
        (
            root
            / "experiments/hayflow/05d_hayflow_hines_residual_conditioning/result.json"
        ).read_text(encoding="utf-8")
    )
    assert record["status"] == "complete"
    assert record["decision"] == "DIAGNOSTIC_ONLY_NO_FULL_TRAINING"
    assert record["diagnosis"] == "SHARED_REPRESENTATION_BOTTLENECK"
    assert record["artifact"]["sha256"] == (
        "61bf814a22c313093c18a7a555d76515d3d3db741eebae167b8ff479b8d7309c"
    )
    assert record["artifact_integrity"]["all_indexed_members_verified"]
    assert record["free_residual_control"]["passed"]
    assert record["frozen_decoder_sweep"]["passed_run_count"] == 0
    assert not record["interpretation"]["architecture_family_proven_impossible"]
    assert not record["next_step"]["full_training_authorized"]


def test_05c_notebook_has_no_full_training_path_and_uses_public_bundle_api():
    root = Path(__file__).resolve().parents[2]
    notebook = json.loads(
        (root / "notebooks/05c_hayflow_hines_causal_isolation.ipynb").read_text(
            encoding="utf-8"
        )
    )
    source = "\n".join(
        "".join(cell.get("source", [])) for cell in notebook["cells"]
    )
    assert "bundle.report" not in source
    assert "bundle.manifest" in source
    assert "session.run_full" not in source
    assert "assert not final_report['full_training_authorized']" in source
    assert "hayflow_hines_canary_v2.zip" in source


def test_05d_notebook_is_gated_and_uses_the_browser_blob_download():
    root = Path(__file__).resolve().parents[2]
    notebook = json.loads(
        (
            root / "notebooks/05d_hayflow_hines_residual_conditioning.ipynb"
        ).read_text(encoding="utf-8")
    )
    source = "\n".join(
        "".join(cell.get("source", [])) for cell in notebook["cells"]
    )
    assert "bundle.report" not in source
    assert "bundle.manifest" in source
    assert "RUN_NEURAL_LADDER = bool(free_report['passed'])" in source
    assert "session.run_full" not in source
    assert "assert not final_report['full_training_authorized']" in source
    assert "hayflow_hines_causal_isolation.zip" in source
    assert "base64.b64encode" in source
    assert "new Blob" in source
    assert "FileLink" not in source


def test_isolation_config_is_nested_and_binds_the_05b_archive():
    config = HinesIsolationConfig()
    config.validate()
    assert config.subset_sizes == (1, 8, 32, 76)
    assert config.modes == ("timed_masked", "direct_residual")
    assert len(EXPECTED_05B_ARCHIVE_SHA256) == 64
    with pytest.raises(ValueError, match="increasing"):
        HinesIsolationConfig(subset_sizes=(8, 1), subset_epochs=(1, 1)).validate()
    with pytest.raises(ValueError, match="increasing"):
        HinesIsolationConfig(subset_sizes=(1, 1), subset_epochs=(1, 1)).validate()
    with pytest.raises(ValueError, match="non-empty and unique"):
        HinesIsolationConfig(modes=()).validate()


def test_05d_conditioning_config_preserves_the_preregistered_ladder():
    config = HinesConditioningConfig()
    config.validate()
    assert config.unfreezing_stages == (
        "head_only", "local_features", "base_dynamics"
    )
    assert config.decoder_parameterizations == (
        "linear", "scaled_linear", "tanh"
    )
    assert len(EXPECTED_05C_ARCHIVE_SHA256) == 64
    with pytest.raises(ValueError, match="preregistered order"):
        HinesConditioningConfig(
            unfreezing_stages=("base_dynamics", "head_only")
        ).validate()


def test_05c_accepts_a_hash_verified_kaggle_extracted_artifact(
    tmp_path, monkeypatch
):
    report = b'{"scenario":"C_NEITHER_MODEL_OVERFITS_CANARY"}'
    checkpoint = b"synthetic-checkpoint"
    expected = {
        "canary_overfit_report.json": hashlib.sha256(report).hexdigest(),
        "checkpoints/canary_models.pt": hashlib.sha256(checkpoint).hexdigest(),
    }
    monkeypatch.setattr(isolation_module, "EXPECTED_05B_MEMBER_SHA256", expected)
    root = tmp_path / "hayflow_hines_canary_v2"
    (root / "checkpoints").mkdir(parents=True)
    (root / "canary_overfit_report.json").write_bytes(report)
    (root / "checkpoints/canary_models.pt").write_bytes(checkpoint)
    experiment = HinesCausalIsolationExperiment.__new__(
        HinesCausalIsolationExperiment
    )
    experiment.checkpoint_source = root
    payload, contract = experiment._read_05b_source()
    assert payload == checkpoint
    assert contract["source_kind"] == "kaggle_extracted_directory"
    assert contract["archive_sha256"] is None
    assert contract["verified_member_sha256"] == expected


def test_05d_accepts_a_hash_verified_kaggle_extracted_05c_artifact(
    tmp_path, monkeypatch
):
    final = json.dumps({
        "diagnosis": "ENCODER_OR_OPTIMIZATION_BOTTLENECK",
        "full_training_authorized": False,
    }).encode()
    forensics = b"forensics"
    expected = {
        "final_report.json": hashlib.sha256(final).hexdigest(),
        "checkpoint_forensics.json": hashlib.sha256(forensics).hexdigest(),
    }
    monkeypatch.setattr(conditioning_module, "EXPECTED_05C_MEMBER_SHA256", expected)
    root = tmp_path / "hayflow_hines_causal_isolation"
    root.mkdir()
    (root / "final_report.json").write_bytes(final)
    (root / "checkpoint_forensics.json").write_bytes(forensics)
    experiment = HinesResidualConditioningExperiment.__new__(
        HinesResidualConditioningExperiment
    )
    experiment.artifact_05c_source = root
    report, contract = experiment._read_05c_source()
    assert report["diagnosis"] == "ENCODER_OR_OPTIMIZATION_BOTTLENECK"
    assert contract["source_kind"] == "kaggle_extracted_directory"
    assert contract["archive_sha256"] is None
    assert contract["verified_member_sha256"] == expected


def test_hines_config_smoke_profile_is_explicitly_non_decisional():
    config = HinesPrototypeExperimentConfig(
        profile="smoke", model=HayFlowHinesConfig(local_latent_dim=8)
    ).effective()
    assert config.seeds == (17,)
    assert config.canary_epochs == 5
    assert config.canary_voltage_epochs == 2
    assert config.canary_event_epochs == 2
    assert config.canary_joint_epochs == 1
    assert config.rollout_horizons_ms == (2, 4)
    assert config.model.local_latent_dim == 8


def test_canary_checkpoint_score_tracks_acceptance_metrics_not_training_loss():
    experiment = object.__new__(HayFlowHinesExperiment)
    experiment.config = HinesPrototypeExperimentConfig()
    poor = {
        "voltage_rmse_mv": 6.0,
        "maximum_peak_error_mv": 50.0,
        "minimum_present_event_f1": 0.8,
        "branching_retention": 0.2,
    }
    good = {
        "voltage_rmse_mv": 0.8,
        "maximum_peak_error_mv": 4.0,
        "minimum_present_event_f1": 0.95,
        "branching_retention": 0.6,
    }
    assert experiment._canary_selection_score(good) < experiment._canary_selection_score(poor)


def test_explicit_teacher_views_scatter_by_segment_and_component():
    state = np.asarray([[1.0, 2.0, 3.0, 4.0, 5.0]], dtype=np.float32)
    arrays = {
        "concentration_indices": np.asarray([0, 1]),
        "concentration_segment_ids": np.asarray([0, 0]),
        "synapse_indices": np.asarray([2, 3, 4]),
        "synapse_segment_ids": np.asarray([1, 1, 1]),
        "synapse_channels": np.asarray([0, 1, 1]),
    }
    calcium, synapse = explicit_teacher_views(
        state, arrays, segment_count=2, calcium_dim=1, synapse_dim=4
    )
    assert calcium.shape == (1, 2, 1)
    assert calcium[0, 0, 0] == 1.5
    assert synapse[0, 1, 0] == 3.0
    assert synapse[0, 1, 1] == 4.5


def test_realized_drive_keeps_ampa_and_nmda_separate_and_uses_release_result():
    synapse = {
        "id": 0,
        "segment_id": 1,
        "parameters": {"gmax": 0.001},
        "components": [
            {
                "name": "AMPA", "tau_rise_ms": 0.3, "tau_decay_ms": 3.0,
                "reversal_mv": 0.0, "voltage_dependent": False,
            },
            {
                "name": "NMDA", "tau_rise_ms": 2.0, "tau_decay_ms": 70.0,
                "reversal_mv": 0.0, "voltage_dependent": True,
                "magnesium_alpha": 0.08, "magnesium_beta": 3.57,
            },
        ],
    }
    layout = SimpleNamespace(segment_count=2, synapses=[synapse])
    successful = {
        "kind": "synaptic_event", "synapse_id": 0, "offset_ms": 0.25,
        "released_quantity": 1.0, "ampa_state_increment": 1.4,
        "nmda_state_increment": 1.1, "inhibitory_state_increment": 0.0,
    }
    failed = {
        **successful,
        "offset_ms": 0.50,
        "released_quantity": 0.0,
        "ampa_state_increment": 0.0,
        "nmda_state_increment": 0.0,
    }
    store = SimpleNamespace(
        layout=layout,
        actions=lambda index, view: [successful, failed],
    )
    encoded = encode_realized_synaptic_drive(
        store, [0], np.asarray([[-76.0, -65.0]], dtype=np.float32)
    )
    width = len(SYNAPTIC_STATISTICS)
    features = encoded["synaptic_features"][0, 1]
    assert features[0] > 0.0  # AMPA increment
    assert features[width] > 0.0  # NMDA increment
    assert features[3] == 2.0  # both ordered events retained
    assert features[4] == 1.0  # only one release succeeded
    assert encoded["synaptic_conductance_us"][0, 1] > 0.0
    assert len(HINES_SYNAPTIC_FEATURE_NAMES) == 42


def test_anchor_selection_is_stable_and_has_five_entries():
    layout = SimpleNamespace(
        segments=[
            {"region": "soma"}, {"region": "ais"}, {"region": "nexus"},
            {"region": "tuft"}, {"region": "basal"},
        ]
    )
    np.testing.assert_array_equal(canonical_anchor_segment_ids(layout), [0, 1, 2, 3, 4])


def test_training_counterfactual_search_accepts_equal_state_across_snapshot_ids():
    states = np.asarray(
        [[-76.0, -75.0, 0.1], [-76.0, -75.0, 0.1]], dtype=np.float32
    )
    actions = {
        0: [{"synapse_id": 4, "released_quantity": 0.0}],
        1: [{"synapse_id": 4, "released_quantity": 1.0}],
    }
    trajectories = {
        "train-left": np.asarray([0], dtype=np.int64),
        "train-right": np.asarray([1], dtype=np.int64),
    }
    store = SimpleNamespace(
        metadata={
            "trajectory_id": np.asarray(["train-left", "train-right"]),
            "split": np.asarray(["train", "train"]),
            "step_index": np.asarray([0, 0]),
        },
        trajectory_indices=trajectories,
        episode_by_trajectory={
            "train-left": {"snapshot_id": "snapshot-a"},
            "train-right": {"snapshot_id": "snapshot-b"},
        },
        episode_indices=lambda split: list(trajectories.values()) if split == "train" else [],
        read_state=lambda indices, boundary: states[np.asarray(indices, dtype=np.int64)],
        actions=lambda index, view: actions[int(index)],
    )
    experiment = object.__new__(HayFlowHinesExperiment)
    experiment.store = store
    experiment.layout = SimpleNamespace(segment_count=2)
    assert experiment._find_counterfactual_pair(("train",)) == (0, 1)


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="PyTorch is not installed locally")
def test_hayflow_hines_and_convgru_support_real_forward_and_backward():
    import torch

    from src.hayflow_model.hayflow_hines import HayFlowHines, OrderedSegmentConvGRU

    segment_count = 6
    state_width = 12
    parent = np.asarray([0, 0, 0, 1, 1, 2], dtype=np.int64)
    child_ids = np.asarray(
        [[1, 2], [3, 4], [5, 2], [3, 3], [4, 4], [5, 5]], dtype=np.int64
    )
    child_mask = np.asarray(
        [[1, 1], [1, 1], [1, 0], [0, 0], [0, 0], [0, 0]], dtype=np.float32
    )
    axial = np.asarray([0.0, 0.2, 0.15, 0.1, 0.1, 0.08], dtype=np.float32)
    child_axial = np.zeros_like(child_mask)
    for node in range(segment_count):
        for position in range(child_ids.shape[1]):
            if child_mask[node, position]:
                child_axial[node, position] = axial[child_ids[node, position]]
    axial_total = axial.copy()
    for node, owner in enumerate(parent):
        if node != owner:
            axial_total[owner] += axial[node]
    arrays = {
        "parent_ids": parent,
        "child_ids": child_ids,
        "child_mask": child_mask,
        "child_axial_us": child_axial,
        "segment_static": np.zeros((segment_count, 7), dtype=np.float32),
        "segment_region_ids": np.asarray([0, 1, 2, 2, 3, 4]),
        "capacitance_uf": np.full(segment_count, 0.001, dtype=np.float32),
        "leak_conductance_us": np.full(segment_count, 0.01, dtype=np.float32),
        "leak_reversal_mv": np.full(segment_count, -70.0, dtype=np.float32),
        "axial_conductance_to_parent_us": axial,
        "axial_total_us": axial_total,
        "core_segment_ids": np.arange(state_width) % segment_count,
        "core_category_ids": np.arange(state_width) % 2,
        "core_mechanism_ids": np.arange(state_width) % 3,
        "core_variable_ids": np.arange(state_width) % 4,
        "core_kind_ids": np.arange(state_width) % 2,
        "selected_core_indices": np.asarray([0, 1, 2, 3]),
        "selected_core_segment_ids": np.asarray([0, 1, 2, 3]),
        "selected_core_mechanism_ids": np.asarray([0, 1, 2, 0]),
        "selected_core_variable_ids": np.asarray([0, 1, 2, 3]),
        "selected_core_kind_ids": np.asarray([0, 1, 0, 1]),
        "selected_privileged_indices": np.asarray([0, 1]),
        "selected_privileged_segment_ids": np.asarray([4, 5]),
        "selected_privileged_mechanism_ids": np.asarray([1, 2]),
        "selected_privileged_variable_ids": np.asarray([1, 3]),
        "selected_privileged_kind_ids": np.asarray([0, 1]),
        "event_allowed_mask": np.ones((6, segment_count), dtype=np.float32),
        "mechanism_names": np.asarray(["m0", "m1", "m2"], dtype=object),
        "variable_names": np.asarray(["v0", "v1", "v2", "v3"], dtype=object),
        "kind_names": np.asarray(["k0", "k1"], dtype=object),
    }
    metadata = {
        "segment_count": segment_count,
        "state_width": state_width,
        "region_names": ["soma", "ais", "basal", "trunk", "tuft"],
    }
    config = HayFlowHinesConfig(
        local_latent_dim=8, global_latent_dim=16, hidden_width=32,
        synaptic_hidden_width=12, residual_blocks=1,
    )
    batch_size = 2
    batch = {
        "teacher_state_t": torch.randn(batch_size, state_width),
        "voltage_t": torch.full((batch_size, segment_count), -70.0),
        "calcium_t": torch.zeros(batch_size, segment_count, 1),
        "synapse_state_t": torch.zeros(batch_size, segment_count, 4),
        "anchor_voltage_t": torch.full((batch_size, 5), -70.0),
        "anchor_segment_ids": torch.tensor([0, 1, 2, 3, 4]),
        "synaptic_features": torch.randn(
            batch_size, segment_count, len(HINES_SYNAPTIC_FEATURE_NAMES)
        ) * 0.01,
        "synaptic_conductance_us": torch.full((batch_size, segment_count), 0.001),
        "synaptic_source_na": torch.zeros(batch_size, segment_count),
        "somatic_current_na": torch.zeros(batch_size, segment_count),
    }
    model = HayFlowHines(config, metadata, arrays)
    output = model(batch, ablation="H2", decode_teacher=True)
    assert output["voltage"].shape == (batch_size, segment_count)
    assert output["event_logits"].shape == (batch_size, 6)
    assert output["event_segment_logits"].shape == (batch_size, 6, segment_count)
    assert output["event_boundary_delta_mv"].shape == (batch_size, 6)
    assert output["event_boundary_raw_delta_mv"].shape == (batch_size, 6)
    assert output["boundary_features"].shape == (
        batch_size, segment_count, config.hidden_width
    )
    expected_presence = torch.sigmoid(output["event_logits"])
    torch.testing.assert_close(
        output["event_local_gate"].amax(1), expected_presence,
        atol=1e-5, rtol=1e-5,
    )
    objective = (
        output["voltage"].mean() + output["event_logits"].mean()
        + output["selected_state"].mean() + output["selected_privileged"].mean()
        + output["probe_microtrace"].mean()
    )
    objective.backward()
    assert model.effective_conductance.weight.grad is not None
    assert torch.isfinite(model.effective_conductance.weight.grad).all()
    no_jump = model(
        batch, ablation="H2", decode_teacher=False,
        boundary_mode="no_event_jump",
    )
    assert torch.count_nonzero(no_jump["event_jump"]) == 0
    direct = model(
        batch, ablation="H2", decode_teacher=False,
        boundary_mode="direct_residual",
    )
    torch.testing.assert_close(
        direct["event_jump"], direct["direct_boundary_residual"]
    )
    with pytest.raises(ValueError, match="unknown boundary_mode"):
        model(batch, boundary_mode="invalid")
    from src.hayflow_model.hines_conditioning_experiment import (
        ZeroInitializedBoundaryDecoder,
    )
    for parameterization in ("linear", "scaled_linear", "tanh"):
        decoder = ZeroInitializedBoundaryDecoder(
            config.hidden_width,
            parameterization,
            scaled_linear_factor_mv=10.0,
            tanh_limit_mv=120.0,
        )
        residual, raw = decoder(output["boundary_features"].detach())
        assert torch.count_nonzero(residual) == 0
        assert torch.count_nonzero(raw) == 0
    target_residual = torch.linspace(-300.0, 300.0, segment_count).view(1, -1)
    free_residual = torch.nn.Parameter(torch.zeros_like(target_residual))
    optimizer = torch.optim.SGD([free_residual], lr=1.0)
    optimizer.zero_grad()
    (0.5 * torch.sum((free_residual - target_residual).square())).backward()
    optimizer.step()
    torch.testing.assert_close(free_residual, target_residual)
    control = OrderedSegmentConvGRU(config, metadata, arrays)
    control_output = control(batch)
    (control_output["voltage"].mean() + control_output["event_logits"].mean()).backward()
    assert control.input.weight.grad is not None
    with torch.no_grad():
        control.voltage[-1].weight.zero_()
        control.voltage[-1].bias.fill_(10.0)
        unconstrained_spike = control(batch)["voltage"] - batch["voltage_t"]
    assert float(unconstrained_spike.max()) > 100.0
