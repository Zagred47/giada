from types import SimpleNamespace
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import src.hayflow_model.hines_isolation_experiment as isolation_module
import src.hayflow_model.hines_conditioning_experiment as conditioning_module
import src.hayflow_model.hines_capacity_experiment as capacity_module
import src.hayflow_model.hines_segment_canary_experiment as segment_canary_module
import src.hayflow_model.hines_optimization_audit as optimization_audit_module
import src.hayflow_model.hines_representation_forensics as representation_forensics_module
import src.hayflow_model.hines_state_normalization_repair as normalization_repair_module
import src.hayflow_model.hines_netcon_semantic_repair as netcon_repair_module
import src.hayflow_model.hines_synaptic_domain_repair as synaptic_domain_module
import src.hayflow_model.hines_repaired_representation_recheck as repaired_recheck_module
import src.hayflow_model.hines_repaired_representation_revision as repaired_revision_module
import src.hayflow_model.hines_spatial_support_revision as spatial_support_module
import src.hayflow_model.hines_trainable_topology_canary as topology_canary_module
import src.hayflow_model.hines_architecture_reassessment as reassessment_module

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
from src.hayflow_model.hines_capacity_experiment import (
    EXPECTED_05D_ARCHIVE_SHA256,
    HinesCapacityConfig,
    HinesSegmentCapacityExperiment,
    segment_conditioned_rank_path,
    solve_linear_probe,
)
from src.hayflow_model.hines_segment_canary_experiment import (
    EXPECTED_05E_ARCHIVE_SHA256,
    HinesSegmentCanaryConfig,
    HinesSegmentMicroCanaryExperiment,
)
from src.hayflow_model.hines_optimization_audit import (
    HinesOptimizationAuditConfig,
    HinesSegmentOptimizationAudit,
    bounded_segment_prediction,
    dual_ridge_segment_coefficients,
)
from src.hayflow_model.hines_representation_forensics import (
    HinesRepresentationForensics,
    HinesRepresentationForensicsConfig,
    local_linear_projection,
    robust_bounded_features,
)
from src.hayflow_model.hines_state_normalization_repair import (
    HinesStateNormalizationRepair,
    HinesStateNormalizationRepairConfig,
    semantic_state_scale_repair,
)
from src.hayflow_model.hines_netcon_semantic_repair import (
    HinesNetConSemanticRepair,
    HinesNetConSemanticRepairConfig,
    NetConSemanticStateEncoder,
    netcon_semantic_records,
)
from src.hayflow_model.hines_synaptic_domain_repair import (
    BoundedSynapticStateEncoder,
    HinesSynapticDomainRepairConfig,
)
from src.hayflow_model.hines_repaired_representation_recheck import (
    HinesRepairedRepresentationRecheckConfig,
    summarize_robust_family_gate,
)
from src.hayflow_model.hines_repaired_representation_revision import (
    HinesRepairedRepresentationRevisionConfig,
    bounded_target_decode,
    bounded_target_encode,
    dual_ridge_path_predict,
    pair_gate_selection_score,
    revised_feature_transform,
)
from src.hayflow_model.hines_spatial_support_revision import (
    HinesSpatialSupportRevisionConfig,
    apply_channel_pca,
    axial_tree_diffusion,
    deterministic_pair_folds,
    deterministic_pca_components,
    region_global_context,
)
from src.hayflow_model.hines_trainable_topology_canary import (
    HinesTrainableTopologyCanaryConfig,
    TrainableTopologyResidualHead,
    deterministic_stratified_pair_split,
)
from src.hayflow_model.hines_architecture_reassessment import (
    HinesArchitectureReassessmentConfig,
    apply_segment_affine,
    error_energy_concentration,
    fit_segment_affine_calibrator,
)


def test_05je_segment_affine_recovers_fit_only_bias_and_scale():
    rng = np.random.default_rng(91)
    prediction = rng.normal(size=(30, 4))
    slopes = np.asarray([0.5, 1.0, 1.5, 2.0])
    intercepts = np.asarray([-2.0, -1.0, 1.0, 3.0])
    target = prediction * slopes + intercepts
    fitted_slope, fitted_intercept = fit_segment_affine_calibrator(
        prediction, target, 1e-12
    )
    calibrated = apply_segment_affine(prediction, fitted_slope, fitted_intercept)
    np.testing.assert_allclose(calibrated, target, atol=1e-10)


def test_05je_error_energy_concentration_is_ranked_and_normalized():
    error = np.zeros((3, 10)); error[:, 7] = 4.0; error[:, 2] = 2.0
    report = error_energy_concentration(error, 0.10)
    assert report["top_segment_count"] == 1
    assert report["top_segment_ids"] == [7]
    assert report["top_segment_error_energy_fraction"] == pytest.approx(0.8)


def test_05je_config_is_frozen_to_registered_canary():
    HinesArchitectureReassessmentConfig().validate()
    with pytest.raises(ValueError, match="direct-tree"):
        HinesArchitectureReassessmentConfig(audited_family="ridge_corrected_tree").validate()
    with pytest.raises(ValueError, match="registered three seeds"):
        HinesArchitectureReassessmentConfig(seeds=(17, 29, 44)).validate()


def test_05je_checkpoint_reader_does_not_shadow_inherited_checkpoint_bytes():
    assert callable(reassessment_module.HinesArchitectureReassessment._read_05jd_checkpoint_bytes)
    assert "_checkpoint_bytes" not in reassessment_module.HinesArchitectureReassessment.__dict__


def test_05je_exact_05jd_hashes_match_registered_result():
    root = Path(__file__).resolve().parents[2]
    result = json.loads(
        (root / "experiments/hayflow/05j_d_trainable_topology_decoder_micro_canary/result.json")
        .read_text(encoding="utf-8")
    )
    assert result["archive"]["sha256"] == reassessment_module.EXPECTED_05JD_ARCHIVE_SHA256
    assert result["archive"]["artifact_index_sha256"] == reassessment_module.EXPECTED_05JD_INDEX_SHA256
    assert result["archive"]["final_report_sha256"] == reassessment_module.EXPECTED_05JD_FINAL_SHA256
    assert not result["trainable_topology_canary_passed"]


def test_05je_notebook_is_forensic_and_keeps_future_data_sealed():
    root = Path(__file__).resolve().parents[2]
    notebook = json.loads(
        (root / "notebooks/05j_e_architecture_reassessment.ipynb")
        .read_text(encoding="utf-8")
    )
    source = "\n".join(
        "".join(cell.get("source", [])) for cell in notebook["cells"]
    )
    assert "prepare_architecture_reassessment" in source
    assert "reconstruct_frozen_checkpoints" in source
    assert "run_error_anatomy" in source
    assert "run_bias_and_capacity_controls" in source
    assert "run_branch_identifiability_audit" in source
    assert "EXPECTED_05JD_INDEX_SHA256" in source
    assert "assert not final_report['methodology']['retraining_performed']" in source
    assert "assert not final_report['methodology']['development_used_for_model_selection']" in source
    assert "assert not final_report['methodology']['heldout_inputs_extracted']" in source
    assert "assert not final_report['methodology']['rollout_performed']" in source
    assert "base64.b64encode" in source and "application/zip" in source
    assert "run_rollout" not in source
    assert "run_trainable_topology_canary" not in source
    assert "rglob('/kaggle" not in source


def test_05jd_stratified_split_is_deterministic_complete_and_disjoint():
    counts = [6, 5, 4, 11, 11, 11]
    rows = [
        {"protocol_family": f"family-{family}"}
        for family, count in enumerate(counts)
        for _ in range(count)
    ]
    first = deterministic_stratified_pair_split(rows, 12)
    second = deterministic_stratified_pair_split(rows, 12)
    assert first == second
    assert len(first["fit_pair_positions"]) == 36
    assert len(first["calibration_pair_positions"]) == 12
    assert not set(first["fit_pair_positions"]) & set(first["calibration_pair_positions"])
    assert sorted(first["fit_pair_positions"] + first["calibration_pair_positions"]) == list(range(48))
    assert first["fit_family_count"] == first["calibration_family_count"] == 6


def test_05jd_zero_initialized_heads_preserve_registered_baselines():
    torch = pytest.importorskip("torch")
    model = TrainableTopologyResidualHead(9, 7, 12, 3, 120.0)
    features = torch.randn(4, 7, 9)
    direct = model(features)
    assert torch.count_nonzero(direct) == 0
    baseline = torch.linspace(-90.0, 90.0, 28).reshape(4, 7)
    corrected = model(features, baseline)
    torch.testing.assert_close(corrected, baseline, atol=2e-5, rtol=2e-6)


def test_05jd_config_requires_registered_partition_and_robust_seeds():
    HinesTrainableTopologyCanaryConfig().validate()
    with pytest.raises(ValueError, match="48-pair"):
        HinesTrainableTopologyCanaryConfig(fit_pair_count=35).validate()
    with pytest.raises(ValueError, match="distinct seeds"):
        HinesTrainableTopologyCanaryConfig(seeds=(17, 17, 29)).validate()


def test_05jd_exact_05jc_hashes_match_registered_result():
    root = Path(__file__).resolve().parents[2]
    result = json.loads(
        (root / "experiments/hayflow/05j_c_spatial_support_revision/result.json")
        .read_text(encoding="utf-8")
    )
    assert result["archive"]["sha256"] == topology_canary_module.EXPECTED_05JC_ARCHIVE_SHA256
    assert result["archive"]["artifact_index_sha256"] == topology_canary_module.EXPECTED_05JC_INDEX_SHA256
    assert result["archive"]["final_report_sha256"] == topology_canary_module.EXPECTED_05JC_FINAL_SHA256
    assert result["diagnosis"] == "NONLOCAL_CONTEXT_HELPS_BUT_MAPPING_REMAINS_BELOW_GATE"


def test_05jd_notebook_seals_development_heldout_and_rollout():
    root = Path(__file__).resolve().parents[2]
    notebook = json.loads(
        (root / "notebooks/05j_d_trainable_topology_decoder_micro_canary.ipynb")
        .read_text(encoding="utf-8")
    )
    source = "\n".join(
        "".join(cell.get("source", [])) for cell in notebook["cells"]
    )
    assert "prepare_trainable_topology_canary" in source
    assert "prepare_topology_canary_designs" in source
    assert "fit_fixed_tree_ridge_baseline" in source
    assert "run_trainable_topology_canary" in source
    assert "EXPECTED_05JC_INDEX_SHA256" in source
    assert "assert not design_report['development_used_for_checkpoint_selection']" in source
    assert "assert not canary_report['development_used_for_checkpoint_selection']" in source
    assert "assert not final_report['heldout_contract']['inputs_extracted']" in source
    assert "assert not final_report['methodology']['rollout_performed']" in source
    assert "base64.b64encode" in source and "application/zip" in source
    assert "run_rollout" not in source
    assert "rglob('/kaggle" not in source


def test_05jc_tree_diffusion_preserves_constants_and_spreads_delta():
    parents = np.asarray([0, 0, 1, 1])
    axial = np.asarray([0.0, 1.0, 2.0, 1.0])
    constant = np.ones((2, 4, 1))
    preserved = axial_tree_diffusion(constant, parents, axial, [0, 1, 2], 0.5)
    np.testing.assert_allclose(preserved, 1.0)
    delta = np.zeros((1, 4, 1)); delta[:, 3] = 1.0
    spread = axial_tree_diffusion(delta, parents, axial, [0, 1, 2], 0.5)
    assert spread.shape == (1, 4, 3)
    assert spread[0, 1, 1] > 0.0
    assert spread[0, 0, 2] > 0.0


def test_05jc_channel_pca_is_deterministic_and_finite():
    rng = np.random.default_rng(51)
    values = rng.normal(size=(8, 5, 6))
    mean_a, components_a = deterministic_pca_components(values, 3)
    mean_b, components_b = deterministic_pca_components(values, 3)
    np.testing.assert_allclose(mean_a, mean_b)
    np.testing.assert_allclose(components_a, components_b)
    projected = apply_channel_pca(values, mean_a, components_a)
    assert projected.shape == (8, 5, 3)
    assert np.isfinite(projected).all()


def test_05jc_region_context_broadcasts_all_region_summaries():
    values = np.asarray([[[1.0], [3.0], [2.0], [6.0]]])
    context = region_global_context(values, [0, 0, 1, 1])
    assert context.shape == (1, 4, 2)
    np.testing.assert_allclose(context[:, 0], context[:, 3])
    np.testing.assert_allclose(context[0, 0], [4 / np.sqrt(2), 8 / np.sqrt(2)])


def test_05jc_pair_folds_are_complete_and_disjoint():
    folds = deterministic_pair_folds(48, 6)
    assert all(len(fold) == 8 for fold in folds)
    np.testing.assert_array_equal(
        np.sort(np.concatenate(folds)), np.arange(48)
    )
    assert not any(set(left) & set(right) for i, left in enumerate(folds) for right in folds[i + 1:])


def test_05jc_config_rejects_nonexpanded_or_posthoc_contexts():
    HinesSpatialSupportRevisionConfig().validate()
    with pytest.raises(ValueError, match="materially larger"):
        HinesSpatialSupportRevisionConfig(minimum_expanded_pair_count=12).validate()
    with pytest.raises(ValueError, match="contexts"):
        HinesSpatialSupportRevisionConfig(contexts=("tree",)).validate()


def test_05jc_notebook_seals_heldout_and_uses_grouped_train_cv():
    root = Path(__file__).resolve().parents[2]
    notebook = json.loads(
        (root / "notebooks/05j_c_spatial_support_revision.ipynb")
        .read_text(encoding="utf-8")
    )
    source = "\n".join(
        "".join(cell.get("source", [])) for cell in notebook["cells"]
    )
    assert "prepare_spatial_support_revision" in source
    assert "build_expanded_train_support" in source
    assert "run_spatial_design_audit" in source
    assert "run_spatial_support_controls" in source
    assert "train_grouped_pair_cross_validation" in source
    assert "artifact_index_matches" in source
    assert "EXPECTED_05H_INDEX_SHA256" in source
    assert "EXPECTED_05JB_INDEX_SHA256" in source
    assert "assert not controls_report['development_used_for_selection']" in source
    assert "assert not final_report['heldout_contract']['inputs_extracted']" in source
    assert "assert not final_report['methodology']['rollout_performed']" in source
    assert "base64.b64encode" in source and "application/zip" in source
    assert "run_rollout" not in source
    assert "rglob('/kaggle" not in source


def test_05jc_exact_05jb_hashes_match_registered_result():
    root = Path(__file__).resolve().parents[2]
    result = json.loads(
        (
            root
            / "experiments/hayflow/05j_b_repaired_representation_revision/result.json"
        ).read_text(encoding="utf-8")
    )
    assert result["archive"]["sha256"] == spatial_support_module.EXPECTED_05JB_ARCHIVE_SHA256
    assert result["archive"]["artifact_index_sha256"] == spatial_support_module.EXPECTED_05JB_INDEX_SHA256
    assert result["archive"]["final_report_sha256"] == spatial_support_module.EXPECTED_05JB_FINAL_SHA256
    assert not result["representation_revision_passed"]
    assert result["next_step"] == "05j_c_support_and_decoder_revision"


def test_05jb_tail_transform_preserves_order_without_saturating():
    values = np.asarray([-100.0, -10.0, 0.0, 10.0, 100.0])
    tanh = revised_feature_transform(
        values, "tanh", tanh_scale=4.0, asinh_reference_z=8.0
    )
    asinh = revised_feature_transform(
        values, "asinh", tanh_scale=4.0, asinh_reference_z=8.0
    )
    assert np.all(np.diff(asinh) > 0)
    assert asinh[-1] > 1.0
    assert tanh[-1] == pytest.approx(1.0)


def test_05jb_bounded_target_parameterization_roundtrips():
    values = np.asarray([-100.0, -5.0, 0.0, 5.0, 100.0])
    encoded = bounded_target_encode(values, 120.0, 1e-6)
    np.testing.assert_allclose(
        bounded_target_decode(encoded, 120.0), values, atol=1e-10
    )
    with pytest.raises(RuntimeError, match="decoder domain"):
        bounded_target_encode(np.asarray([120.0]), 120.0, 1e-6)


def test_05jb_dual_ridge_path_fits_segment_specific_affine_surface():
    rng = np.random.default_rng(123)
    features = rng.normal(size=(10, 4, 5))
    coefficient = rng.normal(size=(4, 5))
    target = np.einsum("nsf,sf->ns", features, coefficient) + 0.25
    prediction, diagnostics = dual_ridge_path_predict(
        features, target, features, [1e-8, 1.0], pair_branch_weight=1.0
    )
    assert prediction.shape == (2, 10, 4)
    assert np.sqrt(np.mean((prediction[0] - target) ** 2)) < 1e-6
    assert diagnostics[1]["maximum_regularized_condition_number"] < diagnostics[0][
        "maximum_regularized_condition_number"
    ]
    assert diagnostics[0]["fit_row_count"] == 15
    assert diagnostics[0]["pair_branch_weight"] == 1.0


def test_05jb_selection_score_penalizes_branch_collapse():
    healthy = {
        "aggregate_voltage_rmse_mv": 1.0,
        "maximum_segment_error_mv": 2.0,
        "pair_metrics": [{"branching_retention": 1.0}],
    }
    collapsed = {
        **healthy,
        "pair_metrics": [{"branching_retention": 0.1}],
    }
    assert pair_gate_selection_score(
        collapsed, max_error_weight=0.05, branch_log_weight=2.0
    ) > pair_gate_selection_score(
        healthy, max_error_weight=0.05, branch_log_weight=2.0
    )


def test_05jb_config_rejects_posthoc_family_or_lambda_changes():
    HinesRepairedRepresentationRevisionConfig().validate()
    with pytest.raises(ValueError, match="retain h2"):
        HinesRepairedRepresentationRevisionConfig(
            input_families=("h2", "h2_causal")
        ).validate()
    with pytest.raises(ValueError, match="increasing"):
        HinesRepairedRepresentationRevisionConfig(
            ridge_lambdas=(1.0, 0.1)
        ).validate()


def test_05jb_notebook_seals_heldout_rollout_and_uses_browser_zip():
    root = Path(__file__).resolve().parents[2]
    notebook = json.loads(
        (root / "notebooks/05j_b_repaired_representation_revision.ipynb")
        .read_text(encoding="utf-8")
    )
    source = "\n".join(
        "".join(cell.get("source", [])) for cell in notebook["cells"]
    )
    assert "prepare_repaired_representation_revision" in source
    assert "run_transform_geometry_audit" in source
    assert "run_segmentwise_regularized_revision" in source
    assert "train_leave_one_pair_out" in source
    assert "assert not revision_report['development_used_for_selection']" in source
    assert "assert not final_report['heldout_contract']['inputs_extracted']" in source
    assert "assert not final_report['methodology']['rollout_performed']" in source
    assert "base64.b64encode" in source and "application/zip" in source
    assert "run_rollout" not in source
    assert "rglob('/kaggle" not in source


def test_05jb_exact_05j_hashes_match_registered_result():
    root = Path(__file__).resolve().parents[2]
    result = json.loads(
        (
            root
            / "experiments/hayflow/05j_repaired_representation_recheck/result.json"
        ).read_text(encoding="utf-8")
    )
    assert result["archive"]["sha256"] == repaired_revision_module.EXPECTED_05J_ARCHIVE_SHA256
    assert result["archive"]["artifact_index_sha256"] == repaired_revision_module.EXPECTED_05J_INDEX_SHA256
    assert result["archive"]["final_report_sha256"] == repaired_revision_module.EXPECTED_05J_FINAL_SHA256
    assert not result["representation_recheck_passed"]
    assert result["next_step"] == "05j_b_repaired_representation_revision"


def _netcon_layout_for_test():
    records = []
    synapse_type = {0: "ProbAMPANMDA2", 1: "ProbUDFsyn2"}
    for synapse_id, width in ((0, 6), (1, 4)):
        for slot in range(1, width + 1):
            records.append({
                "category": "synapse_states",
                "scope": "synapse",
                "owner_id": synapse_id,
                "mechanism": "NetCon",
                "variable": f"weight[{slot}]",
                "kind": "state",
            })
    return SimpleNamespace(core_records=records, synapse_type=synapse_type)


def test_05ib_netcon_slots_are_decoded_by_point_process_class():
    records, report = netcon_semantic_records(_netcon_layout_for_test())
    assert not report["unmapped"]
    assert [row["variable"] for row in records[:6]] == [
        "weight_AMPA", "weight_NMDA", "Pv", "Pr", "u", "tsyn"
    ]
    assert [row["variable"] for row in records[6:]] == ["Pv", "Pr", "u", "tsyn"]
    assert records[3]["point_process_class"] == "ProbAMPANMDA2"
    assert records[9]["point_process_class"] == "ProbUDFsyn2"


def test_05ib_tsyn_age_encoding_is_causal_and_reversible():
    encoder = NetConSemanticStateEncoder(
        _netcon_layout_for_test(), HinesNetConSemanticRepairConfig(
            expected_synapse_count_per_class=1,
            expected_netcon_coordinate_count=10,
            expected_tsyn_coordinate_count=2,
            expected_probability_coordinate_count=6,
            expected_amplitude_coordinate_count=2,
        )
    )
    raw = np.asarray([[1.0, 1.0, 0.8, 0.2, 0.5, 90.0, 0.7, 0.3, 0.4, 99.5]])
    semantic = encoder.encode(raw, [100.0])
    assert semantic[0, 5] == pytest.approx(10.0)
    assert semantic[0, 9] == pytest.approx(0.5)
    np.testing.assert_allclose(encoder.decode(semantic, [100.0]), raw, atol=1e-12)
    normalizer = SimpleNamespace(
        transform_codes=np.zeros(10, dtype=np.int8), LOG1P=1, LOGIT=2
    )
    encoder.configure_transform_codes(normalizer)
    assert normalizer.transform_codes.tolist() == [1, 1, 2, 2, 2, 1, 2, 2, 2, 1]
    with pytest.raises(ValueError, match="after the causal boundary"):
        encoder.encode(raw, [80.0])


def test_05ib_notebook_keeps_thresholds_targets_heads_and_rollout_sealed():
    root = Path(__file__).resolve().parents[2]
    notebook = json.loads(
        (root / "notebooks/05i_b_netcon_semantic_state_repair.ipynb").read_text(
            encoding="utf-8"
        )
    )
    source = "\n".join(
        "".join(cell.get("source", [])) for cell in notebook["cells"]
    )
    assert "run_netcon_semantic_roundtrip_audit" in source
    assert "run_coordinate_scale_repair" in source
    assert "run_repaired_frozen_h2_audit" in source
    assert "assert prepare_report['thresholds_inherited_unchanged']" in source
    assert "assert not final_report['heldout_contract']['boundary_targets_materialized']" in source
    assert "assert not final_report['heldout_contract']['candidate_head_inference_performed']" in source
    assert "run_bounded_representation_controls" not in source
    assert "run_rollout" not in source
    assert "base64.b64encode" in source and "application/zip" in source
    assert "rglob('/kaggle" not in source


def _synaptic_domain_layout_for_test():
    records = []
    synapse_type = {0: "ProbAMPANMDA2", 1: "ProbUDFsyn2"}
    for variable in ("A_AMPA", "B_AMPA", "A_NMDA", "B_NMDA"):
        records.append({
            "category": "synapse_states", "scope": "synapse", "owner_id": 0,
            "mechanism": "ProbAMPANMDA2", "variable": variable, "kind": "state",
        })
    for slot in range(1, 7):
        records.append({
            "category": "synapse_states", "scope": "synapse", "owner_id": 0,
            "mechanism": "NetCon", "variable": f"weight[{slot}]", "kind": "state",
        })
    for variable in ("A", "B"):
        records.append({
            "category": "synapse_states", "scope": "synapse", "owner_id": 1,
            "mechanism": "ProbUDFsyn2", "variable": variable, "kind": "state",
        })
    for slot in range(1, 5):
        records.append({
            "category": "synapse_states", "scope": "synapse", "owner_id": 1,
            "mechanism": "NetCon", "variable": f"weight[{slot}]", "kind": "state",
        })
    synapses = [
        {"id": 0, "parameters": {"Dep": 100.0, "Fac": 10.0}},
        {"id": 1, "parameters": {"Dep": 0.0, "Fac": 0.0}},
    ]
    return SimpleNamespace(
        core_records=records, synapse_type=synapse_type, synapses=synapses
    )


def test_05ic_bounded_recency_is_causal_parameter_aware_and_reversible():
    layout = _synaptic_domain_layout_for_test()
    encoder = BoundedSynapticStateEncoder(
        layout,
        HinesNetConSemanticRepairConfig(
            expected_synapse_count_per_class=1,
            expected_netcon_coordinate_count=10,
            expected_tsyn_coordinate_count=2,
            expected_probability_coordinate_count=6,
            expected_amplitude_coordinate_count=2,
        ),
        HinesSynapticDomainRepairConfig(
            expected_tsyn_coordinate_count=2,
            expected_dynamic_trace_coordinate_count=6,
        ),
    )
    raw = np.asarray([[
        0.1, 0.2, 0.3, 0.4,
        1.0, 1.0, 0.8, 0.2, 0.5, 90.0,
        0.6, 0.7,
        0.7, 0.3, 0.4, 99.5,
    ]])
    semantic = encoder.encode(raw, [100.0])
    assert encoder.recency_time_ms.tolist() == [100.0, 1.0]
    assert semantic[0, 9] == pytest.approx(100.0 / 110.0)
    assert semantic[0, 15] == pytest.approx(1.0 / 1.5)
    np.testing.assert_allclose(encoder.decode(semantic, [100.0]), raw, atol=1e-12)
    normalizer = SimpleNamespace(
        transform_codes=np.zeros(16, dtype=np.int8), IDENTITY=0, LOG1P=1, LOGIT=2
    )
    encoder.configure_transform_codes(normalizer)
    assert normalizer.transform_codes.tolist() == [
        1, 1, 1, 1, 1, 1, 2, 2, 2, 0, 1, 1, 2, 2, 2, 0
    ]
    with pytest.raises(ValueError, match="after the causal boundary"):
        encoder.encode(raw, [80.0])


def test_05ic_domain_floors_are_preregistered_below_the_global_gate():
    config = HinesSynapticDomainRepairConfig()
    assert 1.0 / config.bounded_recency_scale_floor == pytest.approx(50.0)
    assert (
        np.log1p(config.trace_reference_raw_increment)
        / config.synaptic_trace_log1p_scale_floor
    ) == pytest.approx(35.0)
    assert config.bounded_recency_target_standardized_span < 100.0


def test_05ib_registered_result_requires_separate_domain_revision():
    root = Path(__file__).resolve().parents[2]
    record = json.loads(
        (root / "experiments/hayflow/05i_b_netcon_semantic_state_repair/result.json")
        .read_text(encoding="utf-8")
    )
    assert record["archive"]["sha256"] == synaptic_domain_module.EXPECTED_05IB_ARCHIVE_SHA256
    assert record["netcon_semantic_roundtrip"]["maximum_roundtrip_absolute_error"] == 0.0
    assert record["coordinate_support"]["coordinate_count_above_standardized_limit"] == 7
    assert not record["input_contract_passed"]
    assert record["next_step"] == "05i_c_netcon_semantic_revision"


def test_05ic_notebook_keeps_targets_heads_rollout_and_thresholds_sealed():
    root = Path(__file__).resolve().parents[2]
    notebook = json.loads(
        (root / "notebooks/05i_c_synaptic_domain_repair.ipynb").read_text(
            encoding="utf-8"
        )
    )
    source = "\n".join(
        "".join(cell.get("source", [])) for cell in notebook["cells"]
    )
    assert "run_bounded_recency_roundtrip_audit" in source
    assert "run_coordinate_scale_repair" in source
    assert "run_synaptic_domain_floor_audit" in source
    assert "run_repaired_frozen_h2_audit" in source
    assert "assert prepare_report['threshold_contract']['unchanged']" in source
    assert "assert domain_floor_report['domain_floor_contract_passed']" not in source
    assert "assert not final_report['heldout_contract']['boundary_targets_materialized']" in source
    assert "assert not final_report['heldout_contract']['candidate_head_inference_performed']" in source
    assert "run_rollout" not in source
    assert "base64.b64encode" in source and "application/zip" in source
    assert "rglob('/kaggle" not in source


def test_05j_robust_gate_requires_two_of_three_seeds():
    config = HinesRepairedRepresentationRecheckConfig()
    config.validate()
    assert config.minimum_joint_passing_seed_count == 2
    assert config.expected_seed_count == 3
    with pytest.raises(ValueError, match="cannot exceed"):
        HinesRepairedRepresentationRecheckConfig(
            minimum_joint_passing_seed_count=4
        ).validate()
    with pytest.raises(ValueError, match="must not extract"):
        HinesRepairedRepresentationRecheckConfig(
            forbid_heldout_input_extraction=False
        ).validate()


def test_05j_robust_family_gate_requires_joint_passes():
    runs = []
    for seed, train_pass, development_pass, rmse in (
        (17, True, True, 0.5),
        (29, True, True, 0.7),
        (43, True, False, 1.2),
    ):
        runs.append({
            "family": "h2_causal",
            "seed": seed,
            "train_passed": train_pass,
            "development_passed": development_pass,
            "train": {"aggregate_voltage_rmse_mv": rmse},
            "development": {"aggregate_voltage_rmse_mv": rmse + 0.1},
        })
    row = summarize_robust_family_gate(
        runs, ["h2_causal"],
        expected_seed_count=3,
        minimum_joint_passing_seed_count=2,
    )[0]
    assert row["joint_passing_seed_count"] == 2
    assert row["robust_family_passed"]
    runs[1]["development_passed"] = False
    row = summarize_robust_family_gate(
        runs, ["h2_causal"],
        expected_seed_count=3,
        minimum_joint_passing_seed_count=2,
    )[0]
    assert row["joint_passing_seed_count"] == 1
    assert not row["robust_family_passed"]
def test_05ic_registered_result_authorizes_only_05j_recheck():
    root = Path(__file__).resolve().parents[2]
    record = json.loads(
        (root / "experiments/hayflow/05i_c_synaptic_domain_repair/result.json")
        .read_text(encoding="utf-8")
    )
    assert record["archive"]["sha256"] == repaired_recheck_module.EXPECTED_05IC_ARCHIVE_SHA256
    assert record["input_contract_passed"]
    assert record["coordinate_support"]["coordinate_count_above_standardized_limit"] == 0
    assert record["next_step"] == "05j_repaired_representation_train_development_recheck"
    assert not record["full_training_authorized"]


def test_05j_notebook_never_extracts_heldout_and_has_no_rollout():
    root = Path(__file__).resolve().parents[2]
    notebook = json.loads(
        (
            root
            / "notebooks/05j_repaired_representation_train_development_recheck.ipynb"
        ).read_text(encoding="utf-8")
    )
    source = "\n".join(
        "".join(cell.get("source", [])) for cell in notebook["cells"]
    )
    assert "apply_verified_synaptic_domain_normalizer" in source
    assert "prepare_train_development_features" in source
    assert "run_projection_forensics" in source
    assert "run_repaired_bounded_controls" in source
    assert "run_coordinate_scale_repair" not in source
    assert "run_raw_scale_forensics" not in source
    assert "run_rollout" not in source
    assert "assert not feature_report['heldout_inputs_extracted']" in source
    assert "assert not final_report['heldout_contract']['inputs_extracted']" in source
    assert "base64.b64encode" in source and "application/zip" in source
    assert "rglob('/kaggle" not in source


def _state_record(category, mechanism, variable, owner_id):
    return {
        "category": category,
        "scope": "segment",
        "owner_id": owner_id,
        "mechanism": mechanism,
        "variable": variable,
        "kind": "state",
    }


def test_05i_semantic_scale_repair_is_train_only_and_hierarchical():
    records = [
        _state_record("mechanism_states", "NaTa_t", "m", 0),
        _state_record("mechanism_states", "NaTa_t", "m", 1),
        _state_record("mechanism_states", "NaTa_t", "h", 0),
        _state_record("calcium_ions", "cad", "cai", 0),
    ]
    original = np.asarray([1e-8, 4.0, 1e-8, 1e-8])
    codes = np.asarray([2, 2, 2, 1], dtype=np.int8)
    config = HinesStateNormalizationRepairConfig(
        pooling_quantile=0.25,
        exact_group_multiplier=0.25,
        mechanism_group_multiplier=0.15,
        category_group_multiplier=0.10,
        transform_group_multiplier=0.05,
    )
    repaired, rows = semantic_state_scale_repair(records, codes, original, config)
    assert repaired[0] == pytest.approx(1.0)
    assert rows[0]["floor_source"] == "exact_train_pool"
    assert repaired[1] == pytest.approx(4.0)
    assert repaired[2] == pytest.approx(0.6)
    assert rows[2]["floor_source"] == "mechanism_train_pool"
    assert repaired[3] == pytest.approx(config.log1p_absolute_floor)
    assert rows[3]["floor_source"] == "absolute_semantic_floor"


def test_05i_config_rejects_posthoc_or_unbounded_contracts():
    HinesStateNormalizationRepairConfig().validate()
    with pytest.raises(ValueError, match="pooling_quantile"):
        HinesStateNormalizationRepairConfig(pooling_quantile=0.0).validate()
    with pytest.raises(ValueError, match="maximum_clipping_fraction"):
        HinesStateNormalizationRepairConfig(maximum_clipping_fraction=1.0).validate()


def test_05i_logit_floor_bounds_the_registered_transform_span():
    config = HinesStateNormalizationRepairConfig()
    transformed_span = 2.0 * np.log((1.0 - 1e-6) / 1e-6)
    assert transformed_span / config.logit_absolute_floor < config.standardized_maximum


def test_05i_experimental_record_localizes_netcon_timestamp_semantics():
    root = Path(__file__).resolve().parents[2]
    record = json.loads(
        (
            root
            / "experiments/hayflow/05i_teacher_state_normalization_repair/result.json"
        ).read_text(encoding="utf-8")
    )
    assert record["status"] == "complete"
    assert record["diagnosis"] == "STATE_NORMALIZATION_REPAIR_INSUFFICIENT"
    assert record["artifact"]["sha256"] == (
        "76f94225937e8946c3142753604b0d6eb6c30771dba5d64970099dff83952943"
    )
    assert record["artifact_integrity"]["all_indexed_members_verified"]
    assert record["coordinate_scale_repair"]["heldout_improvement_factor"] > 6_000_000
    assert record["coordinate_scale_repair"][
        "coordinate_count_above_standardized_maximum_on_heldout"
    ] == 2
    assert record["residual_outlier_diagnosis"]["primary_cause"] == (
        "NETCON_SLOT_SEMANTICS_COLLAPSED_BY_RAW_WEIGHT_INDEX"
    )
    assert all(
        row["semantic_variable"] == "ProbUDFsyn2.tsyn"
        for row in record["residual_outlier_diagnosis"]["failing_coordinates"]
    )
    assert record["repaired_frozen_h2_audit"]["input_contract_passed"]
    assert not record["interpretation"]["candidate_head_recheck_authorized"]
    assert not record["next_step"]["full_training_authorized"]


def test_05i_notebook_seals_targets_heads_rollout_and_uses_browser_zip():
    root = Path(__file__).resolve().parents[2]
    notebook = json.loads(
        (root / "notebooks/05i_teacher_state_normalization_repair.ipynb").read_text(
            encoding="utf-8"
        )
    )
    source = "\n".join(
        "".join(cell.get("source", [])) for cell in notebook["cells"]
    )
    assert "run_coordinate_scale_repair" in source
    assert "run_repaired_frozen_h2_audit" in source
    assert "assert not final_report['heldout_contract']['boundary_targets_materialized']" in source
    assert "assert not final_report['heldout_contract']['candidate_head_inference_performed']" in source
    assert "assert not final_report['full_training_authorized']" in source
    assert "base64.b64encode" in source and "application/zip" in source
    assert "run_bounded_representation_controls" not in source
    assert "run_rollout" not in source
    assert "rglob('/kaggle" not in source


def test_05h_notebook_keeps_heldout_targets_and_inference_sealed():
    root = Path(__file__).resolve().parents[2]
    notebook = json.loads(
        (root / "notebooks/05h_hayflow_hines_representation_forensics.ipynb").read_text(
            encoding="utf-8"
        )
    )
    source = "\n".join(
        "".join(cell.get("source", [])) for cell in notebook["cells"]
    )
    assert "run_raw_scale_forensics" in source
    assert "run_projection_forensics" in source
    assert "run_bounded_representation_controls" in source
    assert "assert not controls_report['heldout_candidate_head_inference_performed']" in source
    assert "assert final_report['heldout_contract']['frozen_h2_feature_extraction_performed']" in source
    assert "assert not raw_report['heldout_event_targets_materialized']" in source
    assert "assert not final_report['full_training_authorized']" in source
    assert "base64.b64encode" in source and "application/zip" in source
    assert "rglob('/kaggle" not in source


def test_05h_experimental_record_prioritizes_normalization_repair():
    root = Path(__file__).resolve().parents[2]
    record = json.loads(
        (
            root
            / "experiments/hayflow/05h_hayflow_hines_representation_forensics/result.json"
        ).read_text(encoding="utf-8")
    )
    assert record["status"] == "complete"
    assert record["diagnosis"] == "FROZEN_H2_HELDOUT_INPUT_OOD"
    assert record["artifact"]["sha256"] == (
        "a471b49740154239821de9f9f71096e78bfcd1a9866026500615bae6b6c9524d"
    )
    assert record["artifact_integrity"]["all_indexed_members_verified"]
    assert not record["heldout_contract"]["boundary_targets_materialized"]
    assert not record["heldout_contract"]["event_targets_materialized"]
    assert not record["heldout_contract"]["candidate_head_inference_performed"]
    assert record["raw_scale_forensics"][
        "normalized_teacher_state_heldout_to_train_max_ratio"
    ] > 85_000_000
    assert record["raw_scale_forensics"]["normalizer_posthoc_audit"][
        "variable_count_at_minimum_scale"
    ] == 11888
    assert record["linear_projection_forensics"]["all_train_pairs_passed"]
    assert record["bounded_nonlinear_controls"]["train_passing_run_count"] == 0
    assert record["interpretation"]["normalization_contract_is_primary_blocker"]
    assert not record["next_step"]["full_training_authorized"]


def test_05h_config_accepts_registered_controls_and_rejects_missing_family():
    HinesRepresentationForensicsConfig().validate()
    with pytest.raises(ValueError, match="h2, causal, and h2_causal"):
        HinesRepresentationForensicsConfig(input_families=("h2", "causal")).validate()


def test_05h_robust_bounding_preserves_unclipped_forensic_surface():
    values = np.asarray([[[0.0, 1.0], [1000.0, -1000.0]]])
    mean = np.zeros((1, 1, 2))
    scale = np.ones((1, 1, 2))
    standardized, bounded = robust_bounded_features(values, mean, scale, 4.0)
    assert standardized[0, 1, 0] == 1000.0
    assert np.max(np.abs(bounded)) <= 1.0
    assert bounded[0, 1, 0] > 0.999


def test_05h_local_projection_reports_irreducible_segment_error():
    feature = np.asarray([
        [[0.0], [0.0]], [[1.0], [0.0]], [[2.0], [0.0]], [[3.0], [0.0]],
    ])
    target = np.asarray([
        [1.0, 0.0], [3.0, 1.0], [5.0, 0.0], [7.0, 1.0],
    ])
    prediction, rows = local_linear_projection(feature, target, 1e-12)
    np.testing.assert_allclose(prediction[:, 0], target[:, 0], atol=1e-10)
    assert rows[0]["projection_rmse_mv"] < 1e-10
    assert rows[1]["projection_rmse_mv"] > 0.4


def test_05h_accepts_member_verified_extracted_05g(tmp_path, monkeypatch):
    payloads = {
        "artifact_index.json": b"{}",
        "optimization_audit_config.json": b"{}",
        "optimization_support.json": b'{"valid":true}',
        "feature_scale_audit.json": b"{}",
        "oracle_controls.json": b"{}",
        "regularized_train_audit.json": b"{}",
        "heldout_gate_report.json": b"{}",
        "final_report.json": b'{"diagnosis":"test"}',
    }
    expected = {}
    for name, payload in payloads.items():
        (tmp_path / name).write_bytes(payload)
        expected[name] = hashlib.sha256(payload).hexdigest()
    monkeypatch.setattr(
        representation_forensics_module, "EXPECTED_05G_MEMBER_SHA256", expected
    )
    experiment = object.__new__(HinesRepresentationForensics)
    experiment.artifact_05g_source = tmp_path
    report, support, contract = experiment._read_05g_source()
    assert report["diagnosis"] == "test"
    assert support["valid"]
    assert contract["source_kind"] == "kaggle_extracted_directory"
    assert contract["verified_member_sha256"] == expected


def test_05g_notebook_enforces_train_first_bounded_audit_contract():
    root = Path(__file__).resolve().parents[2]
    notebook = json.loads(
        (root / "notebooks/05g_hayflow_hines_optimization_audit.ipynb").read_text(
            encoding="utf-8"
        )
    )
    source = "\n".join(
        "".join(cell.get("source", [])) for cell in notebook["cells"]
    )
    assert "bundle.report" not in source
    assert "run_regularized_train_audit" in source
    assert "reveal_heldout_if_safe" in source
    assert "assert not final_report['full_training_authorized']" in source
    assert "application/zip" in source and "base64.b64encode" in source
    assert "rglob('/kaggle" not in source


def test_05g_experimental_record_preserves_scoped_no_go_and_raw_ood_caveat():
    root = Path(__file__).resolve().parents[2]
    record = json.loads(
        (
            root
            / "experiments/hayflow/05g_hayflow_hines_optimization_audit/result.json"
        ).read_text(encoding="utf-8")
    )
    assert record["status"] == "complete"
    assert record["diagnosis"] == (
        "REGULARIZED_FROZEN_FEATURES_CANNOT_FIT_TRAIN_SUPPORT"
    )
    assert record["artifact"]["sha256"] == (
        "f369723f0d7184ea672e90fd4f530c8ad301eed088788067dae6a5d0524be66d"
    )
    assert record["artifact_integrity"]["all_indexed_members_verified"]
    assert record["support"]["selected_protocol_family_count"] == 6
    assert record["oracle_controls"]["direct_per_transition_residual"][
        "all_pairs_passed"
    ]
    assert record["regularized_audit"]["train_passing_candidate_count"] == 0
    assert not record["heldout_gate"]["revealed"]
    assert (
        record["feature_scale_contract"][
            "raw_heldout_to_train_maximum_segment_norm_ratio"
        ]
        > 29_000
    )
    assert not record["next_step"]["full_training_authorized"]


def test_05g_config_rejects_unregistered_ranks_and_accepts_default():
    HinesOptimizationAuditConfig().validate()
    with pytest.raises(ValueError, match="ranks 64 and 96"):
        HinesOptimizationAuditConfig(ranks=(32, 96)).validate()


def test_dual_ridge_recovers_centered_segment_map_and_bounds_residual():
    rng = np.random.default_rng(17)
    features = rng.normal(size=(24, 3, 4))
    coefficients = rng.normal(size=(3, 4))
    bias = np.asarray([1.5, -2.0, 0.25])
    target = bias[None, :] + np.einsum("nsf,sf->ns", features, coefficients)
    target_mean, feature_mean, fitted, diagnostics = dual_ridge_segment_coefficients(
        features, target, 1e-10
    )
    fitted_bias = target_mean - np.einsum("sf,sf->s", feature_mean, fitted)
    predicted, safety = bounded_segment_prediction(
        features, fitted_bias, fitted, residual_limit_mv=120.0
    )
    np.testing.assert_allclose(predicted, target, atol=1e-7, rtol=1e-7)
    assert diagnostics["minimum_local_rank"] == 4
    assert safety["clipped_fraction"] == 0.0
    clipped, clipped_safety = bounded_segment_prediction(
        features, np.full(3, 500.0), fitted, residual_limit_mv=120.0
    )
    assert np.max(np.abs(clipped)) == 120.0
    assert clipped_safety["clipped_fraction"] > 0.0


def test_05g_accepts_only_member_verified_extracted_05f(tmp_path, monkeypatch):
    payloads = {
        "artifact_index.json": b"{}",
        "micro_canary_config.json": b"{}",
        "pair_plan.json": b"{}",
        "feature_contract.json": b"{}",
        "spectral_basis_report.json": b"{}",
        "micro_canary_report.json": b"{}",
        "rank_64_report.json": b"{}",
        "rank_96_report.json": b"{}",
        "final_report.json": b'{"diagnosis":"test"}',
    }
    expected = {}
    for name, payload in payloads.items():
        (tmp_path / name).write_bytes(payload)
        expected[name] = hashlib.sha256(payload).hexdigest()
    monkeypatch.setattr(optimization_audit_module, "EXPECTED_05F_MEMBER_SHA256", expected)
    experiment = object.__new__(HinesSegmentOptimizationAudit)
    experiment.artifact_05f_source = tmp_path
    report, contract = experiment._read_05f_source()
    assert report["diagnosis"] == "test"
    assert contract["source_kind"] == "kaggle_extracted_directory"
    assert contract["verified_member_sha256"] == expected


def test_05g_protocol_family_does_not_require_optional_metadata_column():
    experiment = object.__new__(HinesSegmentOptimizationAudit)
    experiment.store = SimpleNamespace(
        metadata={
            "trajectory_id": np.asarray(["episode-a"]),
            "protocol": np.asarray(["fallback-protocol"]),
        },
        episode_by_trajectory={
            "episode-a": {"protocol": "targeted-bap", "protocol_variant": "paired"}
        },
    )
    assert experiment._protocol_family(0) == "targeted-bap|paired"
    experiment.store.episode_by_trajectory = {}
    assert experiment._protocol_family(0) == "fallback-protocol|unknown_variant"


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


def test_05e_experimental_record_limits_the_capacity_go_to_a_micro_canary():
    root = Path(__file__).resolve().parents[2]
    record = json.loads(
        (
            root
            / "experiments/hayflow/05e_hayflow_hines_segment_capacity/result.json"
        ).read_text(encoding="utf-8")
    )
    assert record["status"] == "complete"
    assert record["decision"] == "DIAGNOSTIC_ONLY_NO_FULL_TRAINING"
    assert record["diagnosis"] == "SEGMENT_CONDITIONED_CAPACITY_SUFFICIENT"
    assert record["artifact"]["sha256"] == (
        "a8e47a979678cef19ca45e5647528bf82dab4f49923dd4d5400705abbe104a48"
    )
    assert record["artifact_integrity"]["all_indexed_members_verified"]
    assert record["artifact_integrity"]["all_indexed_sizes_verified"]
    assert record["one_transition"]["segment_bias_only"]["passed"]
    assert not record["branch_pair_base_probes"]["segment_bias_only"]["passed"]
    rank_path = {row["rank"]: row for row in record["segment_conditioned_rank_path"]}
    assert not rank_path[64]["passed"]
    assert rank_path[96]["passed"]
    assert record["interpretation"]["maximum_tested_rank_required"]
    assert not record["interpretation"]["compact_low_rank_solution_demonstrated"]
    assert not record["interpretation"]["out_of_sample_generalization_demonstrated"]
    assert not record["next_step"]["full_training_authorized"]


def test_05f_experimental_record_scopes_the_optimization_failure():
    root = Path(__file__).resolve().parents[2]
    record = json.loads(
        (
            root
            / "experiments/hayflow/05f_hayflow_hines_segment_micro_canary/result.json"
        ).read_text(encoding="utf-8")
    )
    assert record["status"] == "complete"
    assert record["decision"] == "DIAGNOSTIC_ONLY_NO_FULL_TRAINING"
    assert record["diagnosis"] == (
        "SEGMENT_CONDITIONED_MICRO_CANARY_OPTIMIZATION_FAILURE"
    )
    assert record["artifact"]["sha256"] == (
        "3a641d10ede14d426c640964cf0a6491259f6297403564db0d694f544da22239"
    )
    assert record["artifact_integrity"]["all_indexed_members_verified"]
    assert record["artifact_integrity"]["all_indexed_sizes_verified"]
    assert record["pair_plan"]["development_pair_excluded_from_training"]
    assert record["pair_plan"]["train_heldout_episode_overlap"] == []
    assert record["pair_plan"]["all_training_pairs_share_protocol"]
    assert record["feature_and_spectral_contract"]["minimum_local_design_rank"] == 13
    assert record["feature_and_spectral_contract"]["maximum_local_design_rank"] == 15
    assert not record["runs"]["rank_64"]["train"]["all_pairs_passed"]
    assert not record["runs"]["rank_96"]["heldout"]["all_pairs_passed"]
    assert record["interpretation"]["catastrophic_absolute_heldout_extrapolation"]
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


def test_05e_notebook_is_closed_form_gated_and_uses_blob_download():
    root = Path(__file__).resolve().parents[2]
    notebook = json.loads(
        (
            root / "notebooks/05e_hayflow_hines_segment_capacity_probe.ipynb"
        ).read_text(encoding="utf-8")
    )
    source = "\n".join(
        "".join(cell.get("source", [])) for cell in notebook["cells"]
    )
    assert "bundle.report" not in source
    assert "bundle.manifest" in source
    assert "hayflow_hines_residual_conditioning.zip" in source
    assert "session.run_capacity_probes()" in source
    assert "session.run_full" not in source
    assert "assert not final_report['full_training_authorized']" in source
    assert "base64.b64encode" in source
    assert "new Blob" in source
    assert "FileLink" not in source


def test_05f_notebook_uses_disjoint_pairs_and_has_no_full_training_path():
    root = Path(__file__).resolve().parents[2]
    notebook = json.loads(
        (
            root / "notebooks/05f_hayflow_hines_segment_micro_canary.ipynb"
        ).read_text(encoding="utf-8")
    )
    source = "\n".join(
        "".join(cell.get("source", [])) for cell in notebook["cells"]
    )
    assert "bundle.report" not in source
    assert "bundle.manifest" in source
    assert "hayflow_hines_segment_capacity.zip" in source
    assert "session.build_pair_plan()" in source
    assert "session.run_micro_canary()" in source
    assert "development_pair_excluded_from_training" in source
    assert "session.run_full" not in source
    assert "assert not final_report['full_training_authorized']" in source
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


def test_05e_capacity_config_and_closed_form_solver_are_deterministic():
    config = HinesCapacityConfig()
    config.validate()
    assert config.rank_candidates[-1] == 96
    assert len(EXPECTED_05D_ARCHIVE_SHA256) == 64
    with pytest.raises(ValueError, match="unique and increasing"):
        HinesCapacityConfig(rank_candidates=(2, 1)).validate()
    design = np.asarray([
        [1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, -1.0], [-1.0, 2.0]
    ])
    target = 2.0 * design[:, 0] - 3.0 * design[:, 1]
    coefficients, prediction, report = solve_linear_probe(
        design, target, config.svd_rcond
    )
    np.testing.assert_allclose(coefficients, [2.0, -3.0], atol=1e-10)
    np.testing.assert_allclose(prediction, target, atol=1e-10)
    assert report["numerical_rank"] == 2
    assert report["irreducible_rmse"] < 1e-10


def test_segment_conditioned_rank_path_separates_static_and_dynamic_terms():
    rng = np.random.default_rng(12)
    temporal_features = rng.normal(size=(3, 4))
    temporal_features -= temporal_features.mean(axis=0, keepdims=True)
    features = np.tile(temporal_features[:, None, :], (1, 5, 1))
    segment_factor = np.linspace(0.5, 1.5, 5)
    feature_factor = np.asarray([0.2, -0.4, 0.7, 0.1])
    coefficients = np.outer(segment_factor, feature_factor)
    static_bias = np.linspace(-2.0, 2.0, 5)
    target = static_bias[None, :] + np.einsum(
        "nsf,sf->ns", features, coefficients
    )
    rows, diagnostics = segment_conditioned_rank_path(
        features, target, ranks=(1, 2), rcond=1e-12
    )
    assert diagnostics["locally_unidentifiable_segment_count"] == 0
    assert diagnostics["coefficient_matrix_rank"] == 1
    np.testing.assert_allclose(rows[0]["predicted_residual"], target, atol=1e-10)
    assert rows[0]["dynamic_residual_rmse_mv"] < 1e-10


def test_05f_config_and_disjoint_pair_selection_preserve_the_preregistered_roles():
    config = HinesSegmentCanaryConfig()
    config.validate()
    assert config.ranks == (64, 96)
    assert config.minimum_train_pair_count > 1
    assert config.minimum_heldout_split_count == 2
    assert "train" not in config.heldout_splits
    assert len(EXPECTED_05E_ARCHIVE_SHA256) == 64
    with pytest.raises(ValueError, match="ranks 64 and 96"):
        HinesSegmentCanaryConfig(ranks=(96, 64)).validate()
    candidates = [
        {
            "left_index": 2 * row,
            "right_index": 2 * row + 1,
            "left_episode_id": f"left-{row}",
            "right_episode_id": f"right-{row}",
        }
        for row in range(4)
    ]
    selected = HinesSegmentMicroCanaryExperiment._select_disjoint_pairs(
        candidates, 2, excluded_indices=(0, 1),
        excluded_episode_ids=("left-2", "right-2"),
    )
    assert len(selected) == 2
    assert all(row["left_index"] not in {0, 1} for row in selected)
    assert all(row["left_episode_id"] != "left-2" for row in selected)


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="PyTorch is not installed locally")
def test_05f_segment_residual_starts_at_exactly_zero():
    import torch
    from src.hayflow_model.hines_segment_canary_experiment import (
        ZeroOutputSpectralSegmentResidual,
    )

    basis = np.eye(4, 6, dtype=np.float32)
    model = ZeroOutputSpectralSegmentResidual(5, 6, 4, basis)
    features = torch.randn(3, 2, 5, 6)
    residual = model(features)
    assert residual.shape == (3, 2, 5)
    assert torch.count_nonzero(residual) == 0
    residual.sum().backward()
    assert model.segment_bias.grad is not None
    assert model.segment_factors.grad is not None


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


def test_05e_accepts_a_hash_verified_kaggle_extracted_05d_artifact(
    tmp_path, monkeypatch
):
    final = json.dumps({
        "diagnosis": "SHARED_REPRESENTATION_BOTTLENECK",
        "full_training_authorized": False,
    }).encode()
    free = b"free-control"
    expected = {
        "final_report.json": hashlib.sha256(final).hexdigest(),
        "free_residual_report.json": hashlib.sha256(free).hexdigest(),
    }
    monkeypatch.setattr(capacity_module, "EXPECTED_05D_MEMBER_SHA256", expected)
    root = tmp_path / "hayflow_hines_residual_conditioning"
    root.mkdir()
    (root / "final_report.json").write_bytes(final)
    (root / "free_residual_report.json").write_bytes(free)
    experiment = HinesSegmentCapacityExperiment.__new__(
        HinesSegmentCapacityExperiment
    )
    experiment.artifact_05d_source = root
    report, contract = experiment._read_05d_source()
    assert report["diagnosis"] == "SHARED_REPRESENTATION_BOTTLENECK"
    assert contract["source_kind"] == "kaggle_extracted_directory"
    assert contract["archive_sha256"] is None
    assert contract["verified_member_sha256"] == expected


def test_05f_accepts_a_hash_verified_kaggle_extracted_05e_artifact(
    tmp_path, monkeypatch
):
    final = json.dumps({
        "diagnosis": "SEGMENT_CONDITIONED_CAPACITY_SUFFICIENT",
        "full_training_authorized": False,
    }).encode()
    capacity = b"capacity-probe"
    expected = {
        "final_report.json": hashlib.sha256(final).hexdigest(),
        "capacity_probe_report.json": hashlib.sha256(capacity).hexdigest(),
    }
    monkeypatch.setattr(
        segment_canary_module, "EXPECTED_05E_MEMBER_SHA256", expected
    )
    root = tmp_path / "hayflow_hines_segment_capacity"
    root.mkdir()
    (root / "final_report.json").write_bytes(final)
    (root / "capacity_probe_report.json").write_bytes(capacity)
    experiment = HinesSegmentMicroCanaryExperiment.__new__(
        HinesSegmentMicroCanaryExperiment
    )
    experiment.artifact_05e_source = root
    report, contract = experiment._read_05e_source()
    assert report["diagnosis"] == "SEGMENT_CONDITIONED_CAPACITY_SUFFICIENT"
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
