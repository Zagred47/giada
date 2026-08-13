"""Full-state flow-map baselines and the future latent HayFlow model."""

from .full_state_flowmap import (
    DualRidgeBaseline,
    FlatResidualMLP,
    FlowmapModelConfig,
    PersistenceBaseline,
    StructuredSharedResidual,
    parameter_count,
    require_torch,
    ridge_design_matrix,
    structured_arrays,
)
from .flowmap_experiment import (
    AuxiliaryNormalizer,
    FlowmapExperimentConfig,
    FullStateFlowmapExperiment,
)
from .reconditioned_full_state import ReconditionedStructuredResidual
from .reconditioned_experiment import (
    ReconditionedExperimentConfig,
    ReconditionedFlowmapExperiment,
    ReconditionedRunSpec,
)
from .release_identifiability_experiment import (
    ReleaseExperimentConfig,
    ReleaseIdentifiabilityExperiment,
    ReleaseRunSpec,
)
from .hines_layer import (
    DifferentiableHinesSolve,
    HinesDiagnostics,
    morphology_arrays,
    tree_depths,
)
from .hayflow_hines import (
    HINES_SYNAPTIC_FEATURE_NAMES,
    SYNAPTIC_COMPONENTS,
    SYNAPTIC_STATISTICS,
    HayFlowHines,
    HayFlowHinesConfig,
    OrderedSegmentConvGRU,
    hayflow_hines_arrays,
    model_parameter_count,
)
from .hines_experiment import (
    HayFlowHinesExperiment,
    HinesPrototypeExperimentConfig,
)
from .hines_isolation_experiment import (
    BOUNDARY_MODES,
    EXPECTED_05B_ARCHIVE_SHA256,
    HinesCausalIsolationExperiment,
    HinesIsolationConfig,
)
from .hines_conditioning_experiment import (
    DECODER_PARAMETERIZATIONS,
    EXPECTED_05C_ARCHIVE_SHA256,
    HinesConditioningConfig,
    HinesResidualConditioningExperiment,
    ZeroInitializedBoundaryDecoder,
)
from .hines_capacity_experiment import (
    EXPECTED_05D_ARCHIVE_SHA256,
    HinesCapacityConfig,
    HinesSegmentCapacityExperiment,
    design_spectrum,
    segment_conditioned_rank_path,
    solve_linear_probe,
    standardize_design,
)
from .hines_segment_canary_experiment import (
    EXPECTED_05E_ARCHIVE_SHA256,
    HinesSegmentCanaryConfig,
    HinesSegmentMicroCanaryExperiment,
    ZeroOutputSpectralSegmentResidual,
)
from .hines_optimization_audit import (
    EXPECTED_05F_ARCHIVE_SHA256,
    HinesOptimizationAuditConfig,
    HinesSegmentOptimizationAudit,
    bounded_segment_prediction,
    dual_ridge_segment_coefficients,
)
from .hines_representation_forensics import (
    EXPECTED_05G_ARCHIVE_SHA256,
    BoundedLocalResidualHead,
    HinesRepresentationForensics,
    HinesRepresentationForensicsConfig,
    local_linear_projection,
    robust_bounded_features,
)
from .hines_state_normalization_repair import (
    EXPECTED_05H_ARCHIVE_SHA256,
    HinesStateNormalizationRepair,
    HinesStateNormalizationRepairConfig,
    semantic_state_scale_repair,
)
from .hines_netcon_semantic_repair import (
    EXPECTED_05I_ARCHIVE_SHA256,
    HinesNetConSemanticRepair,
    HinesNetConSemanticRepairConfig,
    NetConSemanticStateEncoder,
    netcon_semantic_records,
)
from .hines_synaptic_domain_repair import (
    EXPECTED_05IB_ARCHIVE_SHA256,
    BoundedSynapticStateEncoder,
    HinesSynapticDomainRepair,
    HinesSynapticDomainRepairConfig,
)

__all__ = [
    "DualRidgeBaseline",
    "FlatResidualMLP",
    "FlowmapModelConfig",
    "PersistenceBaseline",
    "StructuredSharedResidual",
    "parameter_count",
    "require_torch",
    "ridge_design_matrix",
    "structured_arrays",
    "AuxiliaryNormalizer",
    "FlowmapExperimentConfig",
    "FullStateFlowmapExperiment",
    "ReconditionedStructuredResidual",
    "ReconditionedExperimentConfig",
    "ReconditionedFlowmapExperiment",
    "ReconditionedRunSpec",
    "ReleaseExperimentConfig",
    "ReleaseIdentifiabilityExperiment",
    "ReleaseRunSpec",
    "DifferentiableHinesSolve",
    "HinesDiagnostics",
    "morphology_arrays",
    "tree_depths",
    "HINES_SYNAPTIC_FEATURE_NAMES",
    "SYNAPTIC_COMPONENTS",
    "SYNAPTIC_STATISTICS",
    "HayFlowHines",
    "HayFlowHinesConfig",
    "OrderedSegmentConvGRU",
    "hayflow_hines_arrays",
    "model_parameter_count",
    "HayFlowHinesExperiment",
    "HinesPrototypeExperimentConfig",
    "BOUNDARY_MODES",
    "EXPECTED_05B_ARCHIVE_SHA256",
    "HinesCausalIsolationExperiment",
    "HinesIsolationConfig",
    "DECODER_PARAMETERIZATIONS",
    "EXPECTED_05C_ARCHIVE_SHA256",
    "HinesConditioningConfig",
    "HinesResidualConditioningExperiment",
    "ZeroInitializedBoundaryDecoder",
    "EXPECTED_05D_ARCHIVE_SHA256",
    "HinesCapacityConfig",
    "HinesSegmentCapacityExperiment",
    "design_spectrum",
    "segment_conditioned_rank_path",
    "solve_linear_probe",
    "standardize_design",
    "EXPECTED_05E_ARCHIVE_SHA256",
    "HinesSegmentCanaryConfig",
    "HinesSegmentMicroCanaryExperiment",
    "ZeroOutputSpectralSegmentResidual",
    "EXPECTED_05F_ARCHIVE_SHA256",
    "HinesOptimizationAuditConfig",
    "HinesSegmentOptimizationAudit",
    "bounded_segment_prediction",
    "dual_ridge_segment_coefficients",
    "EXPECTED_05G_ARCHIVE_SHA256",
    "BoundedLocalResidualHead",
    "HinesRepresentationForensics",
    "HinesRepresentationForensicsConfig",
    "local_linear_projection",
    "robust_bounded_features",
    "EXPECTED_05H_ARCHIVE_SHA256",
    "HinesStateNormalizationRepair",
    "HinesStateNormalizationRepairConfig",
    "semantic_state_scale_repair",
    "EXPECTED_05I_ARCHIVE_SHA256",
    "HinesNetConSemanticRepair",
    "HinesNetConSemanticRepairConfig",
    "NetConSemanticStateEncoder",
    "netcon_semantic_records",
    "EXPECTED_05IB_ARCHIVE_SHA256",
    "BoundedSynapticStateEncoder",
    "HinesSynapticDomainRepair",
    "HinesSynapticDomainRepairConfig",
]
