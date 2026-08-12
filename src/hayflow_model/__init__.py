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
]
