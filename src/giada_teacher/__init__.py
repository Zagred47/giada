"""Teacher-introspection utilities for GIADA."""

from .mechanism_inventory import build_inventory, parse_mod_file, write_inventory
from .causal_classification import build_causal_classification, validate_classification, write_causal_classification
from .atomic_data_contract import build_atomic_data_contract, validate_atomic_data_contract, write_atomic_data_contract
from .double_oracle import (
    ExtractedGateFormula,
    NeuronIsolatedGateOracle,
    OracleUnavailable,
    compile_nmodl,
    run_double_oracle,
    write_double_oracle_report,
)
from .domain_splits import (
    AtomicDomainSplitConfig,
    build_atomic_domain_splits,
    render_atomic_domain_splits_markdown,
)
from .gpu_baseline_contract import (
    build_gpu_baseline_contract,
    validate_gpu_baseline_contract,
    render_gpu_baseline_markdown,
)
from .gpu_baseline_runtime import (
    paired_index_stream,
    paired_index_generator,
    configure_torch_runtime,
    environment_manifest,
    benchmark_cuda,
)
from .atomic_gate_playground import (
    AtomicGateTaskConfig,
    materialize_atomic_gate_dataset,
    build_atomic_gate_models,
    train_and_select_atomic_gate,
    evaluate_frozen_atomic_gate,
)
from .atomic_gate_diagnosis import (
    GateMDiagnosisConfig,
    prepare_gate_m_diagnosis,
    run_gate_m_diagnosis,
    evaluate_gate_m_diagnosis,
)
from .atomic_gate_h_identifiability import (
    GateHIdentifiabilityConfig,
    prepare_gate_h_identifiability,
    run_gate_h_identifiability,
    evaluate_gate_h_identifiability,
    verified_task2_artifact_root,
)
from .joint_gate_cell_playground import (
    JointGateCellConfig,
    prepare_joint_gate_dataset,
    run_joint_gate_playground,
    evaluate_joint_gate_playground,
)
from .joint_gate_optimization_diagnosis import (
    JointGateOptimizationDiagnosisConfig,
    augment_joint_gate_rate_targets,
    run_joint_gate_optimization_diagnosis,
)
from .joint_gate_symmetric_confirmation import (
    JointGateSymmetricConfirmationConfig,
    prepare_joint_gate_symmetric_confirmation,
    train_vectorized_symmetric_joint_gate,
    evaluate_frozen_symmetric_joint_gate,
    verified_task3b_diagnosis_root,
    freeze_task3c_from_task3b,
)
from .joint_gate_generalization_diagnosis import (
    JointGateGeneralizationDiagnosisConfig,
    prepare_joint_gate_generalization_diagnosis,
    run_joint_gate_generalization_diagnosis,
)
from .joint_gate_full_repair_confirmation import (
    JointGateFullRepairConfirmationConfig,
    verified_task3d_root,
    freeze_full_repair_from_task3d,
    evaluate_frozen_full_repair,
)
from .primitive_matrix_playground import (
    PrimitiveMatrixConfig,
    verified_task3e_root,
    prepare_primitive_matrix,
    train_and_freeze_primitive_matrix,
    evaluate_frozen_primitive_matrix,
)
from .primitive_scaling import (
    PrimitiveScalingConfig,
    verified_task4_root,
    prepare_scaling_data,
    train_and_freeze_scaling,
    evaluate_frozen_scaling,
)

__all__ = [
    "build_inventory", "parse_mod_file", "write_inventory",
    "build_causal_classification", "validate_classification", "write_causal_classification",
    "build_atomic_data_contract", "validate_atomic_data_contract", "write_atomic_data_contract",
    "ExtractedGateFormula", "NeuronIsolatedGateOracle", "OracleUnavailable",
    "compile_nmodl", "run_double_oracle", "write_double_oracle_report",
    "AtomicDomainSplitConfig", "build_atomic_domain_splits",
    "render_atomic_domain_splits_markdown",
    "build_gpu_baseline_contract", "validate_gpu_baseline_contract",
    "render_gpu_baseline_markdown",
    "paired_index_stream", "paired_index_generator",
    "configure_torch_runtime", "environment_manifest",
    "benchmark_cuda",
    "AtomicGateTaskConfig", "materialize_atomic_gate_dataset",
    "build_atomic_gate_models", "train_and_select_atomic_gate",
    "evaluate_frozen_atomic_gate",
    "GateMDiagnosisConfig", "prepare_gate_m_diagnosis",
    "run_gate_m_diagnosis", "evaluate_gate_m_diagnosis",
    "GateHIdentifiabilityConfig", "prepare_gate_h_identifiability",
    "run_gate_h_identifiability", "evaluate_gate_h_identifiability",
    "verified_task2_artifact_root",
    "JointGateCellConfig", "prepare_joint_gate_dataset",
    "run_joint_gate_playground", "evaluate_joint_gate_playground",
    "JointGateOptimizationDiagnosisConfig", "augment_joint_gate_rate_targets",
    "run_joint_gate_optimization_diagnosis",
    "JointGateSymmetricConfirmationConfig", "prepare_joint_gate_symmetric_confirmation",
    "train_vectorized_symmetric_joint_gate", "evaluate_frozen_symmetric_joint_gate",
    "verified_task3b_diagnosis_root", "freeze_task3c_from_task3b",
    "JointGateGeneralizationDiagnosisConfig",
    "prepare_joint_gate_generalization_diagnosis",
    "run_joint_gate_generalization_diagnosis",
    "JointGateFullRepairConfirmationConfig", "verified_task3d_root",
    "freeze_full_repair_from_task3d", "evaluate_frozen_full_repair",
    "PrimitiveMatrixConfig", "verified_task3e_root", "prepare_primitive_matrix",
    "train_and_freeze_primitive_matrix", "evaluate_frozen_primitive_matrix",
    "PrimitiveScalingConfig", "verified_task4_root", "prepare_scaling_data",
    "train_and_freeze_scaling", "evaluate_frozen_scaling",
]
