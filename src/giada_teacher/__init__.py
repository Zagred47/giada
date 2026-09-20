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
    configure_torch_runtime,
    environment_manifest,
    benchmark_cuda,
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
    "paired_index_stream", "configure_torch_runtime", "environment_manifest",
    "benchmark_cuda",
]
