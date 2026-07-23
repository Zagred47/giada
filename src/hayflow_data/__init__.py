"""HayFlow dataset validation, reading, sampling, and batching.

This package intentionally does not reuse the NeuronIO pickle contract. Shared
batching utilities may be extracted only when their semantics match.
"""

from .diagnostic_contract import (
    BOUNDARY_INTERVAL_MS,
    DATASET_SCHEMA_VERSION,
    DIAGNOSTIC_DATASET_V1_SCHEMA_VERSION,
    DIAGNOSTIC_SPLITS,
    BurnInCriteria,
    InputAction,
    ProtocolTrajectory,
    estimate_dataset_size_bytes,
    schema_record,
    stable_split,
    validate_input_actions,
    validate_split_isolation,
    write_json,
)
from .hdf5_transition_store import TransitionH5Writer, validate_hdf5_store

__all__ = [
    "BOUNDARY_INTERVAL_MS",
    "BurnInCriteria",
    "DATASET_SCHEMA_VERSION",
    "DIAGNOSTIC_DATASET_V1_SCHEMA_VERSION",
    "DIAGNOSTIC_SPLITS",
    "estimate_dataset_size_bytes",
    "InputAction",
    "ProtocolTrajectory",
    "schema_record",
    "stable_split",
    "TransitionH5Writer",
    "validate_input_actions",
    "validate_split_isolation",
    "validate_hdf5_store",
    "write_json",
]
