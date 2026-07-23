"""NEURON-facing adapters used only for teacher instrumentation."""

from .backend import BoundaryState, TeacherBackend, TeacherSnapshot
from .audit import (
    detect_spikes,
    git_commit,
    load_source_functions,
    repository_file_record,
    sha256_file,
    validate_parent_tree,
    write_json,
)
from .neuron_manifest import (
    NeuronManifestConfig,
    NeuronManifestExtractor,
    NeuronSynapseBinding,
)
from .audit_runtime import TeacherAuditSession, resolve_audit_repositories
from .event_extractor import (
    EVENT_DETECTOR_VERSION,
    EventDefinition,
    default_event_definitions,
    event_ids_by_transition,
    extract_events,
)
from .diagnostic_dataset import (
    DiagnosticDatasetSession,
    expected_audit_hashes,
)

__all__ = [
    "BoundaryState",
    "default_event_definitions",
    "detect_spikes",
    "DiagnosticDatasetSession",
    "EVENT_DETECTOR_VERSION",
    "EventDefinition",
    "event_ids_by_transition",
    "extract_events",
    "expected_audit_hashes",
    "git_commit",
    "load_source_functions",
    "NeuronManifestConfig",
    "NeuronManifestExtractor",
    "NeuronSynapseBinding",
    "repository_file_record",
    "sha256_file",
    "TeacherBackend",
    "TeacherAuditSession",
    "TeacherSnapshot",
    "validate_parent_tree",
    "write_json",
    "resolve_audit_repositories",
]
