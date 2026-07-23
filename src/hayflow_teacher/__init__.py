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

__all__ = [
    "BoundaryState",
    "detect_spikes",
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
