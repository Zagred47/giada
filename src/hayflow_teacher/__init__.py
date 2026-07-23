"""NEURON-facing adapters used only for teacher instrumentation."""

from .backend import BoundaryState, TeacherBackend, TeacherSnapshot
from .neuron_manifest import (
    NeuronManifestConfig,
    NeuronManifestExtractor,
    NeuronSynapseBinding,
)

__all__ = [
    "BoundaryState",
    "NeuronManifestConfig",
    "NeuronManifestExtractor",
    "NeuronSynapseBinding",
    "TeacherBackend",
    "TeacherSnapshot",
]
