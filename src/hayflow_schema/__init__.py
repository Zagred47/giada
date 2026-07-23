"""Dependency-light contracts shared by HayFlow generation and training."""

from .events import EventKind, EventLabel
from .manifest import (
    SCHEMA_VERSION,
    MechanismVariable,
    MorphologicalRegion,
    SectionManifest,
    SegmentManifest,
    SynapseComponent,
    SynapseManifest,
    TeacherManifest,
    VariableKind,
    VariableScope,
)

__all__ = [
    "SCHEMA_VERSION",
    "EventKind",
    "EventLabel",
    "MechanismVariable",
    "MorphologicalRegion",
    "SectionManifest",
    "SegmentManifest",
    "SynapseComponent",
    "SynapseManifest",
    "TeacherManifest",
    "VariableKind",
    "VariableScope",
]
