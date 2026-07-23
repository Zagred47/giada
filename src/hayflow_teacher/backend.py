from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, Sequence

from ..hayflow_schema import TeacherManifest


@dataclass(frozen=True)
class TeacherSnapshot:
    """Opaque native snapshot; its payload must remain lossless for restore."""

    simulation_time_ms: float
    payload: bytes
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BoundaryState:
    """Complete named state recorded at a one-millisecond boundary."""

    simulation_time_ms: float
    values: Mapping[str, Sequence[float]]


class TeacherBackend(Protocol):
    """Runtime boundary between HayFlow tooling and a NEURON teacher."""

    def build_manifest(self) -> TeacherManifest:
        ...

    def read_boundary_state(self) -> BoundaryState:
        ...

    def snapshot(self) -> TeacherSnapshot:
        ...

    def restore(self, snapshot: TeacherSnapshot) -> None:
        ...

    def advance_to(self, simulation_time_ms: float) -> None:
        ...
