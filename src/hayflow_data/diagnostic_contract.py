"""Dependency-light contracts for the first HayFlow transition dataset.

The teacher runtime writes these contracts, while training code can import and
validate them without importing NEURON.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


DATASET_SCHEMA_VERSION = "0.1.0"
BOUNDARY_INTERVAL_MS = 1.0


@dataclass(frozen=True)
class BurnInCriteria:
    """Convergence criteria evaluated at one-millisecond boundaries."""

    voltage_delta_mv: float = 1e-3
    calcium_relative_delta: float = 1e-4
    slow_state_delta: float = 1e-5
    consecutive_ms: int = 20
    maximum_duration_ms: int = 5000
    calcium_floor: float = 1e-12
    slow_mechanisms: Tuple[str, ...] = ("CaDynamics_E2", "Ih", "Im")

    def validate(self) -> None:
        positive = {
            "voltage_delta_mv": self.voltage_delta_mv,
            "calcium_relative_delta": self.calcium_relative_delta,
            "slow_state_delta": self.slow_state_delta,
            "calcium_floor": self.calcium_floor,
        }
        for name, value in positive.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if self.consecutive_ms <= 0:
            raise ValueError("consecutive_ms must be positive")
        if self.maximum_duration_ms < self.consecutive_ms:
            raise ValueError(
                "maximum_duration_ms must be at least consecutive_ms"
            )
        if not self.slow_mechanisms:
            raise ValueError("at least one slow mechanism is required")


@dataclass(frozen=True)
class InputAction:
    """One ordered action inside a one-millisecond transition."""

    kind: str
    offset_ms: float
    synapse_id: Optional[int] = None
    weight_multiplier: float = 1.0
    duration_ms: Optional[float] = None
    amplitude_na: Optional[float] = None
    release_observed: Optional[bool] = None
    rng_sequence_before: Optional[float] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def validate(self, step_ms: float = BOUNDARY_INTERVAL_MS) -> None:
        if self.kind not in {"synaptic_event", "somatic_current"}:
            raise ValueError(f"unsupported input kind {self.kind!r}")
        if not 0.0 <= self.offset_ms < step_ms:
            raise ValueError("input offset must be inside [0, step_ms)")
        if self.kind == "synaptic_event":
            if self.synapse_id is None or self.synapse_id < 0:
                raise ValueError("synaptic events require a non-negative id")
            if self.duration_ms is not None or self.amplitude_na is not None:
                raise ValueError("synaptic events cannot define current fields")
        else:
            if self.synapse_id is not None:
                raise ValueError("somatic current cannot reference a synapse")
            if self.duration_ms is None or self.duration_ms <= 0.0:
                raise ValueError("somatic current requires positive duration")
            if self.offset_ms + self.duration_ms > step_ms + 1e-12:
                raise ValueError("somatic current extends past the transition")
            if self.amplitude_na is None:
                raise ValueError("somatic current requires amplitude_na")
        if self.weight_multiplier <= 0.0:
            raise ValueError("weight_multiplier must be positive")

    def to_dict(self) -> Dict[str, Any]:
        self.validate()
        return dict(asdict(self))


@dataclass(frozen=True)
class ProtocolTrajectory:
    """A short trajectory kept intact in exactly one diagnostic split."""

    trajectory_id: str
    category: str
    protocol: str
    seed: int
    duration_ms: int
    split: str
    actions_by_step: Mapping[int, Tuple[InputAction, ...]] = field(
        default_factory=dict
    )
    event_enriched: bool = False

    def validate(self) -> None:
        if not self.trajectory_id:
            raise ValueError("trajectory_id is required")
        if self.category not in {
            "rest_subthreshold",
            "local_synaptic",
            "somatic_events",
            "dendritic_events",
        }:
            raise ValueError(f"unsupported category {self.category!r}")
        if self.split not in {"train", "validation", "test"}:
            raise ValueError(f"unsupported split {self.split!r}")
        if self.duration_ms <= 0:
            raise ValueError("duration_ms must be positive")
        for step, actions in self.actions_by_step.items():
            if not 0 <= int(step) < self.duration_ms:
                raise ValueError("action step lies outside trajectory")
            offsets = []
            for action in actions:
                action.validate()
                offsets.append(action.offset_ms)
            if offsets != sorted(offsets):
                raise ValueError("actions in a step must be time ordered")


def stable_split(seed: int, protocol: str) -> str:
    """Assign a whole seed/protocol pair without leaking trajectory windows."""

    digest = hashlib.sha256(f"{int(seed)}:{protocol}".encode("utf-8")).digest()
    bucket = int.from_bytes(digest[:8], "big") % 100
    if bucket < 70:
        return "train"
    if bucket < 85:
        return "validation"
    return "test"


def validate_split_isolation(rows: Iterable[Mapping[str, Any]]) -> None:
    """Reject seed/protocol or trajectory leakage across diagnostic splits."""

    trajectory_splits: Dict[str, str] = {}
    pair_splits: Dict[Tuple[int, str], str] = {}
    for row in rows:
        split = str(row["split"])
        trajectory = str(row["trajectory_id"])
        pair = (int(row["seed"]), str(row["protocol"]))
        previous = trajectory_splits.setdefault(trajectory, split)
        if previous != split:
            raise ValueError(f"trajectory {trajectory!r} leaks across splits")
        previous = pair_splits.setdefault(pair, split)
        if previous != split:
            raise ValueError(f"seed/protocol {pair!r} leaks across splits")


def validate_input_actions(actions: Sequence[InputAction]) -> None:
    offsets = []
    for action in actions:
        action.validate()
        offsets.append(action.offset_ms)
    if offsets != sorted(offsets):
        raise ValueError("transition inputs must be ordered by offset_ms")


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def estimate_dataset_size_bytes(
    transition_count: int,
    state_width: int,
    microtrace_width: int,
    microtrace_samples: int,
    segment_count: int,
    float_bytes: int = 4,
) -> Dict[str, int]:
    """Return an uncompressed planning estimate, including one-million scale."""

    per_transition = float_bytes * (
        2 * state_width
        + microtrace_samples * microtrace_width
        + microtrace_samples * segment_count
        + 3 * segment_count
    )
    return {
        "estimated_uncompressed_bytes_per_transition": int(per_transition),
        "estimated_uncompressed_bytes_for_dataset": int(
            transition_count * per_transition
        ),
        "estimated_uncompressed_bytes_per_million_transitions": int(
            1_000_000 * per_transition
        ),
    }


def schema_record(
    *,
    variable_id: str,
    category: str,
    index: int,
    scope: str,
    owner_id: Optional[int],
    mechanism: str,
    variable: str,
    kind: str,
    unit: Optional[str],
) -> Dict[str, Any]:
    """Create one stable `(owner, mechanism, variable)` index record."""

    return {
        "variable_id": variable_id,
        "category": category,
        "index": int(index),
        "scope": scope,
        "owner_id": owner_id,
        "mechanism": mechanism,
        "variable": variable,
        "kind": kind,
        "unit": unit,
    }
