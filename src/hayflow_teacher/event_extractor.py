"""Configurable diagnostic event extraction for teacher microtraces.

These definitions are deliberately versioned and provisional.  The notebook
plots every detected event class so thresholds can be reviewed before they are
treated as biological labels.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


EVENT_DETECTOR_VERSION = "diagnostic-v0.2.0"


@dataclass(frozen=True)
class EventDefinition:
    kind: str
    signal: str
    segment_id: int
    region: str
    threshold: float
    reset_threshold: float
    min_duration_ms: float = 0.0
    reference_window_ms: float = 1.0
    requires_kind: Optional[str] = None
    maximum_delay_ms: Optional[float] = None
    maximum_event_duration_ms: Optional[float] = None
    unit: str = "mV"
    detector_version: str = EVENT_DETECTOR_VERSION

    def validate(self) -> None:
        if self.kind not in {
            "axonal_spike",
            "somatic_spike",
            "backpropagating_ap",
            "calcium_spike",
            "nmda_spike",
            "nmda_plateau",
        }:
            raise ValueError(f"unsupported event kind {self.kind!r}")
        if not self.signal:
            raise ValueError("signal is required")
        if self.segment_id < 0:
            raise ValueError("segment_id must be non-negative")
        if self.reset_threshold > self.threshold:
            raise ValueError("reset_threshold cannot exceed onset threshold")
        if self.min_duration_ms < 0.0:
            raise ValueError("min_duration_ms cannot be negative")
        if self.reference_window_ms <= 0.0:
            raise ValueError("reference_window_ms must be positive")
        if self.requires_kind and self.maximum_delay_ms is None:
            raise ValueError("linked events require maximum_delay_ms")
        if (
            self.maximum_event_duration_ms is not None
            and self.maximum_event_duration_ms < self.min_duration_ms
        ):
            raise ValueError(
                "maximum_event_duration_ms cannot be below min_duration_ms"
            )

    def to_dict(self) -> Dict[str, Any]:
        self.validate()
        return asdict(self)


def default_event_definitions(
    representatives: Mapping[str, int],
) -> List[EventDefinition]:
    """Return explicit starting hypotheses, not final biological definitions."""

    return [
        EventDefinition(
            "axonal_spike", "ais", representatives["ais"], "ais", 0.0, -20.0
        ),
        EventDefinition(
            "somatic_spike", "soma", representatives["soma"], "soma", 0.0, -20.0
        ),
        EventDefinition(
            "backpropagating_ap",
            "trunk",
            representatives["trunk"],
            "apical_trunk",
            -20.0,
            -40.0,
            requires_kind="somatic_spike",
            maximum_delay_ms=3.0,
        ),
        EventDefinition(
            "calcium_spike",
            "hot_zone",
            representatives["hot_zone"],
            "hot_zone",
            -20.0,
            -40.0,
            min_duration_ms=2.0,
        ),
        EventDefinition(
            "nmda_spike",
            "nexus",
            representatives["nexus"],
            "nexus",
            -40.0,
            -50.0,
            min_duration_ms=1.0,
        ),
        EventDefinition(
            "nmda_plateau",
            "tuft",
            representatives["tuft"],
            "tuft",
            -40.0,
            -50.0,
            min_duration_ms=10.0,
        ),
    ]


def extract_events(
    time_ms: Sequence[float],
    traces: Mapping[str, Sequence[float]],
    definitions: Iterable[EventDefinition],
) -> List[Dict[str, Any]]:
    """Extract threshold-duration events on an absolute time grid."""

    import numpy as np

    time = np.asarray(time_ms, dtype=float)
    if time.ndim != 1 or time.size < 2:
        raise ValueError("time_ms must be a one-dimensional grid")
    if not np.isfinite(time).all() or not (np.diff(time) > 0).all():
        raise ValueError("time_ms must be finite and strictly increasing")

    definitions = list(definitions)
    events: List[Dict[str, Any]] = []
    for definition in definitions:
        definition.validate()
        if definition.signal not in traces:
            raise KeyError(f"missing event signal {definition.signal!r}")
        values = np.asarray(traces[definition.signal], dtype=float)
        if values.shape != time.shape or not np.isfinite(values).all():
            raise ValueError(
                f"signal {definition.signal!r} does not match time grid"
            )

        cursor = 0
        while cursor < len(time):
            crossings = np.flatnonzero(values[cursor:] >= definition.threshold)
            if not crossings.size:
                break
            onset_index = cursor + int(crossings[0])
            below = np.flatnonzero(
                values[onset_index + 1 :] <= definition.reset_threshold
            )
            right_censored = not bool(below.size)
            offset_index = (
                onset_index + 1 + int(below[0])
                if below.size
                else len(time) - 1
            )
            peak_index = onset_index + int(
                np.argmax(values[onset_index : offset_index + 1])
            )
            duration = float(time[offset_index] - time[onset_index])
            cursor = max(offset_index + 1, onset_index + 1)
            if duration + 1e-12 < definition.min_duration_ms:
                continue
            if (
                definition.maximum_event_duration_ms is not None
                and duration > definition.maximum_event_duration_ms + 1e-12
            ):
                continue

            baseline_start = float(time[onset_index]) - definition.reference_window_ms
            baseline_mask = (time >= baseline_start) & (time < time[onset_index])
            baseline = (
                float(np.median(values[baseline_mask]))
                if baseline_mask.any()
                else float(values[onset_index])
            )
            events.append(
                {
                    "kind": definition.kind,
                    "segment_id": int(definition.segment_id),
                    "region": definition.region,
                    "signal": definition.signal,
                    "onset_ms": float(time[onset_index]),
                    "peak_ms": float(time[peak_index]),
                    "offset_ms": float(time[offset_index]),
                    "duration_ms": duration,
                    "right_censored": right_censored,
                    "duration_is_lower_bound": right_censored,
                    "amplitude": float(values[peak_index] - baseline),
                    "peak_value": float(values[peak_index]),
                    "unit": definition.unit,
                    "detector_version": definition.detector_version,
                    "rule": "threshold_hysteresis_duration",
                    "parameters": definition.to_dict(),
                }
            )

    by_kind: Dict[str, List[Dict[str, Any]]] = {}
    for event in events:
        by_kind.setdefault(event["kind"], []).append(event)
    filtered: List[Dict[str, Any]] = []
    definitions_by_kind = {item.kind: item for item in definitions}
    for event in events:
        definition = definitions_by_kind[event["kind"]]
        if not definition.requires_kind:
            filtered.append(event)
            continue
        candidates = by_kind.get(definition.requires_kind, [])
        delays = [
            event["onset_ms"] - candidate["onset_ms"]
            for candidate in candidates
        ]
        valid = [
            delay
            for delay in delays
            if 0.0 <= delay <= float(definition.maximum_delay_ms)
        ]
        if valid:
            event = dict(event)
            event["linked_kind"] = definition.requires_kind
            event["linked_delay_ms"] = float(min(valid))
            filtered.append(event)
    return sorted(filtered, key=lambda row: (row["onset_ms"], row["kind"]))


def annotate_backpropagation(
    time_ms: Sequence[float],
    traces: Mapping[str, Sequence[float]],
    events: Sequence[Mapping[str, Any]],
    *,
    regional_distances_um: Mapping[str, float],
    threshold_mv: float = -20.0,
    maximum_delay_ms: float = 5.0,
) -> List[Dict[str, Any]]:
    """Require outward temporal order and attach regional bAP propagation data."""

    import numpy as np

    time = np.asarray(time_ms, dtype=float)
    path = [name for name in ("soma", "trunk", "nexus", "tuft") if name in traces]
    basal_path = [name for name in ("soma", "basal") if name in traces]
    result: List[Dict[str, Any]] = []
    for raw in events:
        if raw["kind"] != "backpropagating_ap":
            result.append(dict(raw))
            continue
        event = dict(raw)
        soma_onset = float(event["onset_ms"] - event.get("linked_delay_ms", 0.0))

        def propagation_rows(labels: Sequence[str]) -> List[Dict[str, float]]:
            rows = []
            previous_onset = soma_onset - 1.0e-12
            for label in labels:
                values = np.asarray(traces[label], dtype=float)
                mask = (
                    (time >= soma_onset)
                    & (time <= soma_onset + float(maximum_delay_ms))
                    & (values >= float(threshold_mv))
                )
                indices = np.flatnonzero(mask)
                if not indices.size:
                    break
                index = int(indices[0])
                onset = float(time[index])
                if onset + 1.0e-12 < previous_onset:
                    break
                local = (time >= onset) & (time <= soma_onset + maximum_delay_ms)
                rows.append(
                    {
                        "region": label,
                        "onset_ms": onset,
                        "delay_from_soma_ms": onset - soma_onset,
                        "peak_mv": float(np.max(values[local])),
                        "distance_from_soma_um": float(
                            regional_distances_um.get(label, 0.0)
                        ),
                    }
                )
                previous_onset = onset
            return rows

        apical = propagation_rows(path)
        basal = propagation_rows(basal_path)
        reached_trunk = any(row["region"] == "trunk" for row in apical)
        if not reached_trunk:
            continue
        combined = apical + [row for row in basal if row["region"] != "soma"]
        event.update(
            {
                "origin": "soma",
                "propagation_rule": (
                    "ordered regional threshold crossings after somatic spike"
                ),
                "regional_propagation": combined,
                "maximum_distance_um": max(
                    row["distance_from_soma_um"] for row in combined
                ),
                "propagation_failure": len(apical) < len(path),
                "attenuation_mv": (
                    float(apical[0]["peak_mv"] - apical[-1]["peak_mv"])
                    if len(apical) > 1
                    else 0.0
                ),
            }
        )
        result.append(event)
    return sorted(result, key=lambda row: (row["onset_ms"], row["kind"]))


def event_ids_by_transition(
    events: Sequence[Mapping[str, Any]],
    transition_starts_ms: Sequence[float],
    step_ms: float = 1.0,
) -> List[List[int]]:
    """Assign an event to the transition containing its onset."""

    result: List[List[int]] = []
    for start in transition_starts_ms:
        stop = float(start) + float(step_ms)
        result.append(
            [
                event_id
                for event_id, event in enumerate(events)
                if float(start) <= float(event["onset_ms"]) < stop
            ]
        )
    return result
