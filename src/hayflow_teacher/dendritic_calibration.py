"""Controlled dendritic-event protocol calibration for the canonical teacher.

The calibration changes only event schedules and the number of canonical
synapses recruited.  It never changes point-process parameters, NetCon
weights, membrane mechanisms, morphology, or solver tolerances.
"""

from __future__ import annotations

import json
import shutil
from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from ..hayflow_data import InputAction, write_json
from .audit_runtime import PINNED_TEACHER_COMMIT
from .event_extractor import EVENT_DETECTOR_VERSION, extract_events


DENDRITIC_CALIBRATION_SCHEMA_VERSION = "0.3.0"

SELECTION_MODES = {"target_nearest", "branch_cluster"}
EVENT_PROBE_MODES = {"target_representative", "cluster_center"}


class InsufficientCanonicalSynapsesError(RuntimeError):
    """The requested local cluster is larger than the canonical local pool."""


@dataclass(frozen=True)
class DendriticCandidate:
    """One canonical-weight dendritic stimulation candidate."""

    family: str
    target: str
    required_event_kinds: Tuple[str, ...]
    synapse_count: int
    burst_count: int
    events_per_synapse_per_burst: int = 1
    burst_start_ms: int = 3
    burst_interval_ms: int = 1
    pair_with_somatic_spike: bool = False
    maximum_tree_distance_um: Optional[float] = None
    forbidden_event_kinds: Tuple[str, ...] = ()
    selection_mode: str = "target_nearest"
    event_probe_mode: str = "target_representative"
    event_probe_kinds: Tuple[str, ...] = ()
    event_window_ms: float = 0.8

    @property
    def candidate_id(self) -> str:
        pairing = "paired" if self.pair_with_somatic_spike else "unpaired"
        selection = (
            "branch" if self.selection_mode == "branch_cluster" else "nearest"
        )
        window_us = int(round(self.event_window_ms * 1000.0))
        return (
            f"{self.family}-{self.target}-n{self.synapse_count}-"
            f"b{self.burst_count}-r{self.events_per_synapse_per_burst}-"
            f"{pairing}-{selection}-w{window_us:03d}"
        )

    @property
    def event_cost(self) -> int:
        return (
            self.synapse_count
            * self.burst_count
            * self.events_per_synapse_per_burst
        )

    def validate(self, duration_ms: int) -> None:
        if not self.family or not self.target:
            raise ValueError("candidate family and target are required")
        if not self.required_event_kinds:
            raise ValueError("candidate requires at least one event kind")
        if set(self.required_event_kinds) & set(self.forbidden_event_kinds):
            raise ValueError("required and forbidden event kinds overlap")
        if self.selection_mode not in SELECTION_MODES:
            raise ValueError(f"unsupported selection mode {self.selection_mode!r}")
        if self.event_probe_mode not in EVENT_PROBE_MODES:
            raise ValueError(
                f"unsupported event probe mode {self.event_probe_mode!r}"
            )
        if not 0.0 < float(self.event_window_ms) < 1.0:
            raise ValueError("event window must lie strictly inside one ms")
        if (
            self.event_probe_mode == "cluster_center"
            and not set(self.required_event_kinds).issubset(
                set(self.event_probe_kinds)
            )
        ):
            raise ValueError(
                "cluster-center probes must include every required event kind"
            )
        if min(
            self.synapse_count,
            self.burst_count,
            self.events_per_synapse_per_burst,
            self.burst_interval_ms,
        ) <= 0:
            raise ValueError("candidate counts and intervals must be positive")
        final_burst = self.burst_start_ms + (
            self.burst_count - 1
        ) * self.burst_interval_ms
        if self.burst_start_ms < 0 or final_burst >= int(duration_ms):
            raise ValueError("candidate bursts lie outside the trial")
        if (
            self.maximum_tree_distance_um is not None
            and self.maximum_tree_distance_um <= 0.0
        ):
            raise ValueError("maximum tree distance must be positive")


@dataclass(frozen=True)
class SynapseSelection:
    """A deterministic local selection and the segment used as event probe."""

    mode: str
    target_segment_id: int
    center_segment_id: int
    synapse_ids: Tuple[int, ...]
    segment_ids: Tuple[int, ...]
    center_distances_um: Tuple[float, ...]
    center_to_target_distance_um: float

    @property
    def maximum_center_distance_um(self) -> float:
        return float(max(self.center_distances_um, default=0.0))


def candidate_from_mapping(
    family: str,
    family_config: Mapping[str, Any],
    level: Mapping[str, Any],
) -> DendriticCandidate:
    """Build a validated candidate from the versioned YAML structure."""

    return DendriticCandidate(
        family=str(family),
        target=str(family_config["target"]),
        required_event_kinds=tuple(family_config["required_event_kinds"]),
        synapse_count=int(level["synapse_count"]),
        burst_count=int(level["burst_count"]),
        events_per_synapse_per_burst=int(
            level.get("events_per_synapse_per_burst", 1)
        ),
        burst_start_ms=int(family_config.get("burst_start_ms", 3)),
        burst_interval_ms=int(family_config.get("burst_interval_ms", 1)),
        pair_with_somatic_spike=bool(
            family_config.get("pair_with_somatic_spike", False)
        ),
        maximum_tree_distance_um=(
            float(family_config["maximum_tree_distance_um"])
            if family_config.get("maximum_tree_distance_um") is not None
            else None
        ),
        forbidden_event_kinds=tuple(
            family_config.get("forbidden_event_kinds", ())
        ),
        selection_mode=str(
            level.get(
                "selection_mode",
                family_config.get("selection_mode", "target_nearest"),
            )
        ),
        event_probe_mode=str(
            level.get(
                "event_probe_mode",
                family_config.get(
                    "event_probe_mode", "target_representative"
                ),
            )
        ),
        event_probe_kinds=tuple(
            level.get(
                "event_probe_kinds",
                family_config.get(
                    "event_probe_kinds",
                    family_config["required_event_kinds"],
                ),
            )
        ),
        event_window_ms=float(
            level.get(
                "event_window_ms",
                family_config.get("event_window_ms", 0.8),
            )
        ),
    )


def candidate_from_selected_protocol(
    protocol: Mapping[str, Any],
) -> DendriticCandidate:
    """Reconstruct an exact selected candidate for long-horizon replay."""

    return DendriticCandidate(
        family=str(protocol["family"]),
        target=str(protocol["target"]),
        required_event_kinds=tuple(protocol["required_event_kinds"]),
        synapse_count=int(protocol["synapse_count"]),
        burst_count=int(protocol["burst_count"]),
        events_per_synapse_per_burst=int(
            protocol["events_per_synapse_per_burst"]
        ),
        burst_start_ms=int(protocol["burst_start_ms"]),
        burst_interval_ms=int(protocol["burst_interval_ms"]),
        pair_with_somatic_spike=bool(protocol["pair_with_somatic_spike"]),
        maximum_tree_distance_um=(
            None
            if protocol.get("maximum_tree_distance_um") is None
            else float(protocol["maximum_tree_distance_um"])
        ),
        forbidden_event_kinds=tuple(protocol["forbidden_event_kinds"]),
        selection_mode=str(protocol["selection_mode"]),
        event_probe_mode=str(protocol["event_probe_mode"]),
        event_probe_kinds=tuple(protocol["event_probe_kinds"]),
        event_window_ms=float(protocol["event_window_ms"]),
    )


def evenly_spaced_offsets(
    count: int, event_window_ms: float = 0.8
) -> List[float]:
    """Return deterministic event offsets strictly inside one millisecond."""

    if int(count) <= 0:
        raise ValueError("offset count must be positive")
    window = float(event_window_ms)
    if not 0.0 < window < 1.0:
        raise ValueError("event window must lie strictly inside one ms")
    left = 0.5 - 0.5 * window
    return [
        left + window * (index + 0.5) / int(count)
        for index in range(int(count))
    ]


def build_candidate_actions(
    candidate: DendriticCandidate,
    synapse_ids: Sequence[int],
    *,
    duration_ms: int,
    somatic_current_na: Optional[float] = None,
) -> Dict[int, Tuple[InputAction, ...]]:
    """Create ordered canonical-weight inputs for one calibration trial."""

    candidate.validate(duration_ms)
    selected = list(map(int, synapse_ids))
    if len(selected) != candidate.synapse_count:
        raise ValueError("selected synapse count does not match the candidate")
    actions: Dict[int, List[InputAction]] = {}
    event_total = len(selected) * candidate.events_per_synapse_per_burst
    offsets = evenly_spaced_offsets(event_total, candidate.event_window_ms)
    for burst_index in range(candidate.burst_count):
        step = candidate.burst_start_ms + (
            burst_index * candidate.burst_interval_ms
        )
        rows = []
        offset_index = 0
        for _ in range(candidate.events_per_synapse_per_burst):
            for synapse_id in selected:
                rows.append(
                    InputAction(
                        "synaptic_event",
                        offsets[offset_index],
                        synapse_id=synapse_id,
                    )
                )
                offset_index += 1
        actions.setdefault(step, []).extend(rows)

    if candidate.pair_with_somatic_spike:
        if somatic_current_na is None or float(somatic_current_na) <= 0.0:
            raise ValueError("paired candidates require calibrated somatic current")
        first_step = max(0, candidate.burst_start_ms - 1)
        for step in (first_step, candidate.burst_start_ms):
            actions.setdefault(step, []).append(
                InputAction(
                    "somatic_current",
                    0.05,
                    duration_ms=0.9,
                    amplitude_na=float(somatic_current_na),
                )
            )

    ordered = {}
    for step, rows in actions.items():
        rows.sort(
            key=lambda action: (
                float(action.offset_ms),
                action.kind,
                -1 if action.synapse_id is None else int(action.synapse_id),
            )
        )
        ordered[int(step)] = tuple(rows)
    return ordered


class DendriticProtocolCalibrator:
    """Run lightweight, replayable dendritic protocol searches."""

    def __init__(
        self,
        diagnostic_session: Any,
        output_dir: Optional[Path] = None,
        sample_interval_ms: float = 0.025,
    ) -> None:
        self.session = diagnostic_session
        self.audit = diagnostic_session.audit
        self.h = diagnostic_session.h
        self.cvode = diagnostic_session.cvode
        self.np = diagnostic_session.np
        self.output_dir = Path(
            output_dir
            or diagnostic_session.output_dir.parent
            / "dendritic_protocol_calibration"
        ).resolve()
        self.traces_dir = self.output_dir / "traces"
        self.confirmation_traces_dir = (
            self.output_dir / "confirmation_traces"
        )
        self.figures_dir = self.output_dir / "figures"
        for directory in (
            self.output_dir,
            self.traces_dir,
            self.confirmation_traces_dir,
            self.figures_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)
        self.sample_interval_ms = float(sample_interval_ms)
        if self.sample_interval_ms <= 0.0:
            raise ValueError("sample_interval_ms must be positive")
        self.report: Dict[str, Any] = {}

    def _require_ready(self) -> None:
        self.session._require_equilibrium()
        if not self.session.event_definitions:
            raise RuntimeError("event definitions are not initialized")

    def _parent_map(self) -> Dict[int, Optional[int]]:
        result = {}
        for row in self.audit.segment_df.itertuples(index=False):
            parent = row.parent_segment_id
            result[int(row.segment_id)] = (
                None if self.audit.pd.isna(parent) else int(parent)
            )
        return result

    def _distance_map(self) -> Dict[int, float]:
        return {
            int(row.segment_id): float(row.distance_from_soma_um)
            for row in self.audit.segment_df.itertuples(index=False)
        }

    @staticmethod
    def _ancestors(
        segment_id: int, parents: Mapping[int, Optional[int]]
    ) -> List[int]:
        result = []
        cursor: Optional[int] = int(segment_id)
        seen = set()
        while cursor is not None:
            if cursor in seen:
                raise RuntimeError("segment parent graph contains a cycle")
            seen.add(cursor)
            result.append(cursor)
            cursor = parents[cursor]
        return result

    @classmethod
    def tree_distance_um(
        cls,
        first: int,
        second: int,
        parents: Mapping[int, Optional[int]],
        soma_distances_um: Mapping[int, float],
    ) -> float:
        """Distance between two segment centers through the morphology tree."""

        first_path = cls._ancestors(int(first), parents)
        second_ancestors = set(cls._ancestors(int(second), parents))
        lca = next(item for item in first_path if item in second_ancestors)
        return float(
            soma_distances_um[int(first)]
            + soma_distances_um[int(second)]
            - 2.0 * soma_distances_um[lca]
        )

    @classmethod
    def compact_synapse_cluster(
        cls,
        candidates: Sequence[Tuple[int, int]],
        count: int,
        parents: Mapping[int, Optional[int]],
        soma_distances_um: Mapping[int, float],
        *,
        target_segment_id: int,
        maximum_center_distance_um: Optional[float],
    ) -> Tuple[int, List[Tuple[float, int, int]]]:
        """Choose the most compact tree-local cluster in a candidate pool."""

        requested = int(count)
        if requested <= 0:
            raise ValueError("cluster count must be positive")
        if len(candidates) < requested:
            raise InsufficientCanonicalSynapsesError(
                f"candidate pool has only {len(candidates)} canonical "
                f"excitatory synapses, fewer than requested {requested}"
            )

        best: Optional[
            Tuple[
                Tuple[float, float, float, int, int],
                int,
                List[Tuple[float, int, int]],
            ]
        ] = None
        best_available = 0
        ancestors = {
            int(segment_id): set(cls._ancestors(int(segment_id), parents))
            for segment_id, _ in candidates
        }
        for center_segment_id, center_synapse_id in candidates:
            center_ancestors = ancestors[int(center_segment_id)]
            ranked = sorted(
                (
                    cls.tree_distance_um(
                        int(segment_id),
                        int(center_segment_id),
                        parents,
                        soma_distances_um,
                    ),
                    int(segment_id),
                    int(synapse_id),
                )
                for segment_id, synapse_id in candidates
                if int(segment_id) in center_ancestors
                or int(center_segment_id) in ancestors[int(segment_id)]
            )
            if maximum_center_distance_um is None:
                local = ranked
            else:
                local = [
                    row
                    for row in ranked
                    if row[0] <= float(maximum_center_distance_um)
                ]
            best_available = max(best_available, len(local))
            if len(local) < requested:
                continue
            selected = local[:requested]
            target_distance = cls.tree_distance_um(
                int(center_segment_id),
                int(target_segment_id),
                parents,
                soma_distances_um,
            )
            score = (
                float(target_distance),
                float(selected[-1][0]),
                float(sum(item[0] for item in selected)),
                int(center_segment_id),
                int(center_synapse_id),
            )
            if best is None or score < best[0]:
                best = (score, int(center_segment_id), selected)

        if best is None:
            radius_note = (
                "without a radius limit"
                if maximum_center_distance_um is None
                else f"within {float(maximum_center_distance_um):g} um"
            )
            raise InsufficientCanonicalSynapsesError(
                f"best branch-local pool has only {best_available} canonical "
                f"excitatory synapses {radius_note}, fewer than requested "
                f"{requested}"
            )
        return best[1], best[2]

    def select_synapse_cluster(
        self,
        target: str,
        count: int,
        maximum_tree_distance_um: Optional[float] = None,
        selection_mode: str = "target_nearest",
    ) -> SynapseSelection:
        """Select canonical excitatory synapses with explicit tree geometry."""

        if target not in self.audit.representatives:
            raise KeyError(f"unknown representative target {target!r}")
        if selection_mode not in SELECTION_MODES:
            raise ValueError(f"unsupported selection mode {selection_mode!r}")
        target_id = int(self.audit.representatives[target])
        parents = self._parent_map()
        distances = self._distance_map()
        hot_zone_ids = set(map(int, self.audit.hot_zone_segment_ids))
        region_by_id = {
            int(row.segment_id): str(row.region)
            for row in self.audit.segment_df.itertuples(index=False)
        }

        def eligible(segment_id: int) -> bool:
            region = region_by_id[segment_id]
            if target == "tuft":
                return region == "tuft"
            if target == "hot_zone":
                return segment_id in hot_zone_ids or region in {
                    "nexus",
                    "apical_trunk",
                }
            if target == "nexus":
                return region in {
                    "nexus",
                    "apical_trunk",
                    "apical_oblique",
                }
            return True

        candidates: List[Tuple[int, int]] = []
        for record in self.audit.synapse_records:
            if record["class_name"] != "ProbAMPANMDA2":
                continue
            segment_id = int(record["segment_id"])
            if not eligible(segment_id):
                continue
            candidates.append(
                (segment_id, int(record["synapse_id"]))
            )

        if selection_mode == "branch_cluster":
            center_id, selected = self.compact_synapse_cluster(
                candidates,
                count,
                parents,
                distances,
                target_segment_id=target_id,
                maximum_center_distance_um=maximum_tree_distance_um,
            )
            center_to_target = self.tree_distance_um(
                center_id, target_id, parents, distances
            )
            return SynapseSelection(
                mode=selection_mode,
                target_segment_id=target_id,
                center_segment_id=center_id,
                synapse_ids=tuple(row[2] for row in selected),
                segment_ids=tuple(row[1] for row in selected),
                center_distances_um=tuple(float(row[0]) for row in selected),
                center_to_target_distance_um=float(center_to_target),
            )

        nearest = []
        for segment_id, synapse_id in candidates:
            tree_distance = self.tree_distance_um(
                segment_id, target_id, parents, distances
            )
            if (
                maximum_tree_distance_um is None
                or tree_distance <= float(maximum_tree_distance_um)
            ):
                nearest.append((tree_distance, segment_id, synapse_id))
        nearest.sort()
        if len(nearest) < int(count):
            raise InsufficientCanonicalSynapsesError(
                f"target {target!r} has only {len(nearest)} eligible "
                f"canonical excitatory synapses, fewer than requested {count}"
            )
        selected = nearest[: int(count)]
        return SynapseSelection(
            mode=selection_mode,
            target_segment_id=target_id,
            center_segment_id=target_id,
            synapse_ids=tuple(row[2] for row in selected),
            segment_ids=tuple(row[1] for row in selected),
            center_distances_um=tuple(float(row[0]) for row in selected),
            center_to_target_distance_um=0.0,
        )

    def select_local_excitatory_synapses(
        self,
        target: str,
        count: int,
        maximum_tree_distance_um: Optional[float] = None,
    ) -> List[int]:
        """Backward-compatible target-nearest selection."""

        selection = self.select_synapse_cluster(
            target,
            count,
            maximum_tree_distance_um,
            selection_mode="target_nearest",
        )
        return list(selection.synapse_ids)

    @staticmethod
    def _safe_read(owner: Any, name: str) -> float:
        return float(getattr(owner, name)) if hasattr(owner, name) else 0.0

    def _sample_observables(
        self,
        synapse_ids: Sequence[int],
        event_probe_segment_id: Optional[int] = None,
    ) -> Dict[str, float]:
        values = {
            f"voltage_{label}_mv": float(
                self.audit.live_segments[int(segment_id)].v
            )
            for label, segment_id in self.audit.representatives.items()
        }
        if event_probe_segment_id is not None:
            event_probe = self.audit.live_segments[
                int(event_probe_segment_id)
            ]
            values["voltage_event_probe_mv"] = float(event_probe.v)
            values["cai_event_probe_mM"] = self._safe_read(
                event_probe, "cai"
            )
            values["ica_event_probe_mA_per_cm2"] = self._safe_read(
                event_probe, "ica"
            )
            values["ica_hva_event_probe_mA_per_cm2"] = self._safe_read(
                event_probe, "ica_Ca_HVA"
            )
            values["ica_lva_event_probe_mA_per_cm2"] = self._safe_read(
                event_probe, "ica_Ca_LVAst"
            )
        for label in ("nexus", "hot_zone", "tuft"):
            segment = self.audit.live_segments[
                int(self.audit.representatives[label])
            ]
            values[f"cai_{label}_mM"] = self._safe_read(segment, "cai")
            values[f"ica_{label}_mA_per_cm2"] = self._safe_read(
                segment, "ica"
            )
            values[f"ica_hva_{label}_mA_per_cm2"] = self._safe_read(
                segment, "ica_Ca_HVA"
            )
            values[f"ica_lva_{label}_mA_per_cm2"] = self._safe_read(
                segment, "ica_Ca_LVAst"
            )
        for name in ("g_NMDA", "i_NMDA", "g_AMPA", "i_AMPA"):
            values[f"sum_{name}"] = sum(
                self._safe_read(
                    self.audit.synapse_records[int(synapse_id)][
                        "point_process"
                    ],
                    name,
                )
                for synapse_id in synapse_ids
            )
        values["somatic_current_na"] = float(self.session.somatic_clamp.i)
        return values

    def _sample_one_ms(
        self,
        start_time: float,
        actions: Sequence[InputAction],
        synapse_ids: Sequence[int],
        event_probe_segment_id: Optional[int] = None,
    ) -> Tuple[List[float], Dict[str, List[float]]]:
        times, _, samples = self.session._drive_one_ms(
            start_time,
            actions,
            lambda: self._sample_observables(
                synapse_ids, event_probe_segment_id
            ),
            sample_interval_ms=self.sample_interval_ms,
        )
        rows: Dict[str, List[float]] = {}
        for observed in samples:
            for name, value in observed.items():
                rows.setdefault(name, []).append(float(value))
        return times.tolist(), rows

    def run_trial(
        self,
        candidate: DendriticCandidate,
        seed: int,
        duration_ms: int,
        *,
        trace_directory: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """Run one candidate from the canonical equilibrium snapshot."""

        self._require_ready()
        candidate.validate(duration_ms)
        selection = self.select_synapse_cluster(
            candidate.target,
            candidate.synapse_count,
            candidate.maximum_tree_distance_um,
            candidate.selection_mode,
        )
        synapse_ids = list(selection.synapse_ids)
        event_probe_segment_id = (
            selection.center_segment_id
            if candidate.event_probe_mode == "cluster_center"
            else selection.target_segment_id
        )
        canonical_weights = {
            synapse_id: float(
                self.audit.synapse_records[synapse_id]["binding"].base_weight
            )
            for synapse_id in synapse_ids
        }
        observed_weights = {
            synapse_id: float(
                self.audit.synapse_records[synapse_id]["netcon"].weight[0]
            )
            for synapse_id in synapse_ids
        }
        if observed_weights != canonical_weights:
            raise RuntimeError(
                "canonical NetCon weights differ before dendritic calibration"
            )
        if candidate.pair_with_somatic_spike:
            if self.session.calibrated_somatic_current_na is None:
                self.session.calibrate_somatic_spike_current()
            somatic_current = float(
                self.session.calibrated_somatic_current_na
            )
        else:
            somatic_current = None
        actions_by_step = build_candidate_actions(
            candidate,
            synapse_ids,
            duration_ms=duration_ms,
            somatic_current_na=somatic_current,
        )
        equilibrium_rng = json.loads(
            self.session.equilibrium_rng_path.read_text(encoding="utf-8")
        )
        self.session._restore_native_snapshot(
            self.session.equilibrium_snapshot_path,
            equilibrium_rng["sequences"],
            equilibrium_rng.get("random123_seed", self.session.seed),
        )
        self.session._rekey_rngs(int(seed))

        times: List[float] = []
        traces: Dict[str, List[float]] = {}
        for step in range(int(duration_ms)):
            start_time = float(self.h.t)
            step_times, step_traces = self._sample_one_ms(
                start_time,
                list(actions_by_step.get(step, ())),
                synapse_ids,
                event_probe_segment_id,
            )
            start = 0 if step == 0 else 1
            times.extend(step_times[start:])
            for name, values in step_traces.items():
                traces.setdefault(name, []).extend(values[start:])

        if not self.np.isfinite(self.np.asarray(times, dtype=float)).all():
            raise RuntimeError("calibration time grid contains NaN/Inf")
        if not all(
            self.np.isfinite(self.np.asarray(values, dtype=float)).all()
            for values in traces.values()
        ):
            raise RuntimeError("calibration trace contains NaN/Inf")
        final_weights = {
            synapse_id: float(
                self.audit.synapse_records[synapse_id]["netcon"].weight[0]
            )
            for synapse_id in synapse_ids
        }
        if final_weights != canonical_weights:
            raise RuntimeError(
                "dendritic calibration changed a canonical NetCon weight"
            )

        voltage_traces = {
            label: traces[f"voltage_{label}_mv"]
            for label in self.audit.representatives
        }
        event_definitions = list(self.session.event_definitions)
        if candidate.event_probe_mode == "cluster_center":
            voltage_traces["event_probe"] = traces[
                "voltage_event_probe_mv"
            ]
            probe_kinds = set(candidate.event_probe_kinds)
            event_definitions = [
                replace(
                    definition,
                    signal="event_probe",
                    segment_id=int(event_probe_segment_id),
                    region=f"{candidate.target}_stimulus_cluster",
                )
                if definition.kind in probe_kinds
                else definition
                for definition in event_definitions
            ]
        events = extract_events(
            times, voltage_traces, event_definitions
        )
        counts = Counter(event["kind"] for event in events)
        success = all(
            counts.get(kind, 0) > 0 for kind in candidate.required_event_kinds
        ) and not any(
            counts.get(kind, 0) > 0 for kind in candidate.forbidden_event_kinds
        )
        target = candidate.target
        cai = self.np.asarray(traces[f"cai_{target}_mM"], dtype=float)
        probe_cai = self.np.asarray(
            traces["cai_event_probe_mM"], dtype=float
        )
        nmda_current = self.np.asarray(traces["sum_i_NMDA"], dtype=float)
        result = {
            "candidate_id": candidate.candidate_id,
            "family": candidate.family,
            "target": candidate.target,
            "seed": int(seed),
            "success": bool(success),
            "required_event_kinds": list(candidate.required_event_kinds),
            "forbidden_event_kinds": list(candidate.forbidden_event_kinds),
            "event_counts": dict(sorted(counts.items())),
            "events": events,
            "synapse_count": candidate.synapse_count,
            "burst_count": candidate.burst_count,
            "events_per_synapse_per_burst": (
                candidate.events_per_synapse_per_burst
            ),
            "event_window_ms": candidate.event_window_ms,
            "canonical_synaptic_event_count": candidate.event_cost,
            "canonical_netcon_weights": canonical_weights,
            "canonical_weights_unchanged": True,
            "pair_with_somatic_spike": candidate.pair_with_somatic_spike,
            "somatic_current_na": somatic_current,
            "selection_mode": candidate.selection_mode,
            "event_probe_mode": candidate.event_probe_mode,
            "event_probe_kinds": list(candidate.event_probe_kinds),
            "event_probe_segment_id": int(event_probe_segment_id),
            "event_probe_region": (
                f"{candidate.target}_stimulus_cluster"
                if candidate.event_probe_mode == "cluster_center"
                else candidate.target
            ),
            "selected_synapse_ids": synapse_ids,
            "selected_segment_ids": list(selection.segment_ids),
            "selected_synapse_tree_distances_um": list(
                selection.center_distances_um
            ),
            "maximum_selected_tree_distance_um": (
                selection.maximum_center_distance_um
            ),
            "cluster_center_to_target_distance_um": (
                selection.center_to_target_distance_um
            ),
            "target_peak_voltage_mv": float(
                self.np.max(traces[f"voltage_{target}_mv"])
            ),
            "event_probe_peak_voltage_mv": float(
                self.np.max(traces["voltage_event_probe_mv"])
            ),
            "target_cai_increase_mM": float(self.np.max(cai) - cai[0]),
            "event_probe_cai_increase_mM": float(
                self.np.max(probe_cai) - probe_cai[0]
            ),
            "peak_sum_g_nmda_us": float(
                self.np.max(traces["sum_g_NMDA"])
            ),
            "minimum_sum_i_nmda_na": float(self.np.min(nmda_current)),
            "absolute_nmda_current_integral_na_ms": float(
                self.np.trapz(self.np.abs(nmda_current), times)
            ),
            "input_schedule": {
                str(step): [action.to_dict() for action in actions]
                for step, actions in sorted(actions_by_step.items())
            },
        }
        destination = Path(trace_directory or self.traces_dir)
        destination.mkdir(parents=True, exist_ok=True)
        trace_path = destination / (
            f"{candidate.candidate_id}-seed{int(seed)}.npz"
        )
        self.np.savez_compressed(
            trace_path,
            time_ms=self.np.asarray(times, dtype=float),
            **{
                name: self.np.asarray(values, dtype=float)
                for name, values in traces.items()
            },
        )
        result["trace_path"] = trace_path.relative_to(
            self.output_dir
        ).as_posix()
        return result

    def run(self, config: Mapping[str, Any]) -> Dict[str, Any]:
        """Run staged family sweeps and select the least-cost robust protocols."""

        self._require_ready()
        self._reset_run_outputs()
        duration_ms = int(config["duration_ms"])
        seeds = list(map(int, config["seeds"]))
        required_fraction = float(config["required_success_fraction"])
        if not seeds or not 0.0 < required_fraction <= 1.0:
            raise ValueError("calibration seeds/fraction are invalid")
        if not bool(config.get("preserve_canonical_synaptic_weights", False)):
            raise ValueError(
                "dendritic calibration requires canonical synaptic weights"
            )
        if not bool(config.get("stop_at_first_robust_level", True)):
            raise NotImplementedError(
                "calibration-v0.2 selects the first robust configured level"
            )

        trials = []
        selected: Dict[str, Dict[str, Any]] = {}
        for family, family_config in config["families"].items():
            family_selected = None
            for level in family_config["levels"]:
                candidate = candidate_from_mapping(
                    family, family_config, level
                )
                candidate.validate(duration_ms)
                candidate_trials = []
                try:
                    for seed in seeds:
                        trial = self.run_trial(candidate, seed, duration_ms)
                        trials.append(trial)
                        candidate_trials.append(trial)
                except InsufficientCanonicalSynapsesError as error:
                    trials.append(
                        {
                            "candidate_id": candidate.candidate_id,
                            "family": candidate.family,
                            "target": candidate.target,
                            "skipped": True,
                            "reason": str(error),
                        }
                    )
                    continue
                success_fraction = sum(
                    trial["success"] for trial in candidate_trials
                ) / len(candidate_trials)
                if success_fraction >= required_fraction:
                    family_selected = {
                        "candidate_id": candidate.candidate_id,
                        "family": candidate.family,
                        "target": candidate.target,
                        "required_event_kinds": list(
                            candidate.required_event_kinds
                        ),
                        "forbidden_event_kinds": list(
                            candidate.forbidden_event_kinds
                        ),
                        "synapse_count": candidate.synapse_count,
                        "burst_count": candidate.burst_count,
                        "events_per_synapse_per_burst": (
                            candidate.events_per_synapse_per_burst
                        ),
                        "event_window_ms": candidate.event_window_ms,
                        "burst_start_ms": candidate.burst_start_ms,
                        "burst_interval_ms": candidate.burst_interval_ms,
                        "maximum_tree_distance_um": (
                            candidate.maximum_tree_distance_um
                        ),
                        "pair_with_somatic_spike": (
                            candidate.pair_with_somatic_spike
                        ),
                        "selection_mode": candidate.selection_mode,
                        "event_probe_mode": candidate.event_probe_mode,
                        "event_probe_kinds": list(
                            candidate.event_probe_kinds
                        ),
                        "canonical_weights_unchanged": True,
                        "duration_ms": duration_ms,
                        "success_fraction": success_fraction,
                        "successful_seeds": [
                            trial["seed"]
                            for trial in candidate_trials
                            if trial["success"]
                        ],
                        "trial_trace_paths": [
                            trial["trace_path"] for trial in candidate_trials
                        ],
                        "selected_synapse_ids": candidate_trials[0][
                            "selected_synapse_ids"
                        ],
                        "selected_segment_ids": candidate_trials[0][
                            "selected_segment_ids"
                        ],
                        "maximum_selected_tree_distance_um": (
                            candidate_trials[0][
                                "maximum_selected_tree_distance_um"
                            ]
                        ),
                        "cluster_center_to_target_distance_um": (
                            candidate_trials[0][
                                "cluster_center_to_target_distance_um"
                            ]
                        ),
                        "event_probe_segment_id": candidate_trials[0][
                            "event_probe_segment_id"
                        ],
                        "event_success_fractions": {
                            kind: sum(
                                trial["event_counts"].get(kind, 0) > 0
                                for trial in candidate_trials
                            )
                            / len(candidate_trials)
                            for kind in sorted(
                                {
                                    event_kind
                                    for trial in candidate_trials
                                    for event_kind in trial["event_counts"]
                                }
                            )
                        },
                        "input_schedule_template": candidate_trials[0][
                            "input_schedule"
                        ],
                        "somatic_current_na": candidate_trials[0][
                            "somatic_current_na"
                        ],
                    }
                    break
            if family_selected is not None:
                selected[family] = family_selected

        required_families = [
            name
            for name, family in config["families"].items()
            if bool(family.get("required_for_completion", True))
        ]
        missing = [name for name in required_families if name not in selected]
        required_event_kinds = list(
            map(str, config.get("required_event_kinds_for_completion", ()))
        )
        event_coverage = {}
        for kind in required_event_kinds:
            protocols = [
                {
                    "family": family,
                    "candidate_id": protocol["candidate_id"],
                    "success_fraction": protocol[
                        "event_success_fractions"
                    ].get(kind, 0.0),
                }
                for family, protocol in selected.items()
                if protocol["event_success_fractions"].get(kind, 0.0)
                >= required_fraction
            ]
            event_coverage[kind] = {
                "covered": bool(protocols),
                "protocols": protocols,
            }
        missing_event_kinds = [
            kind
            for kind, coverage in event_coverage.items()
            if not coverage["covered"]
        ]
        self.report = {
            "schema_version": DENDRITIC_CALIBRATION_SCHEMA_VERSION,
            "valid": not missing and not missing_event_kinds,
            "teacher_commit": PINNED_TEACHER_COMMIT,
            "teacher_source_hashes": {
                str(record["path"]): str(record["sha256"])
                for record in self.audit.environment["source_files"]
            },
            "duration_ms": duration_ms,
            "sample_interval_ms": self.sample_interval_ms,
            "seeds": seeds,
            "required_success_fraction": required_fraction,
            "canonical_synaptic_weights_unchanged": True,
            "search_axes": [
                "canonical synapse count",
                "burst count",
                "events per synapse",
                "event synchrony window",
                "tree-local branch cluster",
                "optional somatic-spike pairing",
            ],
            "selected_protocols": selected,
            "required_families": required_families,
            "missing_required_families": missing,
            "required_event_kinds_for_completion": required_event_kinds,
            "event_coverage": event_coverage,
            "missing_required_event_kinds": missing_event_kinds,
            "trial_count": len(
                [trial for trial in trials if not trial.get("skipped")]
            ),
            "skipped_candidate_count": len(
                [trial for trial in trials if trial.get("skipped")]
            ),
            "trials": trials,
        }
        diagnostic_figures = self._plot_diagnostics(selected, trials)
        self.report["diagnostic_figures"] = diagnostic_figures
        write_json(self.output_dir / "calibration_report.json", self.report)
        write_json(
            self.output_dir / "selected_dendritic_protocols.json",
            {
                "schema_version": DENDRITIC_CALIBRATION_SCHEMA_VERSION,
                "teacher_commit": self.report["teacher_commit"],
                "canonical_synaptic_weights_unchanged": True,
                "selected_protocols": selected,
                "event_coverage": event_coverage,
            },
        )
        self._write_artifact_index()
        return self.report

    def _compare_trace_prefix(
        self,
        short_trace_path: Path,
        long_trace_path: Path,
        tolerance: float,
    ) -> Dict[str, Any]:
        """Compare the complete short trace with the long replay prefix."""

        missing_variables = []
        shape_mismatches = []
        per_variable_error = {}
        short_sample_count = 0
        with self.np.load(short_trace_path) as short_data, self.np.load(
            long_trace_path
        ) as long_data:
            short_sample_count = int(short_data["time_ms"].shape[0])
            for name in short_data.files:
                if name not in long_data.files:
                    missing_variables.append(name)
                    continue
                short_values = self.np.asarray(short_data[name], dtype=float)
                long_values = self.np.asarray(long_data[name], dtype=float)
                if (
                    short_values.ndim == 0
                    or short_values.ndim != long_values.ndim
                    or long_values.shape[0] < short_values.shape[0]
                    or long_values.shape[1:] != short_values.shape[1:]
                ):
                    shape_mismatches.append(
                        {
                            "variable": name,
                            "short_shape": list(short_values.shape),
                            "long_shape": list(long_values.shape),
                        }
                    )
                    continue
                long_prefix = long_values[: short_values.shape[0]]
                if long_prefix.shape != short_values.shape:
                    shape_mismatches.append(
                        {
                            "variable": name,
                            "short_shape": list(short_values.shape),
                            "long_prefix_shape": list(long_prefix.shape),
                        }
                    )
                    continue
                error = float(
                    self.np.max(self.np.abs(short_values - long_prefix))
                )
                per_variable_error[name] = error

        maximum_error = max(per_variable_error.values(), default=0.0)
        return {
            "valid": (
                not missing_variables
                and not shape_mismatches
                and maximum_error <= float(tolerance)
            ),
            "tolerance": float(tolerance),
            "short_sample_count": short_sample_count,
            "maximum_absolute_error": maximum_error,
            "per_variable_max_absolute_error": per_variable_error,
            "missing_variables": missing_variables,
            "shape_mismatches": shape_mismatches,
        }

    def confirm_selected_protocols(
        self,
        *,
        duration_ms: int,
        overlap_tolerance: float = 1e-9,
    ) -> Dict[str, Any]:
        """Replay selected protocols long enough to observe event recovery."""

        self._require_ready()
        if not self.report or not self.report.get("selected_protocols"):
            raise RuntimeError("run calibration before long-horizon confirmation")
        self._upgrade_event_metadata_for_confirmation()
        short_duration_ms = int(self.report["duration_ms"])
        duration_ms = int(duration_ms)
        if duration_ms <= short_duration_ms:
            raise ValueError(
                "confirmation duration must exceed calibration duration"
            )
        if float(overlap_tolerance) < 0.0:
            raise ValueError("overlap tolerance cannot be negative")

        if self.confirmation_traces_dir.is_dir():
            shutil.rmtree(self.confirmation_traces_dir)
        self.confirmation_traces_dir.mkdir(parents=True, exist_ok=True)

        short_trials = {
            (str(trial["candidate_id"]), int(trial["seed"])): trial
            for trial in self.report["trials"]
            if not trial.get("skipped")
        }
        protocol_reports = {}
        confirmation_trials = []
        figure_records = []
        for family, protocol in self.report["selected_protocols"].items():
            candidate = candidate_from_selected_protocol(protocol)
            candidate.validate(duration_ms)
            family_trials = []
            for seed in self.report["seeds"]:
                long_trial = self.run_trial(
                    candidate,
                    int(seed),
                    duration_ms,
                    trace_directory=self.confirmation_traces_dir,
                )
                short_trial = short_trials[(candidate.candidate_id, int(seed))]
                overlap = self._compare_trace_prefix(
                    self.output_dir / short_trial["trace_path"],
                    self.output_dir / long_trial["trace_path"],
                    float(overlap_tolerance),
                )
                required_kinds = set(candidate.required_event_kinds)
                required_events = [
                    event
                    for event in long_trial["events"]
                    if event["kind"] in required_kinds
                ]
                detected_kinds = {event["kind"] for event in required_events}
                censored_events = [
                    {
                        "kind": event["kind"],
                        "segment_id": event["segment_id"],
                        "onset_ms": event["onset_ms"],
                        "offset_ms": event["offset_ms"],
                    }
                    for event in required_events
                    if bool(event.get("right_censored", False))
                ]
                recovered = (
                    required_kinds.issubset(detected_kinds)
                    and not censored_events
                )
                confirmed = bool(
                    long_trial["success"]
                    and recovered
                    and overlap["valid"]
                )
                long_trial["confirmation"] = {
                    "confirmed": confirmed,
                    "required_events_recovered_below_reset": recovered,
                    "right_censored_required_events": censored_events,
                    "short_horizon_overlap": overlap,
                }
                confirmation_trials.append(long_trial)
                family_trials.append(long_trial)

            protocol_valid = all(
                trial["confirmation"]["confirmed"]
                for trial in family_trials
            )
            protocol_reports[family] = {
                "candidate_id": candidate.candidate_id,
                "valid": protocol_valid,
                "seed_count": len(family_trials),
                "confirmed_seed_count": sum(
                    trial["confirmation"]["confirmed"]
                    for trial in family_trials
                ),
                "maximum_short_horizon_overlap_error": max(
                    trial["confirmation"]["short_horizon_overlap"][
                        "maximum_absolute_error"
                    ]
                    for trial in family_trials
                ),
                "right_censored_required_event_count": sum(
                    len(
                        trial["confirmation"][
                            "right_censored_required_events"
                        ]
                    )
                    for trial in family_trials
                ),
                "trial_trace_paths": [
                    trial["trace_path"] for trial in family_trials
                ],
            }
            plot_trial = family_trials[0]
            relative = Path("figures") / f"confirmed_{family}.png"
            self._plot_trial(
                plot_trial,
                title=f"Long-horizon confirmation: {family}",
                output_path=self.output_dir / relative,
            )
            figure_records.append(
                {
                    "status": "long_horizon_confirmation",
                    "family": family,
                    "candidate_id": candidate.candidate_id,
                    "seed": plot_trial["seed"],
                    "path": relative.as_posix(),
                }
            )

        confirmation_report = {
            "schema_version": DENDRITIC_CALIBRATION_SCHEMA_VERSION,
            "valid": all(
                item["valid"] for item in protocol_reports.values()
            ),
            "teacher_commit": PINNED_TEACHER_COMMIT,
            "short_duration_ms": short_duration_ms,
            "confirmation_duration_ms": duration_ms,
            "overlap_tolerance": float(overlap_tolerance),
            "protocols": protocol_reports,
            "trials": confirmation_trials,
            "figures": figure_records,
        }
        write_json(
            self.output_dir / "confirmation_report.json",
            confirmation_report,
        )
        self.report["long_horizon_confirmation"] = confirmation_report
        self.report.setdefault("diagnostic_figures", []).extend(
            figure_records
        )
        write_json(self.output_dir / "calibration_report.json", self.report)
        write_json(
            self.output_dir / "selected_dendritic_protocols.json",
            {
                "schema_version": DENDRITIC_CALIBRATION_SCHEMA_VERSION,
                "teacher_commit": self.report["teacher_commit"],
                "canonical_synaptic_weights_unchanged": True,
                "selected_protocols": self.report["selected_protocols"],
                "event_coverage": self.report["event_coverage"],
                "long_horizon_confirmation": {
                    "valid": confirmation_report["valid"],
                    "report": "confirmation_report.json",
                    "duration_ms": duration_ms,
                },
            },
        )
        self._write_artifact_index()
        return confirmation_report

    def _upgrade_event_metadata_for_confirmation(self) -> None:
        """Upgrade an in-memory v0.2 sweep for an incremental confirmation."""

        self.session.event_definitions = [
            replace(
                definition,
                detector_version=EVENT_DETECTOR_VERSION,
            )
            for definition in self.session.event_definitions
        ]
        event_config_path = self.session.output_dir / "event_definition_config.json"
        if event_config_path.is_file():
            event_config = json.loads(
                event_config_path.read_text(encoding="utf-8")
            )
            event_config["event_detector_version"] = EVENT_DETECTOR_VERSION
            event_config["definitions"] = [
                definition.to_dict()
                for definition in self.session.event_definitions
            ]
            write_json(event_config_path, event_config)

        for trial in self.report.get("trials", ()):
            if trial.get("skipped") or not trial.get("trace_path"):
                continue
            trace_path = self.output_dir / str(trial["trace_path"])
            with self.np.load(trace_path) as data:
                trace_end_ms = float(data["time_ms"][-1])
            for event in trial.get("events", ()):
                event.setdefault(
                    "right_censored",
                    abs(float(event["offset_ms"]) - trace_end_ms) <= 1e-9,
                )
                event.setdefault(
                    "duration_is_lower_bound",
                    bool(event["right_censored"]),
                )
                event["detector_version"] = EVENT_DETECTOR_VERSION
                if isinstance(event.get("parameters"), dict):
                    event["parameters"][
                        "detector_version"
                    ] = EVENT_DETECTOR_VERSION
        self.report["schema_version"] = DENDRITIC_CALIBRATION_SCHEMA_VERSION

    def _reset_run_outputs(self) -> None:
        """Remove only prior calibration outputs, preserving teacher state."""

        for directory in (
            self.traces_dir,
            self.confirmation_traces_dir,
            self.figures_dir,
        ):
            if directory.is_dir():
                shutil.rmtree(directory)
            directory.mkdir(parents=True, exist_ok=True)
        for name in (
            "artifact_index.json",
            "calibration_report.json",
            "confirmation_report.json",
            "selected_dendritic_protocols.json",
        ):
            path = self.output_dir / name
            if path.is_file():
                path.unlink()

    @staticmethod
    def _rejected_trial_score(trial: Mapping[str, Any]) -> Tuple[Any, ...]:
        counts = trial.get("event_counts", {})
        required_hits = sum(
            counts.get(kind, 0) > 0
            for kind in trial.get("required_event_kinds", ())
        )
        forbidden_hits = sum(
            counts.get(kind, 0) > 0
            for kind in trial.get("forbidden_event_kinds", ())
        )
        return (
            int(required_hits),
            -int(forbidden_hits),
            float(trial.get("event_probe_peak_voltage_mv", -1e9)),
            -int(trial.get("canonical_synaptic_event_count", 0)),
            -int(trial.get("seed", 0)),
        )

    def _plot_trial(
        self,
        trial: Mapping[str, Any],
        *,
        title: str,
        output_path: Path,
    ) -> None:
        trace_path = self.output_dir / str(trial["trace_path"])
        with self.np.load(trace_path) as data:
            time = data["time_ms"]
            figure, axes = self.audit.plt.subplots(
                3, 1, figsize=(11, 9), sharex=True
            )
            for label in self.audit.representatives:
                axes[0].plot(
                    time, data[f"voltage_{label}_mv"], label=label
                )
            if trial.get("event_probe_mode") == "cluster_center":
                axes[0].plot(
                    time,
                    data["voltage_event_probe_mv"],
                    label=(
                        "event_probe "
                        f"(seg {trial['event_probe_segment_id']})"
                    ),
                    linewidth=2.2,
                    linestyle="--",
                )
            target = str(trial["target"])
            if trial.get("event_probe_mode") == "cluster_center":
                cai_key = "cai_event_probe_mM"
                ica_key = "ica_event_probe_mA_per_cm2"
                calcium_label = "event_probe"
            else:
                cai_key = f"cai_{target}_mM"
                ica_key = f"ica_{target}_mA_per_cm2"
                calcium_label = target
            baseline = float(data[cai_key][0])
            axes[1].plot(
                time,
                data[cai_key] - baseline,
                label=f"delta cai ({calcium_label})",
            )
            axes[1].plot(
                time,
                data[ica_key],
                label=f"ica ({calcium_label})",
            )
            axes[2].plot(time, data["sum_g_NMDA"], label="sum g_NMDA")
            axes[2].plot(time, data["sum_i_NMDA"], label="sum i_NMDA")
            required = set(trial.get("required_event_kinds", ()))
            for event in trial.get("events", ()):
                if event.get("kind") in required:
                    axes[0].axvline(
                        float(event["onset_ms"]),
                        color="black",
                        alpha=0.18,
                        linewidth=1.0,
                    )
            axes[0].set_ylabel("voltage (mV)")
            axes[1].set_ylabel("Ca observables")
            axes[2].set_ylabel("NMDA observables")
            axes[2].set_xlabel("absolute teacher time (ms)")
            axes[0].set_title(title)
            for axis in axes:
                axis.grid(alpha=0.2)
                axis.legend(ncol=4)
            figure.tight_layout()
            figure.savefig(output_path, dpi=160)
            self.audit.plt.close(figure)

    def _plot_diagnostics(
        self,
        selected: Mapping[str, Mapping[str, Any]],
        trials: Sequence[Mapping[str, Any]],
    ) -> List[Dict[str, Any]]:
        records: List[Dict[str, Any]] = []
        for family, protocol in selected.items():
            successful = next(
                trial
                for trial in trials
                if trial.get("candidate_id") == protocol["candidate_id"]
                and trial.get("success")
            )
            relative = Path("figures") / f"selected_{family}.png"
            self._plot_trial(
                successful,
                title=f"Selected dendritic protocol: {family}",
                output_path=self.output_dir / relative,
            )
            records.append(
                {
                    "status": "selected",
                    "family": family,
                    "candidate_id": successful["candidate_id"],
                    "seed": successful["seed"],
                    "path": relative.as_posix(),
                }
            )

        families = sorted(
            {
                str(trial["family"])
                for trial in trials
                if not trial.get("skipped")
            }
        )
        for family in families:
            selected_id = selected.get(family, {}).get("candidate_id")
            rejected = [
                trial
                for trial in trials
                if trial.get("family") == family
                and not trial.get("skipped")
                and (
                    trial.get("candidate_id") != selected_id
                    or not trial.get("success")
                )
            ]
            if not rejected:
                continue
            best = max(rejected, key=self._rejected_trial_score)
            relative = Path("figures") / f"best_rejected_{family}.png"
            self._plot_trial(
                best,
                title=f"Best rejected candidate: {family}",
                output_path=self.output_dir / relative,
            )
            records.append(
                {
                    "status": "best_rejected",
                    "family": family,
                    "candidate_id": best["candidate_id"],
                    "seed": best["seed"],
                    "required_event_hits": {
                        kind: int(best["event_counts"].get(kind, 0))
                        for kind in best["required_event_kinds"]
                    },
                    "forbidden_event_hits": {
                        kind: int(best["event_counts"].get(kind, 0))
                        for kind in best["forbidden_event_kinds"]
                    },
                    "path": relative.as_posix(),
                }
            )
        return records

    def _write_artifact_index(self) -> None:
        from .audit import sha256_file

        records = []
        for path in sorted(self.output_dir.rglob("*")):
            if path.is_file() and path.name != "artifact_index.json":
                records.append(
                    {
                        "path": path.relative_to(self.output_dir).as_posix(),
                        "size_bytes": path.stat().st_size,
                        "sha256": sha256_file(path),
                    }
                )
        write_json(
            self.output_dir / "artifact_index.json",
            {
                "schema_version": DENDRITIC_CALIBRATION_SCHEMA_VERSION,
                "artifacts": records,
            },
        )

    def package_artifacts(self, archive_base: Optional[Path] = None) -> Path:
        base = Path(
            archive_base
            or self.output_dir.parent / "dendritic_protocol_calibration"
        ).resolve()
        archive = shutil.make_archive(
            str(base),
            "zip",
            root_dir=self.output_dir.parent,
            base_dir=self.output_dir.name,
        )
        return Path(archive)
