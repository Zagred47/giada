"""Controlled dendritic-event protocol calibration for the canonical teacher.

The calibration changes only event schedules and the number of canonical
synapses recruited.  It never changes point-process parameters, NetCon
weights, membrane mechanisms, morphology, or solver tolerances.
"""

from __future__ import annotations

import json
import shutil
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from ..hayflow_data import InputAction, write_json
from .audit_runtime import PINNED_TEACHER_COMMIT
from .event_extractor import extract_events


DENDRITIC_CALIBRATION_SCHEMA_VERSION = "0.1.0"


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

    @property
    def candidate_id(self) -> str:
        pairing = "paired" if self.pair_with_somatic_spike else "unpaired"
        return (
            f"{self.family}-{self.target}-n{self.synapse_count}-"
            f"b{self.burst_count}-r{self.events_per_synapse_per_burst}-"
            f"{pairing}"
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
    )


def evenly_spaced_offsets(count: int) -> List[float]:
    """Return deterministic event offsets strictly inside one millisecond."""

    if int(count) <= 0:
        raise ValueError("offset count must be positive")
    return [0.1 + 0.8 * (index + 0.5) / int(count) for index in range(int(count))]


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
    offsets = evenly_spaced_offsets(event_total)
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
        self.figures_dir = self.output_dir / "figures"
        for directory in (self.output_dir, self.traces_dir, self.figures_dir):
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

    def select_local_excitatory_synapses(
        self,
        target: str,
        count: int,
        maximum_tree_distance_um: Optional[float] = None,
    ) -> List[int]:
        """Select canonical excitatory synapses nearest through the tree."""

        if target not in self.audit.representatives:
            raise KeyError(f"unknown representative target {target!r}")
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

        candidates = []
        for record in self.audit.synapse_records:
            if record["class_name"] != "ProbAMPANMDA2":
                continue
            segment_id = int(record["segment_id"])
            if not eligible(segment_id):
                continue
            tree_distance = self.tree_distance_um(
                segment_id, target_id, parents, distances
            )
            if (
                maximum_tree_distance_um is not None
                and tree_distance > float(maximum_tree_distance_um)
            ):
                continue
            candidates.append(
                (tree_distance, segment_id, int(record["synapse_id"]))
            )
        candidates.sort()
        if len(candidates) < int(count):
            raise InsufficientCanonicalSynapsesError(
                f"target {target!r} has only {len(candidates)} eligible "
                f"canonical excitatory synapses, fewer than requested {count}"
            )
        return [row[2] for row in candidates[: int(count)]]

    @staticmethod
    def _safe_read(owner: Any, name: str) -> float:
        return float(getattr(owner, name)) if hasattr(owner, name) else 0.0

    def _sample_observables(
        self, synapse_ids: Sequence[int]
    ) -> Dict[str, float]:
        values = {
            f"voltage_{label}_mv": float(
                self.audit.live_segments[int(segment_id)].v
            )
            for label, segment_id in self.audit.representatives.items()
        }
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
    ) -> Tuple[List[float], Dict[str, List[float]]]:
        self.session._disable_somatic_clamp()
        self.session._configure_somatic_current(start_time, actions)
        self.cvode.re_init()
        self.session._schedule_actions(start_time, actions)
        sample_count = int(round(1.0 / self.sample_interval_ms)) + 1
        times = start_time + self.np.linspace(0.0, 1.0, sample_count)
        rows: Dict[str, List[float]] = {}
        for sample_time in times:
            self.audit._advance_exact(float(sample_time))
            observed = self._sample_observables(synapse_ids)
            for name, value in observed.items():
                rows.setdefault(name, []).append(float(value))
        return times.tolist(), rows

    def run_trial(
        self,
        candidate: DendriticCandidate,
        seed: int,
        duration_ms: int,
    ) -> Dict[str, Any]:
        """Run one candidate from the canonical equilibrium snapshot."""

        self._require_ready()
        candidate.validate(duration_ms)
        synapse_ids = self.select_local_excitatory_synapses(
            candidate.target,
            candidate.synapse_count,
            candidate.maximum_tree_distance_um,
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
        events = extract_events(
            times, voltage_traces, self.session.event_definitions
        )
        counts = Counter(event["kind"] for event in events)
        success = all(
            counts.get(kind, 0) > 0 for kind in candidate.required_event_kinds
        ) and not any(
            counts.get(kind, 0) > 0 for kind in candidate.forbidden_event_kinds
        )
        target = candidate.target
        cai = self.np.asarray(traces[f"cai_{target}_mM"], dtype=float)
        nmda_current = self.np.asarray(traces["sum_i_NMDA"], dtype=float)
        parents = self._parent_map()
        soma_distances = self._distance_map()
        target_segment_id = int(self.audit.representatives[target])
        synapse_tree_distances = [
            self.tree_distance_um(
                int(self.audit.synapse_records[item]["segment_id"]),
                target_segment_id,
                parents,
                soma_distances,
            )
            for item in synapse_ids
        ]
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
            "canonical_synaptic_event_count": candidate.event_cost,
            "canonical_netcon_weights": canonical_weights,
            "canonical_weights_unchanged": True,
            "pair_with_somatic_spike": candidate.pair_with_somatic_spike,
            "somatic_current_na": somatic_current,
            "selected_synapse_ids": synapse_ids,
            "selected_segment_ids": [
                int(self.audit.synapse_records[item]["segment_id"])
                for item in synapse_ids
            ],
            "selected_synapse_tree_distances_um": synapse_tree_distances,
            "maximum_selected_tree_distance_um": float(
                max(synapse_tree_distances)
            ),
            "target_peak_voltage_mv": float(
                self.np.max(traces[f"voltage_{target}_mv"])
            ),
            "target_cai_increase_mM": float(self.np.max(cai) - cai[0]),
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
        trace_path = self.traces_dir / (
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
                "calibration-v0.1 selects the first robust configured level"
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
                        "burst_start_ms": candidate.burst_start_ms,
                        "burst_interval_ms": candidate.burst_interval_ms,
                        "maximum_tree_distance_um": (
                            candidate.maximum_tree_distance_um
                        ),
                        "pair_with_somatic_spike": (
                            candidate.pair_with_somatic_spike
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
        self.report = {
            "schema_version": DENDRITIC_CALIBRATION_SCHEMA_VERSION,
            "valid": not missing,
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
                "synchrony",
                "optional somatic-spike pairing",
            ],
            "selected_protocols": selected,
            "required_families": required_families,
            "missing_required_families": missing,
            "trial_count": len(
                [trial for trial in trials if not trial.get("skipped")]
            ),
            "skipped_candidate_count": len(
                [trial for trial in trials if trial.get("skipped")]
            ),
            "trials": trials,
        }
        write_json(self.output_dir / "calibration_report.json", self.report)
        write_json(
            self.output_dir / "selected_dendritic_protocols.json",
            {
                "schema_version": DENDRITIC_CALIBRATION_SCHEMA_VERSION,
                "teacher_commit": self.report["teacher_commit"],
                "canonical_synaptic_weights_unchanged": True,
                "selected_protocols": selected,
            },
        )
        self._plot_selected(selected, trials)
        self._write_artifact_index()
        return self.report

    def _reset_run_outputs(self) -> None:
        """Remove only prior calibration outputs, preserving teacher state."""

        for directory in (self.traces_dir, self.figures_dir):
            if directory.is_dir():
                shutil.rmtree(directory)
            directory.mkdir(parents=True, exist_ok=True)
        for name in (
            "artifact_index.json",
            "calibration_report.json",
            "selected_dendritic_protocols.json",
        ):
            path = self.output_dir / name
            if path.is_file():
                path.unlink()

    def _plot_selected(
        self,
        selected: Mapping[str, Mapping[str, Any]],
        trials: Sequence[Mapping[str, Any]],
    ) -> None:
        for family, protocol in selected.items():
            successful = next(
                trial
                for trial in trials
                if trial.get("candidate_id") == protocol["candidate_id"]
                and trial.get("success")
            )
            trace_path = self.output_dir / successful["trace_path"]
            with self.np.load(trace_path) as data:
                time = data["time_ms"]
                figure, axes = self.audit.plt.subplots(
                    3, 1, figsize=(11, 9), sharex=True
                )
                for label in self.audit.representatives:
                    axes[0].plot(
                        time, data[f"voltage_{label}_mv"], label=label
                    )
                target = str(protocol["target"])
                baseline = float(data[f"cai_{target}_mM"][0])
                axes[1].plot(
                    time,
                    data[f"cai_{target}_mM"] - baseline,
                    label=f"delta cai ({target})",
                )
                axes[1].plot(
                    time,
                    data[f"ica_{target}_mA_per_cm2"],
                    label=f"ica ({target})",
                )
                axes[2].plot(time, data["sum_g_NMDA"], label="sum g_NMDA")
                axes[2].plot(time, data["sum_i_NMDA"], label="sum i_NMDA")
                axes[0].set_ylabel("voltage (mV)")
                axes[1].set_ylabel("Ca observables")
                axes[2].set_ylabel("NMDA observables")
                axes[2].set_xlabel("absolute teacher time (ms)")
                axes[0].set_title(
                    f"Selected dendritic protocol: {family}"
                )
                for axis in axes:
                    axis.grid(alpha=0.2)
                    axis.legend(ncol=4)
                figure.tight_layout()
                figure.savefig(
                    self.figures_dir / f"selected_{family}.png", dpi=160
                )
                self.audit.plt.close(figure)

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
