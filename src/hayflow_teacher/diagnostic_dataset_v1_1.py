"""Targeted HayFlow diagnostic transition dataset, schema 1.1.0."""

from __future__ import annotations

import json
import time
import hashlib
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from ..hayflow_data import (
    TARGETED_DATASET_SCHEMA_VERSION,
    CausalReleaseOutcome,
    InputAction,
    ProtocolTrajectory,
    build_input_views,
    write_json,
    summarize_independent_support,
    validate_hdf5_store,
    validate_minimum_support,
    TargetedRecipe,
    action_schedule_from_json,
    select_adaptive_recipe_brackets,
)
from .audit import sha256_file
from .audit_runtime import PINNED_TEACHER_COMMIT
from .causal_release import CausalReleaseRecorder
from .diagnostic_dataset import (
    DEFAULT_MICROTRACE_STEP_MS,
    DiagnosticDatasetSession,
    _ConsoleProgress,
)
from .diagnostic_dataset_v1 import (
    DiagnosticDatasetV1Session,
    canonical_json_sha256,
)
from .event_extractor import annotate_backpropagation, extract_events
from .dendritic_calibration import DendriticCandidate, DendriticProtocolCalibrator


class TargetedDiagnosticDatasetSession(DiagnosticDatasetV1Session):
    """Version 1.1 session with causal release logging and pilot gates."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        if kwargs.get("output_dir") is None:
            elm_repo = Path(args[0] if args else kwargs["elm_repo"])
            kwargs["output_dir"] = (
                elm_repo / "artifacts" / "diagnostic_dataset_v1_1"
            )
        super().__init__(*args, **kwargs)
        self.active_random123_seed = int(self.seed)
        self._active_transition_id = -1
        self._last_release_outcomes: List[CausalReleaseOutcome] = []
        self._collect_release_rows = False
        self.release_rows: List[Dict[str, Any]] = []
        self.causal_release_pilot_report: Dict[str, Any] = {}
        self.targeted_preflight_report: Dict[str, Any] = {}
        self.targeted_recipe_catalog: List[TargetedRecipe] = []
        self.snapshot_bank: Dict[str, Dict[str, Any]] = {}

    @staticmethod
    def _protocol_plan_sha256(
        protocols: Sequence[ProtocolTrajectory],
    ) -> str:
        """Bind actions, labels, split, branch, snapshot and recovery metadata."""

        payload = {
            "schema_version": TARGETED_DATASET_SCHEMA_VERSION,
            "trajectories": [
                {
                    "trajectory_id": row.trajectory_id,
                    "category": row.category,
                    "protocol": row.protocol,
                    "protocol_id": row.protocol_id,
                    "protocol_variant": row.protocol_variant,
                    "seed": int(row.seed),
                    "duration_ms": int(row.duration_ms),
                    "split": row.split,
                    "stimulus_onset_step": int(row.stimulus_onset_step),
                    "required_event_kinds": list(row.required_event_kinds),
                    "negative_control": bool(row.negative_control),
                    "snapshot_source": row.snapshot_source,
                    "metadata": dict(row.metadata),
                    "actions": {
                        str(step): [action.to_dict() for action in actions]
                        for step, actions in sorted(row.actions_by_step.items())
                    },
                }
                for row in sorted(protocols, key=lambda item: item.trajectory_id)
            ],
        }
        return canonical_json_sha256(payload)

    def prepare_targeted_contract(self) -> Dict[str, Any]:
        """Prepare v1.0.1 first, then add only versioned v1.1 metadata."""

        base = self.prepare_v1_contract()
        self.event_definitions = [
            replace(
                row,
                maximum_event_duration_ms=9.975,
            )
            if row.kind == "nmda_spike"
            else row
            for row in self.event_definitions
        ]
        base_layout = {
            "index_contract": self.state_schema["index_contract"],
            "categories": self.state_schema["categories"],
            "rng_state": self.state_schema["rng_state"],
            "variables": self.state_schema["variables"],
        }
        base_layout_sha256 = canonical_json_sha256(base_layout)
        self.state_schema.update(
            {
                "schema_version": TARGETED_DATASET_SCHEMA_VERSION,
                "compatible_base_schema_version": "1.0.1",
                "canonical_state_layout_sha256": base_layout_sha256,
                "v1_1_extension_groups": (
                    "release_outcome,event_labels,input_views"
                ),
            }
        )
        write_json(self.output_dir / "state_schema.json", self.state_schema)
        release_schema = {
            "schema_version": "1.0.0",
            "dataset_schema_version": TARGETED_DATASET_SCHEMA_VERSION,
            "teacher_commit": PINNED_TEACHER_COMMIT,
            "teacher_mechanisms_unchanged": True,
            "mechanisms": {
                "ProbAMPANMDA2": {
                    "random_distribution": "negexp(1)",
                    "release_rule": "erand() < Pr",
                    "direct_state_increments": [
                        "A_AMPA",
                        "B_AMPA",
                        "A_NMDA",
                        "B_NMDA",
                    ],
                },
                "ProbUDFsyn2": {
                    "random_distribution": "negexp(1)",
                    "release_rule": "erand() < Pr",
                    "direct_state_increments": ["A", "B"],
                },
            },
            "causal_boundary": {
                "pre": "callback at event timestamp before original NetCon delivery",
                "decision": "unchanged original NET_RECEIVE",
                "post": (
                    "callback at the same timestamp after NET_RECEIVE and before "
                    "membrane advancement under the new conductance"
                ),
                "forbidden_source": "S_(t+1) or any future membrane state",
            },
            "random_preview": {
                "purpose": "validate the direct same-time state discontinuity",
                "method": (
                    "independent Random123 with identical seed, stream and seq; "
                    "the teacher RNG is never advanced by instrumentation"
                ),
            },
            "input_views": {
                "U_scheduled": "ordered presynaptic schedule only",
                "U_rng": "schedule plus causal Random123 identity and position",
                "U_realized": (
                    "successful releases and direct component increments emitted "
                    "by the authentic synaptic front-end"
                ),
            },
            "deployment_boundary": (
                "HayFlow retains the authentic synaptic front-end. U_realized is "
                "therefore available before the learned membrane core update and "
                "is not target leakage."
            ),
            "fields": list(CausalReleaseOutcome.__dataclass_fields__),
        }
        write_json(self.output_dir / "release_schema.json", release_schema)
        write_json(
            self.output_dir / "event_definition_config.json",
            {
                "schema_version": TARGETED_DATASET_SCHEMA_VERSION,
                "status": "targeted_configurable_thresholds",
                "definitions": [row.to_dict() for row in self.event_definitions],
                "nmda_regime_policy": (
                    "nmda_spike duration <= 9.975 ms; nmda_plateau duration >= 10 ms"
                ),
                "bap_policy": (
                    "somatic origin plus ordered soma-to-trunk regional propagation"
                ),
                "right_censoring_policy": (
                    "retain censored labels; exclude them from duration/offset targets"
                ),
            },
        )
        return {
            **base,
            "schema_version": TARGETED_DATASET_SCHEMA_VERSION,
            "canonical_state_layout_sha256": base_layout_sha256,
            "release_schema": "release_schema.json",
        }

    def _extract_events(
        self,
        time_ms: Sequence[float],
        traces: Mapping[str, Sequence[float]],
    ) -> List[Dict[str, Any]]:
        events = extract_events(time_ms, traces, self.event_definitions)
        distances = {}
        for label in ("soma", "basal", "trunk", "nexus", "tuft"):
            segment_id = self.audit.representatives.get(label)
            if segment_id is None:
                continue
            row = self.audit.segment_df[
                self.audit.segment_df["segment_id"] == int(segment_id)
            ]
            distances[label] = float(row.iloc[0]["distance_from_soma_um"])
        return annotate_backpropagation(
            time_ms,
            traces,
            events,
            regional_distances_um=distances,
        )

    def _configure_rngs(
        self, seed: int, sequences: Sequence[float]
    ) -> None:
        super()._configure_rngs(seed, sequences)
        self.active_random123_seed = int(seed)

    def _trajectory_initial_state(
        self, trajectory: ProtocolTrajectory
    ) -> Tuple[Path, Sequence[float], int]:
        snapshot_id = str(
            trajectory.metadata.get("snapshot_id", trajectory.snapshot_source)
        )
        if snapshot_id not in self.snapshot_bank:
            return super()._trajectory_initial_state(trajectory)
        record = self.snapshot_bank[snapshot_id]
        return (
            self.output_dir / str(record["native_snapshot"]),
            list(record["rng_sequences"]),
            int(record["random123_seed"]),
        )

    def prepare_snapshot_bank(
        self,
        protocols: Sequence[ProtocolTrajectory],
        *,
        conditioning_ms: int = 4,
    ) -> Dict[str, Any]:
        """Create real split-specific subthreshold teacher snapshots."""

        self._require_equilibrium()
        if int(conditioning_ms) <= 0:
            raise ValueError("snapshot conditioning duration must be positive")
        snapshot_splits: Dict[str, set] = {}
        for trajectory in protocols:
            snapshot_id = str(
                trajectory.metadata.get(
                    "snapshot_id", trajectory.snapshot_source
                )
            )
            snapshot_splits.setdefault(snapshot_id, set()).add(trajectory.split)
        leaking = {
            key: sorted(values)
            for key, values in snapshot_splits.items()
            if len(values) > 1
        }
        if leaking:
            raise ValueError(f"snapshot ids leak across splits: {leaking}")
        snapshot_ids = sorted(snapshot_splits)
        equilibrium_rng = json.loads(
            self.equilibrium_rng_path.read_text(encoding="utf-8")
        )
        excitatory = [
            int(row["synapse_id"])
            for row in self.audit.synapse_records
            if row["class_name"] == "ProbAMPANMDA2"
        ]
        inhibitory = [
            int(row["synapse_id"])
            for row in self.audit.synapse_records
            if row["class_name"] == "ProbUDFsyn2"
        ]
        destination = self.snapshots_dir / "initial_states"
        destination.mkdir(parents=True, exist_ok=True)
        progress = _ConsoleProgress("snapshot bank", len(snapshot_ids))
        bank = {}
        for index, snapshot_id in enumerate(snapshot_ids, start=1):
            digest = hashlib.sha256(snapshot_id.encode("utf-8")).digest()
            bank_seed = 800_000 + int.from_bytes(digest[:4], "big") % 100_000
            ex_id = excitatory[int.from_bytes(digest[4:8], "big") % len(excitatory)]
            inh_id = inhibitory[int.from_bytes(digest[8:12], "big") % len(inhibitory)]
            self._restore_native_snapshot(
                self.equilibrium_snapshot_path,
                equilibrium_rng["sequences"],
                equilibrium_rng.get("random123_seed", self.seed),
            )
            self._rekey_rngs(bank_seed)
            dummy = ProtocolTrajectory(
                trajectory_id=f"snapshot-conditioning-{snapshot_id}",
                category="rest_subthreshold",
                protocol="snapshot_conditioning",
                seed=bank_seed,
                duration_ms=int(conditioning_ms),
                split=next(iter(snapshot_splits[snapshot_id])),
                stimulus_onset_step=0,
            )
            peak_soma = -1.0e300
            for step in range(conditioning_ms):
                actions = (
                    InputAction(
                        "synaptic_event", 0.25, synapse_id=ex_id
                    ),
                    InputAction(
                        "synaptic_event", 0.65, synapse_id=inh_id
                    ),
                )
                row = self._run_transition(
                    -1, dummy, step, actions, snapshot_path=None
                )
                soma_index = list(self.audit.representatives).index("soma")
                peak_soma = max(
                    peak_soma,
                    float(self.np.max(row["micro_probe_voltage"][:, soma_index])),
                )
            if peak_soma >= -20.0:
                raise RuntimeError(
                    f"snapshot conditioning became suprathreshold for {snapshot_id}"
                )
            safe_name = hashlib.sha256(snapshot_id.encode("utf-8")).hexdigest()[:20]
            native_path = destination / f"{safe_name}.neuron.bin"
            self._write_native_snapshot(native_path)
            sequences = self.audit._snapshot_rng_sequences()
            bank[snapshot_id] = {
                "snapshot_id": snapshot_id,
                "splits": sorted(snapshot_splits[snapshot_id]),
                "native_snapshot": native_path.relative_to(self.output_dir).as_posix(),
                "native_snapshot_sha256": sha256_file(native_path),
                "random123_seed": bank_seed,
                "rng_sequences": list(map(float, sequences)),
                "conditioning_ms": int(conditioning_ms),
                "conditioning_synapse_ids": [ex_id, inh_id],
                "peak_soma_mv": peak_soma,
                "suprathreshold": False,
            }
            progress.update(
                index,
                detail=f"{snapshot_id}; peak soma={peak_soma:.2f} mV",
            )
        self.snapshot_bank = bank
        report = {
            "schema_version": TARGETED_DATASET_SCHEMA_VERSION,
            "valid": len(bank) == len(snapshot_ids) and not leaking,
            "snapshot_count": len(bank),
            "conditioning_ms": int(conditioning_ms),
            "split_specific": True,
            "shared_only_within_split": True,
            "snapshots": bank,
        }
        write_json(self.output_dir / "snapshot_bank.json", report)
        return report

    def _make_transition_writer(self, *args: Any, **kwargs: Any) -> Any:
        kwargs.update(
            {
                "input_view_names": ("U_scheduled", "U_rng", "U_realized"),
                "store_release_outcomes": True,
            }
        )
        return super()._make_transition_writer(*args, **kwargs)

    def _drive_one_ms(
        self,
        start_time: float,
        actions: Sequence[InputAction],
        observer: Any,
        *,
        sample_interval_ms: float = DEFAULT_MICROTRACE_STEP_MS,
    ) -> Tuple[Any, List[Dict[str, Any]], List[Any]]:
        """Canonical driver plus non-invasive same-time release callbacks."""

        interval = float(sample_interval_ms)
        sample_count = int(round(1.0 / interval)) + 1
        if interval <= 0.0 or abs((sample_count - 1) * interval - 1.0) > 1e-9:
            raise ValueError("sample interval must divide one millisecond")
        self._disable_somatic_clamp()
        self._configure_somatic_current(start_time, actions)
        self.cvode.re_init()
        recorder = CausalReleaseRecorder(
            self,
            transition_id=self._active_transition_id,
            random123_seed=self.active_random123_seed,
        )
        public_actions = recorder.schedule(start_time, actions)
        times = self.np.linspace(
            float(start_time), float(start_time) + 1.0, sample_count
        )
        samples = []
        for sample_time in times:
            self.audit._advance_exact(float(sample_time))
            samples.append(observer())
        self._last_release_outcomes = recorder.outcomes()
        return self.np.asarray(times, dtype=float), public_actions, samples

    def _run_transition(
        self,
        transition_id: int,
        trajectory: ProtocolTrajectory,
        step_index: int,
        actions: Sequence[InputAction],
        snapshot_path: Optional[Path],
    ) -> Dict[str, Any]:
        self._active_transition_id = int(transition_id)
        self._last_release_outcomes = []
        row = super()._run_transition(
            transition_id,
            trajectory,
            step_index,
            actions,
            snapshot_path,
        )
        views = build_input_views(row["inputs"], self._last_release_outcomes)
        releases = [outcome.to_dict() for outcome in self._last_release_outcomes]
        row["inputs"] = views["U_scheduled"]
        row["input_views"] = views
        row["release_outcomes"] = releases
        if self._collect_release_rows and int(transition_id) >= 0:
            for release in releases:
                self.release_rows.append(
                    {
                        "trajectory_id": trajectory.trajectory_id,
                        "protocol_id": trajectory.protocol_id,
                        "split": trajectory.split,
                        "seed": int(trajectory.seed),
                        "step_index": int(step_index),
                        **release,
                    }
                )
        return row

    @staticmethod
    def _outcome_signature(rows: Sequence[Mapping[str, Any]]) -> str:
        payload = []
        for row in rows:
            payload.append(
                {
                    key: value
                    for key, value in row.items()
                    if key not in {"transition_id"}
                }
            )
        return canonical_json_sha256({"outcomes": payload})

    def run_causal_release_pilot(
        self, *, transition_count: int = 8, seed: int = 410001
    ) -> Dict[str, Any]:
        """Prove causal logging and exact replay before dataset generation."""

        self._require_equilibrium()
        if int(transition_count) < 2:
            raise ValueError("causal release pilot needs at least two transitions")
        excitatory = [
            int(row["synapse_id"])
            for row in self.audit.synapse_records
            if row["class_name"] == "ProbAMPANMDA2"
        ][:4]
        inhibitory = [
            int(row["synapse_id"])
            for row in self.audit.synapse_records
            if row["class_name"] == "ProbUDFsyn2"
        ][:4]
        if len(excitatory) != 4 or len(inhibitory) != 4:
            raise RuntimeError("canonical release pilot synapse pool is incomplete")
        offsets = self.np.linspace(0.1, 0.8, 8)
        actions = tuple(
            InputAction("synaptic_event", float(offset), synapse_id=synapse_id)
            for offset, synapse_id in zip(offsets, excitatory + inhibitory)
        )
        trajectory = ProtocolTrajectory(
            trajectory_id=f"causal-release-pilot-seed{seed}",
            category="local_synaptic",
            protocol="causal_release_pilot",
            protocol_id="causal_release_pilot",
            protocol_variant="excitatory_and_inhibitory_repeated",
            seed=int(seed),
            duration_ms=int(transition_count),
            split="release_identifiability_test",
            actions_by_step={step: actions for step in range(transition_count)},
            stimulus_onset_step=0,
        )
        equilibrium_rng = json.loads(
            self.equilibrium_rng_path.read_text(encoding="utf-8")
        )

        def run_once(run_seed: int) -> Tuple[List[Dict[str, Any]], Dict[str, Any], float]:
            self._restore_native_snapshot(
                self.equilibrium_snapshot_path,
                equilibrium_rng["sequences"],
                equilibrium_rng.get("random123_seed", self.seed),
            )
            self._rekey_rngs(run_seed)
            rows = []
            started = time.perf_counter()
            for step_index in range(transition_count):
                rows.append(
                    self._run_transition(
                        -1,
                        trajectory,
                        step_index,
                        actions,
                        snapshot_path=None,
                    )
                )
            elapsed = time.perf_counter() - started
            final = {
                "state": self.capture_boundary_state(),
                "rng": self.np.asarray(
                    self.audit._snapshot_rng_sequences(), dtype=float
                ),
            }
            return rows, final, elapsed

        first, first_final, elapsed = run_once(int(seed))
        second, second_final, _ = run_once(int(seed))
        different_seed_rows, _, _ = run_once(int(seed) + 1)

        self._restore_native_snapshot(
            self.equilibrium_snapshot_path,
            equilibrium_rng["sequences"],
            equilibrium_rng.get("random123_seed", self.seed),
        )
        self._rekey_rngs(int(seed))
        for _ in range(transition_count):
            DiagnosticDatasetSession._drive_one_ms(
                self,
                float(self.h.t),
                actions,
                self._sample_transition_point,
                sample_interval_ms=DEFAULT_MICROTRACE_STEP_MS,
            )
        uninstrumented_final = {
            "state": self.capture_boundary_state(),
            "rng": self.np.asarray(
                self.audit._snapshot_rng_sequences(), dtype=float
            ),
        }
        first_releases = [
            release for row in first for release in row["release_outcomes"]
        ]
        second_releases = [
            release for row in second for release in row["release_outcomes"]
        ]
        different_releases = [
            release
            for row in different_seed_rows
            for release in row["release_outcomes"]
        ]
        state_error = max(
            float(
                self.np.max(
                    self.np.abs(
                        first_final["state"][category]
                        - second_final["state"][category]
                    )
                )
            )
            for category in self.state_variables
        )
        rng_error = float(
            self.np.max(self.np.abs(first_final["rng"] - second_final["rng"]))
        )
        uninstrumented_state_error = max(
            float(
                self.np.max(
                    self.np.abs(
                        first_final["state"][category]
                        - uninstrumented_final["state"][category]
                    )
                )
            )
            for category in self.state_variables
        )
        uninstrumented_rng_error = float(
            self.np.max(
                self.np.abs(
                    first_final["rng"] - uninstrumented_final["rng"]
                )
            )
        )
        same_release = self._outcome_signature(first_releases) == self._outcome_signature(
            second_releases
        )
        different_rng = any(
            abs(float(left["rng_preview_value"]) - float(right["rng_preview_value"]))
            > 0.0
            for left, right in zip(first_releases, different_releases)
        )

        pilot_dir = self.output_dir / "pilot"
        pilot_dir.mkdir(parents=True, exist_ok=True)
        pilot_path = pilot_dir / "causal_release_pilot.h5"
        widths = {
            category: len(variables)
            for category, variables in self.state_variables.items()
        }
        widths["rng_state"] = len(self.audit.synapse_rngs)
        with self._make_transition_writer(
            pilot_path,
            widths,
            41,
            len(self.micro_variables),
            len(self.audit.live_segments),
            len(self.audit.representatives),
            micro_observable_names=self.micro_observable_ids,
        ) as writer:
            writer.set_microtrace_grid(self.np.linspace(0.0, 1.0, 41))
            for index, row in enumerate(first):
                row["metadata"]["transition_id"] = index
                writer.append(row)
        release_counts = Counter(
            (row["synapse_type"], bool(row["release_success"]))
            for row in first_releases
        )
        byte_per_transition = int(pilot_path.stat().st_size / len(first))
        valid = bool(
            first_releases
            and same_release
            and state_error == 0.0
            and rng_error == 0.0
            and uninstrumented_state_error == 0.0
            and uninstrumented_rng_error == 0.0
            and different_rng
            and any(row["release_success"] for row in first_releases)
            and any(not row["release_success"] for row in first_releases)
        )
        report = {
            "schema_version": TARGETED_DATASET_SCHEMA_VERSION,
            "valid": valid,
            "teacher_commit": PINNED_TEACHER_COMMIT,
            "transition_count": int(transition_count),
            "scheduled_release_count": len(first_releases),
            "release_counts": {
                f"{kind}:{'success' if success else 'failure'}": count
                for (kind, success), count in sorted(release_counts.items())
            },
            "same_snapshot_rng_schedule_same_release_outcome": same_release,
            "same_snapshot_rng_schedule_same_transition": state_error == 0.0,
            "maximum_final_state_error": state_error,
            "maximum_final_rng_error": rng_error,
            "instrumentation_vs_canonical_driver_maximum_state_error": (
                uninstrumented_state_error
            ),
            "instrumentation_vs_canonical_driver_maximum_rng_error": (
                uninstrumented_rng_error
            ),
            "instrumentation_changes_teacher_dynamics": bool(
                uninstrumented_state_error != 0.0
                or uninstrumented_rng_error != 0.0
            ),
            "different_seed_changes_causal_rng_stream": different_rng,
            "teacher_rng_advanced_only_by_original_net_receive": True,
            "release_source": (
                "direct same-timestamp pre/post NET_RECEIVE state discontinuity"
            ),
            "future_state_used": False,
            "elapsed_seconds": elapsed,
            "seconds_per_transition": elapsed / len(first),
            "observed_hdf5_bytes_per_transition": byte_per_transition,
            "projected": {
                str(count): {
                    "generation_hours": elapsed / len(first) * count / 3600.0,
                    "hdf5_gib": byte_per_transition * count / (1024.0**3),
                }
                for count in (10_000, 30_000)
            },
            "pilot_store": str(pilot_path.relative_to(self.output_dir)),
            "pilot_store_sha256": sha256_file(pilot_path),
        }
        self.causal_release_pilot_report = report
        write_json(self.output_dir / "causal_release_pilot.json", report)
        if not valid:
            raise RuntimeError(
                "causal release pilot failed; do not generate dataset v1.1: "
                f"{report}"
            )
        return report

    def run_targeted_biological_pilot(
        self, config: Mapping[str, Any]
    ) -> Dict[str, Any]:
        """Coarse-to-boundary pilot for separable event-regime recipes."""

        if not self.causal_release_pilot_report.get("valid"):
            raise RuntimeError("causal release pilot must pass first")
        seeds = list(map(int, config["seeds"]))
        if len(seeds) < 2:
            raise ValueError("biological pilot requires at least two seeds")
        duration_ms = int(config.get("dendritic_duration_ms", 80))
        if duration_ms < 40:
            raise ValueError("dendritic pilot duration is too short")
        calibrator = DendriticProtocolCalibrator(
            self,
            output_dir=self.output_dir / "targeted_pilot",
            sample_interval_ms=0.025,
        )
        trials: List[Dict[str, Any]] = []
        nmda_cfg = config["nmda_sweep"]
        calcium_cfg = config["calcium_sweep"]
        base_trial_total = len(seeds) * (
            len(nmda_cfg["synapse_counts"])
            * len(nmda_cfg["burst_counts"])
            * len(nmda_cfg["event_windows_ms"])
            + len(calcium_cfg["synapse_counts"])
            * len(calcium_cfg["paired"])
            * len(calcium_cfg["event_windows_ms"])
            + len(config["somatic_current_factors"])
        )
        pilot_progress = _ConsoleProgress("pilot biologico", base_trial_total)

        def run_candidate(candidate: DendriticCandidate) -> None:
            for seed in seeds:
                trial = calibrator.run_trial(candidate, seed, duration_ms)
                trial["event_kinds"] = sorted(
                    {str(row["kind"]) for row in trial["events"]}
                )
                trial["stimulus_scalar"] = float(
                    candidate.event_cost / candidate.event_window_ms
                    + (candidate.synapse_count if candidate.pair_with_somatic_spike else 0)
                )
                trial["duration_ms"] = duration_ms
                trial["branch_id"] = f"segment-{trial['event_probe_segment_id']}"
                trials.append(trial)
                pilot_progress.update(
                    len(trials),
                    detail=(
                        f"{candidate.candidate_id}; seed={seed}; "
                        f"eventi={','.join(trial['event_kinds']) or 'nessuno'}"
                    ),
                )

        nmda = config["nmda_sweep"]
        for count in map(int, nmda["synapse_counts"]):
            for bursts in map(int, nmda["burst_counts"]):
                for window in map(float, nmda["event_windows_ms"]):
                    run_candidate(
                        DendriticCandidate(
                            family="targeted_nmda",
                            target="tuft",
                            required_event_kinds=("nmda_spike",),
                            forbidden_event_kinds=(),
                            synapse_count=count,
                            burst_count=bursts,
                            burst_start_ms=int(nmda.get("burst_start_ms", 3)),
                            burst_interval_ms=int(nmda.get("burst_interval_ms", 1)),
                            pair_with_somatic_spike=False,
                            selection_mode="branch_cluster",
                            event_probe_mode="cluster_center",
                            event_probe_kinds=("nmda_spike", "nmda_plateau"),
                            event_window_ms=window,
                        )
                    )

        calcium = config["calcium_sweep"]
        for count in map(int, calcium["synapse_counts"]):
            for paired in map(bool, calcium["paired"]):
                for window in map(float, calcium["event_windows_ms"]):
                    run_candidate(
                        DendriticCandidate(
                            family="targeted_calcium",
                            target="hot_zone",
                            required_event_kinds=("calcium_spike",),
                            forbidden_event_kinds=(),
                            synapse_count=count,
                            burst_count=int(calcium.get("burst_count", 3)),
                            burst_start_ms=int(calcium.get("burst_start_ms", 3)),
                            burst_interval_ms=int(calcium.get("burst_interval_ms", 1)),
                            pair_with_somatic_spike=paired,
                            selection_mode="target_nearest",
                            event_probe_mode="target_representative",
                            event_window_ms=window,
                        )
                    )

        if self.calibrated_somatic_single_spike_current_na is None:
            self.calibrate_somatic_single_spike_current()
        base_current = float(self.calibrated_somatic_single_spike_current_na)
        somatic_duration = int(config.get("somatic_duration_ms", 20))
        for factor in map(float, config["somatic_current_factors"]):
            amplitude = base_current * factor
            candidate_id = f"targeted_somatic-current-factor{factor:.4f}"
            schedule = {
                3: (
                    InputAction(
                        "somatic_current",
                        0.05,
                        duration_ms=0.9,
                        amplitude_na=amplitude,
                    ),
                )
            }
            for seed in seeds:
                trajectory = ProtocolTrajectory(
                    trajectory_id=f"pilot-{candidate_id}-seed{seed}",
                    category="somatic_events",
                    protocol="targeted_somatic_bap",
                    protocol_id=candidate_id,
                    protocol_variant="current_sweep",
                    seed=seed,
                    duration_ms=somatic_duration,
                    split="event_boundary_test",
                    actions_by_step=schedule,
                    stimulus_onset_step=3,
                )
                times, traces = self._run_trajectory_prefix_in_memory(
                    trajectory, somatic_duration
                )
                events = self._extract_events(times, traces)
                trials.append(
                    {
                        "candidate_id": candidate_id,
                        "family": "targeted_somatic_bap",
                        "seed": seed,
                        "event_kinds": sorted(
                            {str(row["kind"]) for row in events}
                        ),
                        "events": events,
                        "stimulus_scalar": amplitude,
                        "duration_ms": somatic_duration,
                        "branch_id": "soma-to-apical-and-basal",
                        "input_schedule": {
                            str(step): [action.to_dict() for action in actions]
                            for step, actions in schedule.items()
                        },
                    }
                )
                pilot_progress.update(
                    len(trials),
                    detail=(
                        f"{candidate_id}; seed={seed}; "
                        f"eventi={','.join(trials[-1]['event_kinds']) or 'nessuno'}"
                    ),
                )

        pilot_progress.update(
            min(len(trials), base_trial_total),
            detail="sweep primario completato",
            force=True,
        )

        # A second tuft branch is pilot-tested but explicitly excluded from
        # train, so the held-out-branch split is a real morphology holdout.
        primary_nmda = [
            row
            for row in trials
            if row.get("family") == "targeted_nmda"
            and int(row.get("synapse_count", 0))
            <= len(self.alternate_tuft_selection["synapse_ids"])
        ]
        if primary_nmda:
            source_id = max(
                primary_nmda, key=lambda row: float(row["stimulus_scalar"])
            )["candidate_id"]
            alternate_ids = list(
                map(int, self.alternate_tuft_selection["synapse_ids"])
            )
            for source in [
                row for row in primary_nmda if row["candidate_id"] == source_id
            ]:
                old_ids = list(map(int, source["selected_synapse_ids"]))
                replacement = {
                    old: new
                    for old, new in zip(old_ids, alternate_ids[: len(old_ids)])
                }
                schedule = action_schedule_from_json(source["input_schedule"])
                replaced = {
                    step: tuple(
                        InputAction(
                            action.kind,
                            action.offset_ms,
                            synapse_id=(
                                replacement[int(action.synapse_id)]
                                if action.kind == "synaptic_event"
                                else None
                            ),
                            weight_multiplier=action.weight_multiplier,
                            duration_ms=action.duration_ms,
                            amplitude_na=action.amplitude_na,
                            metadata=action.metadata,
                        )
                        for action in actions
                    )
                    for step, actions in schedule.items()
                }
                candidate_id = f"{source_id}-held-out-tuft-branch"
                trajectory = ProtocolTrajectory(
                    trajectory_id=f"pilot-{candidate_id}-seed{source['seed']}",
                    category="dendritic_events",
                    protocol="targeted_nmda_heldout_branch",
                    protocol_id=candidate_id,
                    protocol_variant="held_out_morphological_branch",
                    seed=int(source["seed"]),
                    duration_ms=duration_ms,
                    split="held_out_branch_test",
                    actions_by_step=replaced,
                    stimulus_onset_step=min(replaced),
                    metadata={
                        "event_probe_segment_id": int(
                            self.alternate_tuft_selection["center_segment_id"]
                        )
                    },
                )
                times, traces = self._run_trajectory_prefix_in_memory(
                    trajectory, duration_ms
                )
                events = self._extract_events(times, traces)
                trials.append(
                    {
                        "candidate_id": candidate_id,
                        "family": "targeted_nmda_heldout_branch",
                        "seed": int(source["seed"]),
                        "event_kinds": sorted(
                            {str(row["kind"]) for row in events}
                        ),
                        "events": events,
                        "stimulus_scalar": float(source["stimulus_scalar"]),
                        "duration_ms": duration_ms,
                        "branch_id": (
                            "segment-"
                            f"{self.alternate_tuft_selection['center_segment_id']}"
                        ),
                        "input_schedule": {
                            str(step): [action.to_dict() for action in actions]
                            for step, actions in replaced.items()
                        },
                        "train_eligible": False,
                    }
                )

        brackets = select_adaptive_recipe_brackets(
            [row for row in trials if row.get("train_eligible", True)],
            required_seed_count=len(seeds),
        )
        by_candidate: Dict[str, List[Mapping[str, Any]]] = {}
        for trial in trials:
            by_candidate.setdefault(str(trial["candidate_id"]), []).append(trial)
        roles: Dict[str, Dict[str, Any]] = {}
        for event_class, selection in brackets["selections"].items():
            for role in ("positive", "negative"):
                candidate_id = selection[f"{role}_candidate_id"]
                record = roles.setdefault(
                    candidate_id,
                    {
                        "positive_for": set(),
                        "hard_negative_for": set(),
                        "distances": {},
                    },
                )
                key = "positive_for" if role == "positive" else "hard_negative_for"
                record[key].add(event_class)
                record["distances"][event_class] = selection[
                    f"{role}_boundary_distance"
                ]
        recipes = []
        for candidate_id, role in sorted(roles.items()):
            reference = by_candidate[candidate_id][0]
            distances = dict(role["distances"])
            recipes.append(
                TargetedRecipe(
                    recipe_id=candidate_id,
                    family=str(reference["family"]),
                    protocol_variant=(
                        "adaptive_positive"
                        if role["positive_for"]
                        else "adaptive_hard_negative"
                    ),
                    duration_ms=int(reference["duration_ms"]),
                    actions_by_step=action_schedule_from_json(
                        reference["input_schedule"]
                    ),
                    positive_for=tuple(sorted(role["positive_for"])),
                    hard_negative_for=tuple(
                        sorted(role["hard_negative_for"] - role["positive_for"])
                    ),
                    branch_id=str(reference["branch_id"]),
                    boundary_distance=float(
                        min(distances.values(), key=abs)
                    ),
                    metadata={
                        "boundary_distance_by_class": distances,
                        "pilot_seed_count": len(seeds),
                    },
                )
            )

        heldout_trials = [
            row for row in trials if not row.get("train_eligible", True)
        ]
        if heldout_trials:
            heldout_id = str(heldout_trials[0]["candidate_id"])
            heldout_group = [
                row for row in heldout_trials if row["candidate_id"] == heldout_id
            ]
            event_sets = [set(row["event_kinds"]) for row in heldout_group]
            positive_for = tuple(
                kind
                for kind in (
                    "axonal_spike",
                    "somatic_spike",
                    "backpropagating_ap",
                    "calcium_spike",
                    "nmda_spike",
                    "nmda_plateau",
                )
                if event_sets and all(kind in observed for observed in event_sets)
            )
            hard_negative_for = tuple(
                kind
                for kind in (
                    "axonal_spike",
                    "somatic_spike",
                    "backpropagating_ap",
                    "calcium_spike",
                    "nmda_spike",
                    "nmda_plateau",
                )
                if event_sets and all(kind not in observed for observed in event_sets)
            )
            reference = heldout_group[0]
            recipes.append(
                TargetedRecipe(
                    recipe_id=heldout_id,
                    family=str(reference["family"]),
                    protocol_variant="held_out_morphological_branch",
                    duration_ms=int(reference["duration_ms"]),
                    actions_by_step=action_schedule_from_json(
                        reference["input_schedule"]
                    ),
                    positive_for=positive_for,
                    hard_negative_for=hard_negative_for,
                    branch_id=str(reference["branch_id"]),
                    metadata={
                        "train_eligible": False,
                        "pilot_seed_count": len(heldout_group),
                    },
                )
            )

        # Pilot recovery probes by replaying the selected stimulus after a
        # configurable interval. Only empirically recovered variants enter the
        # recipe catalog.
        recovery_recipes: List[TargetedRecipe] = []
        recovery_duration = int(config.get("recovery_duration_ms", 100))
        recovery_candidates = [row for row in recipes if row.positive_for]
        recovery_delays = list(
            map(int, config.get("recovery_probe_delays_ms", (20, 50, 80)))
        )
        recovery_progress = _ConsoleProgress(
            "pilot recovery", len(recovery_candidates) * len(recovery_delays)
        )
        recovery_trial_index = 0
        for recipe in recovery_candidates:
            first_step = min(recipe.actions_by_step, default=0)
            final_step = max(recipe.actions_by_step, default=first_step)
            for delay in recovery_delays:
                recovery_trial_index += 1
                shift = int(delay)
                if final_step + shift >= recovery_duration:
                    recovery_progress.update(
                        recovery_trial_index,
                        detail=f"{recipe.recipe_id}; delay {delay} fuori finestra",
                    )
                    continue
                schedule = {step: tuple(actions) for step, actions in recipe.actions_by_step.items()}
                for step, actions in recipe.actions_by_step.items():
                    schedule[int(step) + shift] = tuple(actions)
                candidate_id = f"{recipe.recipe_id}-recovery-{delay}ms"
                trajectory = ProtocolTrajectory(
                    trajectory_id=f"pilot-{candidate_id}-seed{seeds[0]}",
                    category=(
                        "somatic_events"
                        if recipe.family == "targeted_somatic_bap"
                        else "dendritic_events"
                    ),
                    protocol=recipe.family,
                    protocol_id=candidate_id,
                    protocol_variant="recovery_probe",
                    seed=seeds[0],
                    duration_ms=recovery_duration,
                    split="recovery_test",
                    actions_by_step=schedule,
                    stimulus_onset_step=first_step,
                )
                times, traces = self._run_trajectory_prefix_in_memory(
                    trajectory, recovery_duration
                )
                events = self._extract_events(times, traces)
                observed = {str(row["kind"]) for row in events}
                event_counts = Counter(str(row["kind"]) for row in events)
                uncensored = all(not bool(row["right_censored"]) for row in events)
                voltage_recovered = all(
                    abs(float(values[-1]) - float(values[0]))
                    <= float(config.get("recovery_voltage_tolerance_mv", 5.0))
                    for label, values in traces.items()
                    if label in self.audit.representatives
                )
                calcium_trace = self.np.asarray(
                    traces.get("cai_event_probe_mM", ()), dtype=float
                )
                if calcium_trace.size:
                    calcium_excursion = float(
                        self.np.max(self.np.abs(calcium_trace - calcium_trace[0]))
                    )
                    calcium_tolerance = max(
                        1.0e-8,
                        float(config.get("recovery_calcium_fraction", 0.10))
                        * calcium_excursion,
                    )
                    calcium_recovered = (
                        abs(float(calcium_trace[-1] - calcium_trace[0]))
                        <= calcium_tolerance
                    )
                else:
                    calcium_recovered = False
                selected_synapses = sorted(
                    {
                        int(action.synapse_id)
                        for actions in recipe.actions_by_step.values()
                        for action in actions
                        if action.kind == "synaptic_event"
                    }
                )
                final_pv = []
                for synapse_id in selected_synapses:
                    record = self.audit.synapse_records[synapse_id]
                    pv_index = 3 if record["class_name"] == "ProbAMPANMDA2" else 1
                    final_pv.append(float(record["netcon"].weight[pv_index]))
                synapse_recovered = not final_pv or min(final_pv) >= float(
                    config.get("recovery_min_pv", 0.75)
                )
                repeated_response = all(
                    event_counts.get(kind, 0) >= 2
                    for kind in recipe.positive_for
                )
                if not (
                    set(recipe.positive_for).issubset(observed)
                    and uncensored
                    and voltage_recovered
                    and calcium_recovered
                    and synapse_recovered
                    and repeated_response
                ):
                    recovery_progress.update(
                        recovery_trial_index,
                        detail=f"{candidate_id}; non recuperato",
                    )
                    continue
                recovery_recipes.append(
                    TargetedRecipe(
                        recipe_id=candidate_id,
                        family=recipe.family,
                        protocol_variant="pilot_validated_recovery_probe",
                        duration_ms=recovery_duration,
                        actions_by_step=schedule,
                        positive_for=recipe.positive_for,
                        hard_negative_for=recipe.hard_negative_for,
                        branch_id=recipe.branch_id,
                        boundary_distance=recipe.boundary_distance,
                        recovery_probe_delay_ms=float(delay),
                        metadata={
                            **dict(recipe.metadata),
                            "train_eligible": False,
                            "recovery_probe": True,
                            "pilot_validated": True,
                            "voltage_recovered": True,
                            "calcium_recovered": True,
                            "synapse_recovered": True,
                            "repeated_probe_response": True,
                            "minimum_final_pv": min(final_pv, default=1.0),
                            "events_uncensored": True,
                        },
                    )
                )
                recovery_progress.update(
                    recovery_trial_index,
                    detail=f"{candidate_id}; valido",
                )
        if recovery_candidates:
            recovery_progress.update(
                recovery_trial_index,
                detail=f"ricette valide={len(recovery_recipes)}",
                force=True,
            )
        recipes.extend(recovery_recipes)
        self.targeted_recipe_catalog = recipes
        report = {
            "schema_version": TARGETED_DATASET_SCHEMA_VERSION,
            "valid": brackets["valid"]
            and bool(recipes)
            and bool(heldout_trials)
            and bool(recovery_recipes),
            "trial_count": len(trials),
            "candidate_count": len(by_candidate),
            "seeds": seeds,
            "adaptive_brackets": brackets,
            "recipe_count": len(recipes),
            "heldout_branch_recipe_count": sum(
                not bool(row.metadata.get("train_eligible", True))
                and not bool(row.metadata.get("recovery_probe", False))
                for row in recipes
            ),
            "recovery_recipe_count": len(recovery_recipes),
            "recipes": [
                {
                    "recipe_id": row.recipe_id,
                    "family": row.family,
                    "positive_for": list(row.positive_for),
                    "hard_negative_for": list(row.hard_negative_for),
                    "branch_id": row.branch_id,
                    "duration_ms": row.duration_ms,
                    "boundary_distance": row.boundary_distance,
                }
                for row in recipes
            ],
        }
        pilot_dir = self.output_dir / "targeted_pilot"
        pilot_dir.mkdir(parents=True, exist_ok=True)
        self.pd.DataFrame(self._parquet_safe_rows(trials)).to_parquet(
            pilot_dir / "candidate_trials.parquet", index=False
        )
        write_json(pilot_dir / "pilot_report.json", report)
        write_json(
            pilot_dir / "recipe_catalog.json",
            {
                "schema_version": TARGETED_DATASET_SCHEMA_VERSION,
                "recipes": [
                    {
                        "recipe_id": row.recipe_id,
                        "family": row.family,
                        "protocol_variant": row.protocol_variant,
                        "duration_ms": row.duration_ms,
                        "actions_by_step": {
                            str(step): [action.to_dict() for action in actions]
                            for step, actions in row.actions_by_step.items()
                        },
                        "positive_for": list(row.positive_for),
                        "hard_negative_for": list(row.hard_negative_for),
                        "branch_id": row.branch_id,
                        "snapshot_id": row.snapshot_id,
                        "boundary_distance": row.boundary_distance,
                        "metadata": dict(row.metadata),
                    }
                    for row in recipes
                ],
            },
        )
        if not report["valid"]:
            raise RuntimeError(
                "targeted biological pilot did not bracket every event class; "
                f"inspect {pilot_dir / 'pilot_report.json'}"
            )
        return report

    def generate_dataset(
        self, protocols: Optional[Sequence[ProtocolTrajectory]] = None
    ) -> Dict[str, Any]:
        """Generate only after the release and targeted-plan gates pass."""

        if not self.causal_release_pilot_report.get("valid"):
            raise RuntimeError("run_causal_release_pilot() must pass first")
        protocols = list(protocols or ())
        if not protocols:
            raise ValueError("an explicit targeted protocol plan is required")
        expected_hash = self.targeted_preflight_report.get("protocol_plan_sha256")
        observed_hash = self._protocol_plan_sha256(protocols)
        if not self.targeted_preflight_report.get("valid") or expected_hash != observed_hash:
            raise RuntimeError(
                "targeted preflight must pass for the exact protocol plan before generation"
            )
        required_snapshot_ids = {
            str(row.metadata.get("snapshot_id", row.snapshot_source))
            for row in protocols
        }
        missing_snapshots = required_snapshot_ids - set(self.snapshot_bank)
        if missing_snapshots:
            raise RuntimeError(
                f"snapshot bank is incomplete: {sorted(missing_snapshots)}"
            )
        self.release_rows = []
        self._bind_protocol_registry(protocols)
        self._collect_release_rows = True
        try:
            manifest = DiagnosticDatasetSession.generate_dataset(self, protocols)
        finally:
            self._collect_release_rows = False
        manifest.update(
            {
                "schema_version": TARGETED_DATASET_SCHEMA_VERSION,
                "compatible_base_schema_version": "1.0.1",
                "canonical_state_layout_sha256": self.state_schema[
                    "canonical_state_layout_sha256"
                ],
                "release_schema": "release_schema.json",
                "input_views": ["U_scheduled", "U_rng", "U_realized"],
                "release_outcomes": "release_outcomes.parquet",
                "causal_release_pilot": "causal_release_pilot.json",
                "targeted_preflight": "targeted_preflight_report.json",
                "snapshot_bank": "snapshot_bank.json",
            }
        )
        self.dataset_manifest = manifest
        write_json(self.output_dir / "dataset_manifest.json", manifest)
        self.pd.DataFrame(self._parquet_safe_rows(self.release_rows)).to_parquet(
            self.output_dir / "release_outcomes.parquet", index=False
        )
        table_report = self._write_targeted_tables(protocols)
        manifest["indices"] = {
            "protocols": "protocols.parquet",
            "episodes": "episodes.parquet",
            "transitions": "transition_index.parquet",
            "events": "events.parquet",
            "release_outcomes": "release_outcomes.parquet",
            "branching_pairs": "branching_pairs.parquet",
            "splits": "splits.json",
        }
        manifest["dataset_card"] = "dataset_card.json"
        manifest["storage_report"] = "storage_report.json"
        manifest["table_report"] = table_report
        write_json(self.output_dir / "dataset_manifest.json", manifest)
        return manifest

    def _bind_protocol_registry(
        self, protocols: Sequence[ProtocolTrajectory]
    ) -> None:
        self.protocol_registry = {}
        self.protocol_rows = []
        for row in protocols:
            metadata = dict(row.metadata)
            selected_synapses = sorted(
                {
                    int(action.synapse_id)
                    for actions in row.actions_by_step.values()
                    for action in actions
                    if action.kind == "synaptic_event"
                }
            )
            contract = {
                "trajectory_id": row.trajectory_id,
                "category": row.category,
                "protocol": row.protocol,
                "protocol_id": row.protocol_id or row.protocol,
                "protocol_variant": row.protocol_variant,
                "seed": int(row.seed),
                "duration_ms": int(row.duration_ms),
                "split": row.split,
                "stimulus_onset_step": int(row.stimulus_onset_step),
                "required_event_kinds": list(row.required_event_kinds),
                "negative_control": bool(row.negative_control),
                "event_probe_label": row.event_probe_label,
                "event_probe_segment_id": metadata.get(
                    "event_probe_segment_id"
                ),
                "selected_synapse_ids": selected_synapses,
                "snapshot_source": row.snapshot_source,
                **metadata,
            }
            self.protocol_registry[row.trajectory_id] = contract
            self.protocol_rows.append(contract)

    def _write_targeted_tables(
        self, protocols: Sequence[ProtocolTrajectory]
    ) -> Dict[str, Any]:
        try:
            import h5py
        except ImportError as error:
            raise RuntimeError("targeted tables require h5py") from error

        transition_rows = []
        event_rows = []
        actual_events: Dict[str, set] = {}
        with h5py.File(self.transition_path, "r") as handle:
            count = int(handle.attrs["transition_count"])
            for index in range(count):
                trajectory_id = self._decode(
                    handle["metadata/trajectory_id"][index]
                )
                contract = self.protocol_registry[trajectory_id]
                labels = json.loads(handle["events/labels_json"][index])
                event_flags = sorted({str(row["kind"]) for row in labels})
                actual_events.setdefault(trajectory_id, set()).update(event_flags)
                transition_rows.append(
                    {
                        "transition_id": index,
                        "trajectory_id": trajectory_id,
                        "episode_id": contract.get("episode_id", trajectory_id),
                        "protocol_id": contract["protocol_id"],
                        "protocol_variant": contract["protocol_variant"],
                        "split": contract["split"],
                        "seed": contract["seed"],
                        "branch_id": contract.get("branch_id", "unknown"),
                        "snapshot_id": contract.get(
                            "snapshot_id", contract["snapshot_source"]
                        ),
                        "step_index": int(handle["metadata/step_index"][index]),
                        "absolute_time_ms": float(
                            handle["metadata/start_time_ms"][index]
                        ),
                        "event_flags": event_flags,
                        "release_count": len(
                            json.loads(
                                handle["release_outcomes/records_json"][index]
                            )
                        ),
                        "microtrace_mode": self._decode(
                            handle["metadata/microtrace_mode"][index]
                        ),
                    }
                )
                for event_index, event in enumerate(labels):
                    event_rows.append(
                        {
                            "trajectory_id": trajectory_id,
                            "episode_id": contract.get("episode_id", trajectory_id),
                            "transition_id": index,
                            "trajectory_event_id": event.get(
                                "trajectory_event_id", event_index
                            ),
                            "split": contract["split"],
                            "seed": contract["seed"],
                            **event,
                        }
                    )

        episodes = []
        for contract in self.protocol_rows:
            observed = sorted(actual_events.get(contract["trajectory_id"], set()))
            intended_negatives = list(contract.get("hard_negative_for", ()))
            valid_negatives = [
                kind for kind in intended_negatives if kind not in observed
            ]
            episodes.append(
                {
                    **contract,
                    "episode_id": contract.get(
                        "episode_id", contract["trajectory_id"]
                    ),
                    "snapshot_id": contract.get(
                        "snapshot_id", contract["snapshot_source"]
                    ),
                    "branch_id": contract.get("branch_id", "unknown"),
                    "event_labels": observed,
                    "hard_negative_for": valid_negatives,
                    "hard_negative_failures": sorted(
                        set(intended_negatives) - set(valid_negatives)
                    ),
                }
            )
        support = summarize_independent_support(episodes)
        support_validation = validate_minimum_support(support)
        release_success = Counter(
            (
                str(row["synapse_type"]),
                "success" if bool(row["release_success"]) else "failure",
            )
            for row in self.release_rows
        )
        branching_contracts: Dict[str, List[Mapping[str, Any]]] = {}
        for row in episodes:
            pair_id = row.get("branch_pair_id")
            if pair_id:
                branching_contracts.setdefault(str(pair_id), []).append(row)
        branching_rows = []
        release_pair_rows = []
        transition_ids_by_trajectory: Dict[str, List[int]] = {}
        for row in transition_rows:
            transition_ids_by_trajectory.setdefault(
                row["trajectory_id"], []
            ).append(int(row["transition_id"]))
        with h5py.File(self.transition_path, "r") as handle:
            for pair_id, pair in sorted(branching_contracts.items()):
                if len(pair) != 2:
                    raise RuntimeError(
                        f"branch pair {pair_id} must contain exactly two futures"
                    )
                left, right = pair
                left_ids = sorted(
                    transition_ids_by_trajectory[left["trajectory_id"]]
                )
                right_ids = sorted(
                    transition_ids_by_trajectory[right["trajectory_id"]]
                )
                horizon_count = min(len(left_ids), len(right_ids))
                distances = []
                for horizon in range(horizon_count):
                    left_v = handle["states/voltage/t_plus_1"][
                        left_ids[horizon], :
                    ]
                    right_v = handle["states/voltage/t_plus_1"][
                        right_ids[horizon], :
                    ]
                    distances.append(
                        float(self.np.max(self.np.abs(left_v - right_v)))
                    )
                initial_errors = []
                for category in self.state_variables:
                    initial_errors.append(
                        float(
                            self.np.max(
                                self.np.abs(
                                    handle[f"states/{category}/t"][left_ids[0], :]
                                    - handle[f"states/{category}/t"][right_ids[0], :]
                                )
                            )
                        )
                    )
                initial_errors.append(
                    float(
                        self.np.max(
                            self.np.abs(
                                handle["rng_state/t"][left_ids[0], :]
                                - handle["rng_state/t"][right_ids[0], :]
                            )
                        )
                    )
                )
                first_divergence = next(
                    (
                        index + 1
                        for index, value in enumerate(distances)
                        if value > 1.0e-3
                    ),
                    None,
                )
                regional_max = {}
                for label, segment_id in self.audit.representatives.items():
                    regional_max[label] = max(
                        (
                            abs(
                                float(
                                    handle["states/voltage/t_plus_1"][
                                        left_ids[horizon], int(segment_id)
                                    ]
                                )
                                - float(
                                    handle["states/voltage/t_plus_1"][
                                        right_ids[horizon], int(segment_id)
                                    ]
                                )
                            )
                            for horizon in range(horizon_count)
                        ),
                        default=0.0,
                    )
                release_left = [
                    json.loads(handle["release_outcomes/records_json"][index])
                    for index in left_ids
                ]
                release_right = [
                    json.loads(handle["release_outcomes/records_json"][index])
                    for index in right_ids
                ]
                branching_rows.append(
                    {
                        "branch_pair_id": pair_id,
                        "branching_distance": left.get(
                            "branching_distance", right.get("branching_distance")
                        ),
                        "left_trajectory_id": left["trajectory_id"],
                        "right_trajectory_id": right["trajectory_id"],
                        "snapshot_id": left["snapshot_id"],
                        "seed": left["seed"],
                        "same_initial_state_max_error": max(initial_errors),
                        "distance_teacher_by_horizon_mv": distances,
                        "maximum_teacher_distance_mv": max(distances, default=0.0),
                        "first_divergence_ms": first_divergence,
                        "regional_max_distance_mv": regional_max,
                        "teacher_quasi_identical": max(distances, default=0.0)
                        <= 1.0e-6,
                        "left_event_labels": left["event_labels"],
                        "right_event_labels": right["event_labels"],
                        "event_labels_differ": set(left["event_labels"])
                        != set(right["event_labels"]),
                        "release_outcomes_differ": self._outcome_signature(
                            [item for rows in release_left for item in rows]
                        )
                        != self._outcome_signature(
                            [item for rows in release_right for item in rows]
                        ),
                        "modified_parameter": (
                            f"{left['protocol_id']} vs {right['protocol_id']}"
                        ),
                    }
                )
            release_contracts: Dict[str, List[Mapping[str, Any]]] = {}
            for row in episodes:
                pair_id = row.get("release_pair_id")
                if pair_id:
                    release_contracts.setdefault(str(pair_id), []).append(row)
            for pair_id, pair in sorted(release_contracts.items()):
                if len(pair) != 2:
                    raise RuntimeError(
                        f"release pair {pair_id} must contain exactly two futures"
                    )
                left, right = pair
                left_ids = sorted(
                    transition_ids_by_trajectory[left["trajectory_id"]]
                )
                right_ids = sorted(
                    transition_ids_by_trajectory[right["trajectory_id"]]
                )
                left_inputs = [
                    json.loads(handle["inputs/U_scheduled_json"][index])
                    for index in left_ids
                ]
                right_inputs = [
                    json.loads(handle["inputs/U_scheduled_json"][index])
                    for index in right_ids
                ]
                left_realized = [
                    json.loads(handle["inputs/U_realized_json"][index])
                    for index in left_ids
                ]
                right_realized = [
                    json.loads(handle["inputs/U_realized_json"][index])
                    for index in right_ids
                ]
                release_pair_rows.append(
                    {
                        "release_pair_id": pair_id,
                        "same_snapshot": left["snapshot_id"] == right["snapshot_id"],
                        "different_seed": int(left["seed"]) != int(right["seed"]),
                        "same_scheduled_input": canonical_json_sha256(
                            {"rows": left_inputs}
                        )
                        == canonical_json_sha256({"rows": right_inputs}),
                        "release_outcomes_differ": canonical_json_sha256(
                            {"rows": left_realized}
                        )
                        != canonical_json_sha256({"rows": right_realized}),
                    }
                )
        self.pd.DataFrame(
            self._parquet_safe_rows(self.protocol_rows)
        ).to_parquet(self.output_dir / "protocols.parquet", index=False)
        self.pd.DataFrame(self._parquet_safe_rows(episodes)).to_parquet(
            self.output_dir / "episodes.parquet", index=False
        )
        self.pd.DataFrame(self._parquet_safe_rows(transition_rows)).to_parquet(
            self.output_dir / "transition_index.parquet", index=False
        )
        self.pd.DataFrame(self._parquet_safe_rows(event_rows)).to_parquet(
            self.output_dir / "events.parquet", index=False
        )
        branching_columns = [
            "branch_pair_id",
            "branching_distance",
            "left_trajectory_id",
            "right_trajectory_id",
        ]
        branching_frame = self.pd.DataFrame(
            self._parquet_safe_rows(branching_rows)
        )
        if branching_frame.empty:
            branching_frame = self.pd.DataFrame(columns=branching_columns)
        branching_frame.to_parquet(
            self.output_dir / "branching_pairs.parquet", index=False
        )
        split_payload = {
            split: sorted(
                row["trajectory_id"]
                for row in self.protocol_rows
                if row["split"] == split
            )
            for split in sorted({row["split"] for row in self.protocol_rows})
        }
        write_json(self.output_dir / "splits.json", split_payload)
        dataset_card = {
            "schema_version": TARGETED_DATASET_SCHEMA_VERSION,
            "transition_count": len(transition_rows),
            "episode_count": len(episodes),
            "support": support,
            "support_validation": support_validation,
            "release_outcomes": {
                f"{kind}:{outcome}": count
                for (kind, outcome), count in sorted(release_success.items())
            },
            "release_identifiability_pairs": release_pair_rows,
            "event_count": len(event_rows),
            "event_cooccurrence": dict(
                Counter(
                    "+".join(row["event_labels"]) or "none" for row in episodes
                )
            ),
            "branch_count": len({row["branch_id"] for row in episodes}),
            "seed_count": len({int(row["seed"]) for row in episodes}),
            "snapshot_count": len({row["snapshot_id"] for row in episodes}),
            "protocol_variant_count": len(
                {row["protocol_variant"] for row in episodes}
            ),
            "episodes_by_seed": dict(
                Counter(str(int(row["seed"])) for row in episodes)
            ),
            "episodes_by_branch": dict(
                Counter(str(row["branch_id"]) for row in episodes)
            ),
            "episodes_by_protocol_variant": dict(
                Counter(str(row["protocol_variant"]) for row in episodes)
            ),
            "episodes_by_split": dict(
                Counter(str(row["split"]) for row in episodes)
            ),
            "boundary_distance": {
                "minimum": min(
                    (float(row.get("boundary_distance", 0.0)) for row in episodes),
                    default=0.0,
                ),
                "maximum": max(
                    (float(row.get("boundary_distance", 0.0)) for row in episodes),
                    default=0.0,
                ),
                "values": [
                    float(row.get("boundary_distance", 0.0)) for row in episodes
                ],
            },
            "event_duration_ms_by_kind": {
                kind: [
                    float(row["duration_ms"])
                    for row in event_rows
                    if row["kind"] == kind
                ]
                for kind in sorted({row["kind"] for row in event_rows})
            },
            "recovery_probe_delays_ms": sorted(
                {
                    float(row["recovery_probe_delay_ms"])
                    for row in episodes
                    if row.get("recovery_probe_delay_ms") is not None
                    and not self.pd.isna(row.get("recovery_probe_delay_ms"))
                }
            ),
            "branching_pair_count": len(branching_rows),
            "branching_pairs": branching_rows,
            "hdf5_bytes": self.transition_path.stat().st_size,
        }
        write_json(self.output_dir / "dataset_card.json", dataset_card)
        write_json(
            self.output_dir / "storage_report.json",
            {
                "schema_version": TARGETED_DATASET_SCHEMA_VERSION,
                "hdf5_bytes": self.transition_path.stat().st_size,
                "hdf5_bytes_per_transition": int(
                    self.transition_path.stat().st_size / len(transition_rows)
                ),
                "release_table_bytes": (
                    self.output_dir / "release_outcomes.parquet"
                ).stat().st_size,
                "microtrace_interval_ms": 0.025,
                "canonical_boundary_state_unchanged": True,
            },
        )
        self._plot_targeted_figures(dataset_card)
        return {
            "transition_count": len(transition_rows),
            "episode_count": len(episodes),
            "event_count": len(event_rows),
            "release_count": len(self.release_rows),
            "support_valid": support_validation["valid"],
        }

    def _plot_targeted_figures(self, card: Mapping[str, Any]) -> None:
        import matplotlib.pyplot as plt

        figures = self.output_dir / "figures"
        figures.mkdir(parents=True, exist_ok=True)
        classes = [
            "axonal_spike",
            "somatic_spike",
            "backpropagating_ap",
            "calcium_spike",
            "nmda_spike",
            "nmda_plateau",
        ]
        support = card["support"]
        positives = [
            int(support.get(kind, {}).get("train", {}).get("positive_episode_count", 0))
            for kind in classes
        ]
        negatives = [
            int(
                support.get(kind, {})
                .get("train", {})
                .get("hard_negative_episode_count", 0)
            )
            for kind in classes
        ]
        x = self.np.arange(len(classes))
        figure, axis = plt.subplots(figsize=(11, 4.5))
        axis.bar(x - 0.2, positives, width=0.4, label="positive episodes")
        axis.bar(x + 0.2, negatives, width=0.4, label="hard negatives")
        axis.set_xticks(x, classes, rotation=30, ha="right")
        axis.set_ylabel("independent train episodes")
        axis.legend()
        figure.tight_layout()
        figure.savefig(figures / "independent_support.png", dpi=180)
        plt.close(figure)

        release_counts = card["release_outcomes"]
        labels = sorted(release_counts)
        figure, axis = plt.subplots(figsize=(9, 4))
        axis.bar(labels, [release_counts[label] for label in labels])
        axis.set_ylabel("presynaptic events")
        axis.tick_params(axis="x", rotation=25)
        figure.tight_layout()
        figure.savefig(figures / "release_success_failure.png", dpi=180)
        plt.close(figure)

        figure, axis = plt.subplots(figsize=(8, 4))
        axis.hist(card["boundary_distance"]["values"], bins=31)
        axis.axvline(0.0, color="black", linewidth=1)
        axis.set_xlabel("normalized boundary_distance")
        axis.set_ylabel("episodes")
        figure.tight_layout()
        figure.savefig(figures / "boundary_distance.png", dpi=180)
        plt.close(figure)

        figure, axis = plt.subplots(figsize=(8, 4))
        for row in card["branching_pairs"]:
            axis.plot(
                self.np.arange(1, len(row["distance_teacher_by_horizon_mv"]) + 1),
                row["distance_teacher_by_horizon_mv"],
                label=f"{row['branching_distance']}:{row['branch_pair_id']}",
            )
        axis.set_xlabel("horizon (ms)")
        axis.set_ylabel("teacher max |ΔV| (mV)")
        if card["branching_pairs"]:
            axis.legend()
        figure.tight_layout()
        figure.savefig(figures / "branching_teacher_distance.png", dpi=180)
        plt.close(figure)

    def validate_dataset_v1_1(self) -> Dict[str, Any]:
        """Run structural, exhaustive replay, causal, support and split gates."""

        structural = validate_hdf5_store(self.transition_path)
        exhaustive = self._exhaustive_sequential_replay()
        card = json.loads(
            (self.output_dir / "dataset_card.json").read_text(encoding="utf-8")
        )
        observed_splits = set(
            json.loads((self.output_dir / "splits.json").read_text(encoding="utf-8"))
        )
        required_splits = {
            "train",
            "validation",
            "deterministic_test",
            "event_boundary_test",
            "held_out_branch_test",
            "held_out_seed_test",
            "branching_near_test",
            "branching_far_test",
            "release_identifiability_test",
            "recovery_test",
        }
        release_rows = self.release_rows
        if not release_rows:
            release_rows = self.pd.read_parquet(
                self.output_dir / "release_outcomes.parquet"
            ).to_dict("records")
        release_valid = bool(release_rows) and all(
            not bool(row.get("future_state_used", False))
            and row["source"] == "direct_same_timestamp_synapse_state_discontinuity"
            for row in release_rows
        )
        episode_rows = self.pd.read_parquet(
            self.output_dir / "episodes.parquet"
        ).to_dict("records")
        seed_splits: Dict[int, set] = {}
        snapshot_splits: Dict[str, set] = {}
        for row in episode_rows:
            seed_splits.setdefault(int(row["seed"]), set()).add(str(row["split"]))
            snapshot_splits.setdefault(str(row["snapshot_id"]), set()).add(
                str(row["split"])
            )
        leaking_seeds = {
            seed: sorted(splits)
            for seed, splits in seed_splits.items()
            if len(splits) > 1
        }
        leaking_snapshots = {
            snapshot: sorted(splits)
            for snapshot, splits in snapshot_splits.items()
            if len(splits) > 1
        }
        train_branches = {
            str(row["branch_id"])
            for row in episode_rows
            if row["split"] == "train"
        }
        heldout_branches = {
            str(row["branch_id"])
            for row in episode_rows
            if row["split"] == "held_out_branch_test"
        }
        heldout_branch_valid = bool(heldout_branches) and not (
            train_branches & heldout_branches
        )
        branching_rows = self.pd.read_parquet(
            self.output_dir / "branching_pairs.parquet"
        ).to_dict("records")
        branching_valid = (
            {str(row["branching_distance"]) for row in branching_rows}
            == {"near", "far"}
            and all(
                float(row["same_initial_state_max_error"]) == 0.0
                for row in branching_rows
            )
        )
        recovery_rows = [
            row for row in episode_rows if row["split"] == "recovery_test"
        ]
        recovery_valid = bool(recovery_rows) and all(
            bool(row.get("pilot_validated"))
            and bool(row.get("recovery_probe"))
            for row in recovery_rows
        )
        release_pair_rows = card.get("release_identifiability_pairs", [])
        release_identifiability_valid = bool(release_pair_rows) and all(
            bool(row["same_snapshot"])
            and bool(row["different_seed"])
            and bool(row["same_scheduled_input"])
            and bool(row["release_outcomes_differ"])
            for row in release_pair_rows
        )
        uncensored_required = all(
            not bool(row.get("right_censored"))
            for row in self.pd.read_parquet(self.output_dir / "events.parquet").to_dict(
                "records"
            )
            if row.get("kind") in {"calcium_spike", "nmda_plateau"}
        )
        blockers = []
        if not structural["valid"]:
            blockers.append("HDF5 structural validation failed")
        if not exhaustive["valid"]:
            blockers.append("exhaustive transition/release replay failed")
        if not release_valid:
            blockers.append("causal release contract failed")
        if not card["support_validation"]["valid"]:
            blockers.append("minimum independent support is not satisfied")
        if observed_splits != required_splits:
            blockers.append("required targeted split set is incomplete")
        if leaking_seeds:
            blockers.append("Random123 seed leakage across splits")
        if leaking_snapshots:
            blockers.append("initial snapshot leakage across splits")
        if not heldout_branch_valid:
            blockers.append("held-out branch is present in train")
        if not branching_valid:
            blockers.append("near/far branching contract failed")
        if not recovery_valid:
            blockers.append("recovery probes are missing or not pilot-validated")
        if not release_identifiability_valid:
            blockers.append("release-identifiability pair did not isolate RNG outcome")
        if not uncensored_required:
            blockers.append("a required duration event is right censored")
        if len(self.audit.live_segments) != 642:
            blockers.append("canonical segment mapping changed")
        if self.state_schema["canonical_state_layout_sha256"] != canonical_json_sha256(
            {
                "index_contract": self.state_schema["index_contract"],
                "categories": self.state_schema["categories"],
                "rng_state": self.state_schema["rng_state"],
                "variables": self.state_schema["variables"],
            }
        ):
            blockers.append("canonical state layout changed")
        report = {
            "schema_version": TARGETED_DATASET_SCHEMA_VERSION,
            "valid": not blockers,
            "blockers": blockers,
            "structural": structural,
            "exhaustive_replay": exhaustive,
            "causal_release_valid": release_valid,
            "support": card["support_validation"],
            "required_splits": sorted(required_splits),
            "observed_splits": sorted(observed_splits),
            "seed_split_leaks": leaking_seeds,
            "snapshot_split_leaks": leaking_snapshots,
            "heldout_branch_valid": heldout_branch_valid,
            "branching_valid": branching_valid,
            "recovery_valid": recovery_valid,
            "release_identifiability_valid": release_identifiability_valid,
            "uncensored_required_events": uncensored_required,
            "teacher_commit": PINNED_TEACHER_COMMIT,
            "segment_count": len(self.audit.live_segments),
            "state_layout_unchanged": "canonical state layout changed" not in blockers,
        }
        write_json(self.output_dir / "validation_report.json", report)
        self._write_artifact_index()
        if blockers:
            raise RuntimeError(f"diagnostic dataset v1.1 validation failed: {blockers}")
        return report

    def accept_targeted_protocol_plan(
        self,
        protocols: Sequence[ProtocolTrajectory],
        *,
        pilot_report: Mapping[str, Any],
    ) -> Dict[str, Any]:
        """Bind generation to a reviewed pilot and exact protocol hash."""

        transition_count = sum(int(row.duration_ms) for row in protocols)
        blockers = []
        if not pilot_report.get("valid"):
            blockers.append("targeted biological pilot is not valid")
        if not 10_000 <= transition_count <= 30_000:
            blockers.append("planned transition count is outside 10k-30k")
        snapshot_ids = {
            str(row.metadata.get("snapshot_id", row.snapshot_source))
            for row in protocols
        }
        if snapshot_ids - set(self.snapshot_bank):
            blockers.append("split-specific snapshot bank is incomplete")
        required_splits = {
            "train",
            "validation",
            "deterministic_test",
            "event_boundary_test",
            "held_out_branch_test",
            "held_out_seed_test",
            "branching_near_test",
            "branching_far_test",
            "release_identifiability_test",
            "recovery_test",
        }
        observed_splits = {row.split for row in protocols}
        if observed_splits != required_splits:
            blockers.append("targeted split set is incomplete")
        seed_splits: Dict[int, set] = {}
        snapshot_splits: Dict[str, set] = {}
        planned_episode_rows = []
        for row in protocols:
            metadata = dict(row.metadata)
            seed_splits.setdefault(int(row.seed), set()).add(row.split)
            snapshot_id = str(metadata.get("snapshot_id", row.snapshot_source))
            snapshot_splits.setdefault(snapshot_id, set()).add(row.split)
            planned_episode_rows.append(
                {
                    "episode_id": metadata.get("episode_id", row.trajectory_id),
                    "trajectory_id": row.trajectory_id,
                    "split": row.split,
                    "seed": int(row.seed),
                    "snapshot_id": snapshot_id,
                    "branch_id": str(metadata.get("branch_id", "unknown")),
                    "protocol_variant": row.protocol_variant,
                    "event_labels": list(metadata.get("positive_for", ())),
                    "hard_negative_for": list(
                        metadata.get("hard_negative_for", ())
                    ),
                }
            )
        leaking_seeds = {
            seed: sorted(splits)
            for seed, splits in seed_splits.items()
            if len(splits) > 1
        }
        leaking_snapshots = {
            snapshot: sorted(splits)
            for snapshot, splits in snapshot_splits.items()
            if len(splits) > 1
        }
        if leaking_seeds:
            blockers.append("Random123 seeds leak across splits")
        if leaking_snapshots:
            blockers.append("initial snapshots leak across splits")
        train_branches = {
            row["branch_id"]
            for row in planned_episode_rows
            if row["split"] == "train"
        }
        heldout_branches = {
            row["branch_id"]
            for row in planned_episode_rows
            if row["split"] == "held_out_branch_test"
        }
        if not heldout_branches or train_branches & heldout_branches:
            blockers.append("held-out branch is present in train")
        planned_support = summarize_independent_support(planned_episode_rows)
        planned_support_validation = validate_minimum_support(planned_support)
        if not planned_support_validation["valid"]:
            blockers.append("planned independent support is insufficient")
        recovery_rows = [
            row for row in protocols if row.split == "recovery_test"
        ]
        if not recovery_rows or not all(
            bool(row.metadata.get("pilot_validated"))
            and bool(row.metadata.get("recovery_probe"))
            for row in recovery_rows
        ):
            blockers.append("recovery split is not pilot-validated")
        report = {
            "schema_version": TARGETED_DATASET_SCHEMA_VERSION,
            "valid": not blockers,
            "blockers": blockers,
            "protocol_plan_sha256": self._protocol_plan_sha256(protocols),
            "trajectory_count": len(protocols),
            "transition_count": transition_count,
            "snapshot_count": len(snapshot_ids),
            "required_splits": sorted(required_splits),
            "observed_splits": sorted(observed_splits),
            "seed_split_leaks": leaking_seeds,
            "snapshot_split_leaks": leaking_snapshots,
            "heldout_branches": sorted(heldout_branches),
            "planned_support": planned_support,
            "planned_support_validation": planned_support_validation,
            "pilot_report": dict(pilot_report),
        }
        self.targeted_preflight_report = report
        write_json(self.output_dir / "targeted_preflight_report.json", report)
        return report
