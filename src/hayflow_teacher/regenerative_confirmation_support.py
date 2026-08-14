"""Outcome-blind near-regenerative confirmation shard for HayFlow 05j-i."""

from __future__ import annotations

import hashlib
import json
import time
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np

from ..hayflow_data import InputAction, ProtocolTrajectory, validate_hdf5_store, write_json
from ..hayflow_data.targeted_protocols import action_schedule_from_json
from .audit import git_commit, sha256_file
from .audit_runtime import PINNED_TEACHER_COMMIT
from .bap_support_topup import (
    BapValidationSupportTopupSession,
    _artifact_records,
    _sha256_with_progress,
)
from .diagnostic_dataset import DiagnosticDatasetSession
from .diagnostic_dataset_v1 import canonical_json_sha256


EXPECTED_05JH_ARCHIVE_SHA256 = (
    "b51137b6f73656f9d637a9a1948dc40ac2795e1c8c480148caf5d1831e0d228c"
)
EXPECTED_05JH_INDEX_SHA256 = (
    "6336704675d1c69a71c0fa55db9d61d85bd6679c9f27a40963489ee3d9b93090"
)
EXPECTED_05JH_FINAL_SHA256 = (
    "8ea2abf9ee0366b8c2341e0602613f5b4409fa832481e9e8bd8371430a9c8e5a"
)
REGENERATIVE_CONFIRMATION_SCHEMA_VERSION = "05j-i-regenerative-confirmation-v1"


def _json_value(value: Any, default: Any) -> Any:
    if value is None:
        return default
    if isinstance(value, float) and np.isnan(value):
        return default
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return default
    return value


def _canonical_schedule(value: Any) -> str:
    schedule = _json_value(value, {})
    return json.dumps(schedule, sort_keys=True, separators=(",", ":"))


def _historical_train_eligible(value: Any) -> bool:
    """Missing Parquet values mean ordinary train-eligible pilot rows."""

    if value is None:
        return True
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"", "null", "none", "nan", "<na>"}:
            return True
        return normalized not in {"false", "0", "no"}
    if isinstance(value, float) and np.isnan(value):
        return True
    return bool(value)


@dataclass(frozen=True)
class RegenerativeConfirmationConfig:
    pair_count: int = 24
    minimum_near_pair_count: int = 18
    pilot_seed_count: int = 3
    pilot_candidate_limit: int = 24
    selected_template_count: int = 4
    episode_duration_ms: int = 12
    post_branch_ms: int = 6
    low_arm_keep_fraction: float = 0.5
    near_lower_mv: float = -45.0
    near_upper_mv: float = -20.0
    pilot_source_lower_mv: float = -100.0
    pilot_source_upper_mv: float = 80.0
    pilot_required_near_fraction: float = 1.0
    minimum_teacher_distance_mv: float = 0.01
    seed_gap: int = 20_000
    conditioning_ms: int = 4

    def validate(self) -> None:
        if self.pair_count < self.minimum_near_pair_count:
            raise ValueError("pair_count is below the near-regenerative minimum")
        if self.pilot_seed_count < 2:
            raise ValueError("at least two independent pilot seeds are required")
        if not 0 < self.selected_template_count <= self.pilot_candidate_limit:
            raise ValueError("selected_template_count is invalid")
        if self.episode_duration_ms <= self.post_branch_ms + 1:
            raise ValueError("episode duration leaves no room for the branch prefix")
        if not 0 < self.low_arm_keep_fraction < 1:
            raise ValueError("low_arm_keep_fraction must be inside (0, 1)")
        if not self.near_lower_mv < self.near_upper_mv:
            raise ValueError("near-regenerative voltage bounds are reversed")
        if not self.pilot_source_lower_mv < self.pilot_source_upper_mv:
            raise ValueError("pilot source voltage bounds are reversed")
        if not 0 < self.pilot_required_near_fraction <= 1:
            raise ValueError("pilot_required_near_fraction must be inside (0, 1]")
        if min(
            self.minimum_teacher_distance_mv,
            self.seed_gap,
            self.conditioning_ms,
        ) <= 0:
            raise ValueError("positive configuration values must be positive")

    @classmethod
    def from_mapping(
        cls, values: Mapping[str, Any]
    ) -> "RegenerativeConfirmationConfig":
        result = cls(**dict(values))
        result.validate()
        return result


def discover_pilot_templates(
    rows: Sequence[Mapping[str, Any]],
    config: RegenerativeConfirmationConfig,
) -> List[Dict[str, Any]]:
    """Rank old pilot schedules without using any new acquisition outcome."""

    config.validate()
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for raw in rows:
        row = dict(raw)
        family = str(row.get("family", ""))
        if family not in {"targeted_nmda", "targeted_calcium"}:
            continue
        if not _historical_train_eligible(row.get("train_eligible")):
            continue
        peak = row.get("event_probe_peak_voltage_mv")
        if peak is None or not np.isfinite(float(peak)):
            continue
        if not config.pilot_source_lower_mv <= float(peak) <= config.pilot_source_upper_mv:
            continue
        schedule = _json_value(row.get("input_schedule"), {})
        if not schedule:
            continue
        actions = action_schedule_from_json(schedule)
        if not actions or any(
            action.kind != "synaptic_event" or abs(action.weight_multiplier - 1.0) > 1e-12
            for step_actions in actions.values()
            for action in step_actions
        ):
            continue
        groups.setdefault(str(row["candidate_id"]), []).append(row)

    target = 0.5 * (config.near_lower_mv + config.near_upper_mv)
    templates = []
    for candidate_id, trials in groups.items():
        seeds = sorted({int(row["seed"]) for row in trials})
        if len(seeds) < config.pilot_seed_count:
            continue
        schedules = {_canonical_schedule(row.get("input_schedule")) for row in trials}
        if len(schedules) != 1:
            continue
        peaks = np.asarray(
            [float(row["event_probe_peak_voltage_mv"]) for row in trials], dtype=float
        )
        source = min(trials, key=lambda row: int(row["seed"]))
        schedule = _json_value(source["input_schedule"], {})
        templates.append(
            {
                "candidate_id": candidate_id,
                "family": str(source["family"]),
                "branch_id": str(
                    source.get(
                        "branch_id",
                        f"segment-{int(source.get('event_probe_segment_id', -1))}",
                    )
                ),
                "event_probe_segment_id": int(source.get("event_probe_segment_id", -1)),
                "event_probe_region": str(source.get("event_probe_region", "unknown")),
                "selected_synapse_ids": list(
                    map(int, _json_value(source.get("selected_synapse_ids"), ()))
                ),
                "input_schedule": schedule,
                "source_seed_count": len(seeds),
                "source_peak_min_mv": float(np.min(peaks)),
                "source_peak_median_mv": float(np.median(peaks)),
                "source_peak_max_mv": float(np.max(peaks)),
                "source_target_distance_mv": float(abs(np.median(peaks) - target)),
            }
        )
    templates.sort(
        key=lambda row: (
            row["source_target_distance_mv"],
            row["family"],
            row["candidate_id"],
        )
    )
    # Round-robin family diversity before filling by proximity to the target band.
    selected: List[Dict[str, Any]] = []
    for family in ("targeted_nmda", "targeted_calcium"):
        selected.extend([row for row in templates if row["family"] == family][:2])
    used = {row["candidate_id"] for row in selected}
    selected.extend(row for row in templates if row["candidate_id"] not in used)
    return selected[: config.pilot_candidate_limit]


def low_arm_actions(
    actions: Sequence[InputAction], keep_fraction: float
) -> Tuple[InputAction, ...]:
    """Retain a deterministic canonical-synapse subset without weight scaling."""

    actions = tuple(actions)
    if not actions:
        raise ValueError("branch action set is empty")
    keep = max(0, min(len(actions) - 1, int(np.floor(len(actions) * keep_fraction))))
    if keep == 0:
        return ()
    ranked = sorted(
        actions,
        key=lambda action: hashlib.sha256(
            f"{action.synapse_id}:{action.offset_ms:.12g}".encode()
        ).hexdigest(),
    )
    retained = set((action.synapse_id, action.offset_ms) for action in ranked[:keep])
    return tuple(
        action
        for action in actions
        if (action.synapse_id, action.offset_ms) in retained
    )


class RegenerativeConfirmationSupportSession(BapValidationSupportTopupSession):
    """Generate a separate validation-only shard of causal near-regime pairs."""

    def __init__(
        self,
        *args: Any,
        artifact_05jh_source: Path,
        confirmation_config: RegenerativeConfirmationConfig,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        confirmation_config.validate()
        self.artifact_05jh_source = Path(artifact_05jh_source).resolve()
        self.confirmation_config = confirmation_config
        self.artifact_05jh_contract: Dict[str, Any] = {}
        self.boundary_pilot_report: Dict[str, Any] = {}
        self.confirmation_plan: Dict[str, Any] = {}

    @staticmethod
    def _one_suffix(names: Sequence[str], suffix: str) -> str:
        matches = [name for name in names if name.endswith(suffix)]
        if len(matches) != 1:
            raise RuntimeError(f"expected one {suffix}, found {matches}")
        return matches[0]

    def verify_05jh_artifact(self) -> Dict[str, Any]:
        source = self.artifact_05jh_source
        verified = []
        if source.is_file():
            if sha256_file(source) != EXPECTED_05JH_ARCHIVE_SHA256:
                raise RuntimeError("05j-h archive SHA-256 mismatch")
            with zipfile.ZipFile(source) as archive:
                names = archive.namelist()
                index_name = self._one_suffix(names, "artifact_index.json")
                root = index_name[: -len("artifact_index.json")]
                index_bytes = archive.read(index_name)
                if hashlib.sha256(index_bytes).hexdigest() != EXPECTED_05JH_INDEX_SHA256:
                    raise RuntimeError("05j-h artifact index SHA-256 mismatch")
                index = json.loads(index_bytes)
                for row in index["artifacts"]:
                    payload = archive.read(root + str(row["path"]).replace("\\", "/"))
                    if len(payload) != int(row["size_bytes"]) or hashlib.sha256(payload).hexdigest() != str(row["sha256"]):
                        raise RuntimeError(f"05j-h indexed member mismatch: {row['path']}")
                    verified.append(str(row["path"]))
                final_bytes = archive.read(root + "final_report.json")
            source_kind = "original_zip"
            archive_hash = EXPECTED_05JH_ARCHIVE_SHA256
        else:
            indices = list(source.rglob("artifact_index.json"))
            if len(indices) != 1:
                raise RuntimeError("05j-h extracted artifact index is ambiguous")
            index_path = indices[0]
            index_bytes = index_path.read_bytes()
            if hashlib.sha256(index_bytes).hexdigest() != EXPECTED_05JH_INDEX_SHA256:
                raise RuntimeError("05j-h extracted artifact index SHA-256 mismatch")
            index = json.loads(index_bytes)
            root_path = index_path.parent
            for row in index["artifacts"]:
                path = root_path / str(row["path"])
                if not path.is_file() or path.stat().st_size != int(row["size_bytes"]) or sha256_file(path) != str(row["sha256"]):
                    raise RuntimeError(f"05j-h extracted member mismatch: {row['path']}")
                verified.append(str(row["path"]))
            final_bytes = (root_path / "final_report.json").read_bytes()
            source_kind = "kaggle_extracted_directory"
            archive_hash = None
        if hashlib.sha256(final_bytes).hexdigest() != EXPECTED_05JH_FINAL_SHA256:
            raise RuntimeError("05j-h final report SHA-256 mismatch")
        final = json.loads(final_bytes)
        blockers = []
        if final.get("diagnosis") != "EXISTING_DATASET_LACKS_INDEPENDENT_REGENERATIVE_CONFIRMATION_SUPPORT":
            blockers.append("unexpected 05j-h diagnosis")
        if final.get("candidate_model_authorized") is not False:
            blockers.append("05j-h unexpectedly authorized a candidate")
        if int(final.get("support", {}).get("stratum_counts", {}).get("near_regenerative", -1)) != 0:
            blockers.append("05j-h does not contain the registered zero-near support gap")
        contract = {
            "valid": not blockers,
            "blockers": blockers,
            "source_kind": source_kind,
            "source_path": str(source),
            "archive_sha256": archive_hash,
            "artifact_index_sha256": EXPECTED_05JH_INDEX_SHA256,
            "final_report_sha256": EXPECTED_05JH_FINAL_SHA256,
            "verified_member_count": len(verified),
            "all_indexed_members_verified": len(verified) == int(index["artifact_count"]),
            "diagnosis": final.get("diagnosis"),
        }
        if blockers:
            raise RuntimeError(f"05j-i provenance blockers: {blockers}")
        self.artifact_05jh_contract = contract
        write_json(self.output_dir / "artifact_05jh_verification.json", contract)
        return contract

    def load_source_templates(self) -> List[Dict[str, Any]]:
        path = self.base_dataset / "targeted_pilot" / "candidate_trials.parquet"
        if not path.is_file():
            raise RuntimeError("base targeted pilot candidate_trials.parquet is missing")
        frame = self.pd.read_parquet(path)
        rows = frame.to_dict("records")
        templates = discover_pilot_templates(rows, self.confirmation_config)
        dendritic_rows = [
            row for row in rows
            if str(row.get("family", "")) in {"targeted_nmda", "targeted_calcium"}
        ]
        finite_peaks = [
            float(row["event_probe_peak_voltage_mv"])
            for row in dendritic_rows
            if row.get("event_probe_peak_voltage_mv") is not None
            and np.isfinite(float(row["event_probe_peak_voltage_mv"]))
        ]
        discovery = {
            "schema_version": REGENERATIVE_CONFIRMATION_SCHEMA_VERSION,
            "table_path": str(path),
            "row_count": len(rows),
            "columns": list(map(str, frame.columns)),
            "family_counts": {
                family: sum(str(row.get("family", "")) == family for row in rows)
                for family in sorted({str(row.get("family", "")) for row in rows})
            },
            "dendritic_row_count": len(dendritic_rows),
            "dendritic_rows_with_schedule": sum(
                bool(_json_value(row.get("input_schedule"), {}))
                for row in dendritic_rows
            ),
            "finite_event_probe_peak_count": len(finite_peaks),
            "finite_event_probe_peak_range_mv": (
                [min(finite_peaks), max(finite_peaks)] if finite_peaks else []
            ),
            "source_voltage_filter_mv": [
                self.confirmation_config.pilot_source_lower_mv,
                self.confirmation_config.pilot_source_upper_mv,
            ],
            "discovered_template_count": len(templates),
        }
        write_json(self.output_dir / "source_template_discovery.json", discovery)
        if not templates:
            raise RuntimeError(
                "no canonical dendritic schedules passed source discovery; "
                f"diagnostics={discovery}"
            )
        write_json(
            self.output_dir / "source_template_candidates.json",
            {
                "schema_version": REGENERATIVE_CONFIRMATION_SCHEMA_VERSION,
                "selection_uses_only_historical_pilot": True,
                "new_acquisition_outcomes_used": False,
                "candidate_count": len(templates),
                "candidates": templates,
            },
        )
        return templates

    def _run_boundary_arm(
        self,
        template: Mapping[str, Any],
        branch_step: int,
        branch_actions: Sequence[InputAction],
        seed: int,
        arm: str,
    ) -> Dict[str, Any]:
        equilibrium_rng = json.loads(self.equilibrium_rng_path.read_text(encoding="utf-8"))
        self._restore_native_snapshot(
            self.equilibrium_snapshot_path,
            equilibrium_rng["sequences"],
            equilibrium_rng.get("random123_seed", self.seed),
        )
        self._rekey_rngs(int(seed))
        source_schedule = action_schedule_from_json(template["input_schedule"])
        trajectory = ProtocolTrajectory(
            trajectory_id=f"05ji-pilot-{template['candidate_id']}-{branch_step}-{seed}-{arm}",
            category="branching",
            protocol=f"05ji_{template['family']}",
            protocol_id=str(template["candidate_id"]),
            protocol_variant=f"historical_schedule_boundary_{arm}",
            seed=int(seed),
            duration_ms=int(branch_step) + 1,
            split="validation",
            actions_by_step={},
            stimulus_onset_step=0,
            metadata={
                "event_probe_segment_id": int(template["event_probe_segment_id"]),
                "selected_synapse_ids": list(template["selected_synapse_ids"]),
            },
        )
        for step in range(int(branch_step)):
            self._run_transition(
                -1,
                trajectory,
                step,
                list(source_schedule.get(step, ())),
                snapshot_path=None,
            )
        row = self._run_transition(
            -1, trajectory, int(branch_step), list(branch_actions), snapshot_path=None
        )
        return {
            "state_t": row["state_t"],
            "state_t_plus_1": row["state_t_plus_1"],
            "peak_mv": float(np.max(row["state_t_plus_1"]["voltage"])),
        }

    def run_boundary_template_pilot(
        self, templates: Sequence[Mapping[str, Any]]
    ) -> Dict[str, Any]:
        base_episodes = self._normalized_episodes(
            self.pd.read_parquet(self.base_dataset / "episodes.parquet")
        )
        maximum_seed = max(int(row["seed"]) for row in base_episodes)
        pilot_seeds = [
            maximum_seed + self.confirmation_config.seed_gap // 2 + index
            for index in range(self.confirmation_config.pilot_seed_count)
        ]
        cases = []
        for template in templates:
            schedule = action_schedule_from_json(template["input_schedule"])
            for branch_step in sorted(schedule):
                high = tuple(schedule[branch_step])
                if not high:
                    continue
                low = low_arm_actions(high, self.confirmation_config.low_arm_keep_fraction)
                if len(low) == len(high):
                    continue
                cases.append((template, int(branch_step), low, high))
        print(f"[HayFlow 05j-i][pilot di bordo] avvio: {len(cases) * len(pilot_seeds)} coppie", flush=True)
        rows = []
        started = time.perf_counter()
        total = max(1, len(cases) * len(pilot_seeds))
        completed = 0
        for template, branch_step, low_actions, high_actions in cases:
            for seed in pilot_seeds:
                low = self._run_boundary_arm(template, branch_step, low_actions, seed, "low")
                high = self._run_boundary_arm(template, branch_step, high_actions, seed, "high")
                state_error = max(
                    float(np.max(np.abs(low["state_t"][name] - high["state_t"][name])))
                    for name in self.state_variables
                )
                teacher_distance = float(np.sqrt(np.mean(
                    (low["state_t_plus_1"]["voltage"] - high["state_t_plus_1"]["voltage"]) ** 2
                )))
                target_peak = max(float(low["peak_mv"]), float(high["peak_mv"]))
                rows.append(
                    {
                        "candidate_id": str(template["candidate_id"]),
                        "family": str(template["family"]),
                        "branch_id": str(template["branch_id"]),
                        "event_probe_segment_id": int(template["event_probe_segment_id"]),
                        "event_probe_region": str(template["event_probe_region"]),
                        "selected_synapse_ids": list(template["selected_synapse_ids"]),
                        "input_schedule": dict(template["input_schedule"]),
                        "branch_step": int(branch_step),
                        "low_actions": [action.to_dict() for action in low_actions],
                        "high_actions": [action.to_dict() for action in high_actions],
                        "seed": int(seed),
                        "same_boundary_state_max_error": state_error,
                        "teacher_distance_mv": teacher_distance,
                        "low_peak_mv": float(low["peak_mv"]),
                        "high_peak_mv": float(high["peak_mv"]),
                        "target_peak_mv": target_peak,
                        "near_regenerative": bool(
                            self.confirmation_config.near_lower_mv
                            <= target_peak
                            < self.confirmation_config.near_upper_mv
                        ),
                    }
                )
                completed += 1
                if completed == 1 or completed == total or completed % max(1, total // 20) == 0:
                    elapsed = time.perf_counter() - started
                    eta = elapsed / completed * (total - completed)
                    print(
                        f"[HayFlow 05j-i][pilot di bordo] {completed}/{total} "
                        f"({100 * completed / total:.1f}%) ETA {eta / 60:.1f} min",
                        flush=True,
                    )

        grouped: Dict[Tuple[str, int], List[Dict[str, Any]]] = {}
        for row in rows:
            grouped.setdefault((row["candidate_id"], row["branch_step"]), []).append(row)
        eligible = []
        for (candidate_id, branch_step), trials in grouped.items():
            near_fraction = float(np.mean([row["near_regenerative"] for row in trials]))
            minimum_distance = min(float(row["teacher_distance_mv"]) for row in trials)
            maximum_state_error = max(float(row["same_boundary_state_max_error"]) for row in trials)
            source = trials[0]
            if (
                near_fraction >= self.confirmation_config.pilot_required_near_fraction
                and minimum_distance >= self.confirmation_config.minimum_teacher_distance_mv
                and maximum_state_error <= 1e-10
            ):
                eligible.append(
                    {
                        **{key: source[key] for key in (
                            "candidate_id", "family", "branch_id",
                            "event_probe_segment_id", "event_probe_region",
                            "selected_synapse_ids", "input_schedule", "branch_step",
                            "low_actions", "high_actions",
                        )},
                        "near_fraction": near_fraction,
                        "minimum_teacher_distance_mv": minimum_distance,
                        "maximum_boundary_state_error": maximum_state_error,
                        "median_target_peak_mv": float(np.median([
                            row["target_peak_mv"] for row in trials
                        ])),
                    }
                )
        center = 0.5 * (
            self.confirmation_config.near_lower_mv + self.confirmation_config.near_upper_mv
        )
        eligible.sort(key=lambda row: (
            -row["near_fraction"],
            abs(row["median_target_peak_mv"] - center),
            row["family"],
            row["candidate_id"],
            row["branch_step"],
        ))
        selected = []
        for family in ("targeted_nmda", "targeted_calcium"):
            family_rows = [row for row in eligible if row["family"] == family]
            if family_rows:
                selected.append(family_rows[0])
        selected_ids = {(row["candidate_id"], row["branch_step"]) for row in selected}
        selected.extend(
            row for row in eligible
            if (row["candidate_id"], row["branch_step"]) not in selected_ids
        )
        selected = selected[: self.confirmation_config.selected_template_count]
        report = {
            "schema_version": REGENERATIVE_CONFIRMATION_SCHEMA_VERSION,
            "valid": bool(selected),
            "pilot_seeds": pilot_seeds,
            "pilot_acquisition_seed_overlap": [],
            "trial_count": len(rows),
            "eligible_template_count": len(eligible),
            "selected_template_count": len(selected),
            "selected_templates": selected,
            "trials": rows,
            "selection_uses_only_pilot_seeds": True,
            "all_future_acquisition_episodes_must_be_retained": True,
            "canonical_weight_scaling_used": False,
        }
        self.boundary_pilot_report = report
        write_json(self.output_dir / "boundary_template_pilot.json", report)
        if not report["valid"]:
            raise RuntimeError(
                "no historical canonical schedule produced a robust near-regenerative "
                "one-step boundary on independent pilot seeds; inspect boundary_template_pilot.json"
            )
        return report

    def build_confirmation_plan(self) -> Tuple[List[ProtocolTrajectory], Dict[str, Any]]:
        if not self.boundary_pilot_report.get("valid"):
            raise RuntimeError("run_boundary_template_pilot() must pass first")
        base_episodes = self._normalized_episodes(
            self.pd.read_parquet(self.base_dataset / "episodes.parquet")
        )
        base_seeds = {int(row["seed"]) for row in base_episodes}
        start = max(base_seeds) + self.confirmation_config.seed_gap
        templates = list(self.boundary_pilot_report["selected_templates"])
        protocols: List[ProtocolTrajectory] = []
        plan_rows = []
        for pair_index in range(self.confirmation_config.pair_count):
            template = templates[pair_index % len(templates)]
            seed = start + pair_index
            branch_step = int(template["branch_step"])
            if (
                branch_step + self.confirmation_config.post_branch_ms
                >= self.confirmation_config.episode_duration_ms
            ):
                raise RuntimeError(
                    "selected branch step leaves insufficient registered post-branch follow-up"
                )
            source_schedule = action_schedule_from_json(template["input_schedule"])
            prefix = {step: actions for step, actions in source_schedule.items() if step < branch_step}
            pair_id = f"near-regenerative-confirmation-{pair_index:04d}"
            snapshot_id = f"near-regenerative-confirmation-snapshot-{pair_index:04d}"
            for arm in ("low", "high"):
                branch_actions = action_schedule_from_json(
                    {str(branch_step): template[f"{arm}_actions"]}
                )[branch_step]
                schedule = {**prefix, branch_step: branch_actions}
                trajectory_id = f"{pair_id}-{arm}"
                metadata = {
                    "episode_id": trajectory_id,
                    "snapshot_id": snapshot_id,
                    "branch_pair_id": pair_id,
                    "branching_distance": "near_regenerative_confirmation",
                    "branch_id": str(template["branch_id"]),
                    "source_candidate_id": str(template["candidate_id"]),
                    "source_family": str(template["family"]),
                    "branch_step": branch_step,
                    "branch_arm": arm,
                    "event_probe_segment_id": int(template["event_probe_segment_id"]),
                    "event_probe_region": str(template["event_probe_region"]),
                    "selected_synapse_ids": list(template["selected_synapse_ids"]),
                    "confirmation_support": True,
                    "all_episodes_retained": True,
                }
                trajectory = ProtocolTrajectory(
                    trajectory_id=trajectory_id,
                    category="branching",
                    protocol=f"05ji_{template['family']}",
                    protocol_id=f"05ji-{template['candidate_id']}-step{branch_step}",
                    protocol_variant=f"near_regenerative_causal_{arm}_arm",
                    seed=seed,
                    duration_ms=self.confirmation_config.episode_duration_ms,
                    split="validation",
                    actions_by_step=schedule,
                    event_enriched=False,
                    stimulus_onset_step=min(schedule, default=branch_step),
                    required_event_kinds=(),
                    negative_control=arm == "low",
                    snapshot_source=snapshot_id,
                    metadata=metadata,
                )
                trajectory.validate()
                protocols.append(trajectory)
            plan_rows.append(
                {
                    "branch_pair_id": pair_id,
                    "snapshot_id": snapshot_id,
                    "seed": seed,
                    "source_candidate_id": str(template["candidate_id"]),
                    "source_family": str(template["family"]),
                    "branch_step": branch_step,
                    "trajectory_ids": [f"{pair_id}-low", f"{pair_id}-high"],
                }
            )
        selected_seeds = {int(row["seed"]) for row in plan_rows}
        pilot_seeds = set(map(int, self.boundary_pilot_report["pilot_seeds"]))
        if selected_seeds & base_seeds or selected_seeds & pilot_seeds:
            raise RuntimeError("05j-i acquisition seed namespace overlaps base or pilot")
        payload = {
            "schema_version": REGENERATIVE_CONFIRMATION_SCHEMA_VERSION,
            "policy": "pilot_selected_fixed_batch_all_pairs_retained",
            "selection_was_outcome_blind_to_acquisition": True,
            "target_role": "validation_only_regenerative_confirmation",
            "pair_count": len(plan_rows),
            "episode_count": len(protocols),
            "transition_count": sum(row.duration_ms for row in protocols),
            "minimum_near_pair_count": self.confirmation_config.minimum_near_pair_count,
            "seed_start": start,
            "seed_end": start + self.confirmation_config.pair_count - 1,
            "pilot_seeds": sorted(pilot_seeds),
            "episodes": plan_rows,
            "artifact_05jh": self.artifact_05jh_contract,
            "teacher_commit": PINNED_TEACHER_COMMIT,
            "configuration": asdict(self.confirmation_config),
        }
        payload["protocol_plan_sha256"] = self._protocol_plan_sha256(protocols)
        payload["confirmation_contract_sha256"] = canonical_json_sha256({"plan": payload})
        self.confirmation_plan = payload
        self.topup_plan = payload
        self._bind_protocol_registry(protocols)
        write_json(self.output_dir / "confirmation_plan.json", payload)
        return protocols, payload

    def generate_confirmation_shard(
        self, protocols: Sequence[ProtocolTrajectory]
    ) -> Dict[str, Any]:
        protocols = list(protocols)
        if len(protocols) != 2 * self.confirmation_config.pair_count:
            raise RuntimeError("05j-i must retain both arms of every registered pair")
        persisted = json.loads(
            (self.output_dir / "confirmation_plan.json").read_text(encoding="utf-8")
        )
        if persisted != self.confirmation_plan:
            raise RuntimeError("persisted 05j-i plan changed before generation")
        required_snapshots = {str(row.metadata["snapshot_id"]) for row in protocols}
        if required_snapshots != set(self.snapshot_bank):
            raise RuntimeError("05j-i snapshot bank does not match the fixed plan")
        zero_targets = {"validation": 0}
        write_json(
            self.output_dir / "planning_budget_report.json",
            {
                "schema_version": REGENERATIVE_CONFIRMATION_SCHEMA_VERSION,
                "role": "standalone confirmation shard",
                "effective_positive_targets": zero_targets,
                "effective_hard_negative_targets": zero_targets,
                "minimum_positive_targets": zero_targets,
                "minimum_hard_negative_targets": zero_targets,
            },
        )
        self.targeted_preflight_report = {
            "valid": True,
            "protocol_plan_sha256": self.confirmation_plan["protocol_plan_sha256"],
            "policy": "fixed_batch_all_pairs_retained",
        }
        write_json(
            self.output_dir / "targeted_preflight_report.json",
            self.targeted_preflight_report,
        )
        self.release_rows = []
        self._bind_protocol_registry(protocols)
        self._collect_release_rows = True
        try:
            manifest = DiagnosticDatasetSession.generate_dataset(self, protocols)
        finally:
            self._collect_release_rows = False
        self.pd.DataFrame(self._parquet_safe_rows(self.release_rows)).to_parquet(
            self.output_dir / "release_outcomes.parquet", index=False
        )
        table_report = self._write_targeted_tables(protocols)
        manifest.update(
            {
                "schema_version": REGENERATIVE_CONFIRMATION_SCHEMA_VERSION,
                "dataset_role": "validation_only_regenerative_confirmation_support_shard",
                "compatible_base_schema_version": "1.1.2",
                "selection_policy": "fixed_batch_all_pairs_retained",
                "confirmation_plan": "confirmation_plan.json",
                "artifact_05jh_verification": "artifact_05jh_verification.json",
                "boundary_template_pilot": "boundary_template_pilot.json",
                "table_report": table_report,
                "indices": {
                    "protocols": "protocols.parquet",
                    "episodes": "episodes.parquet",
                    "transitions": "transition_index.parquet",
                    "events": "events.parquet",
                    "release_outcomes": "release_outcomes.parquet",
                    "branching_pairs": "branching_pairs.parquet",
                    "splits": "splits.json",
                },
            }
        )
        self.dataset_manifest = manifest
        write_json(self.output_dir / "dataset_manifest.json", manifest)
        self._write_artifact_index()
        return manifest

    def validate_confirmation_shard(
        self, protocols: Sequence[ProtocolTrajectory]
    ) -> Dict[str, Any]:
        protocols = list(protocols)
        self._bind_protocol_registry(protocols)
        structural = validate_hdf5_store(self.transition_path)
        transition_sha = _sha256_with_progress(
            self.transition_path, label="shard near-regenerative"
        )
        replay = self._exhaustive_sequential_replay()
        transition_index = self.pd.read_parquet(
            self.output_dir / "transition_index.parquet"
        )
        by_key = {
            (str(row.trajectory_id), int(row.step_index)): int(row.transition_id)
            for row in transition_index.itertuples(index=False)
        }
        import h5py

        pair_rows = []
        with h5py.File(self.transition_path, "r") as handle:
            for row in self.confirmation_plan["episodes"]:
                low_id, high_id = row["trajectory_ids"]
                step = int(row["branch_step"])
                low_index, high_index = by_key[(low_id, step)], by_key[(high_id, step)]
                low_t = handle["states/voltage/t"][low_index, :]
                high_t = handle["states/voltage/t"][high_index, :]
                low_t1 = handle["states/voltage/t_plus_1"][low_index, :]
                high_t1 = handle["states/voltage/t_plus_1"][high_index, :]
                peak = float(max(np.max(low_t1), np.max(high_t1)))
                pair_rows.append(
                    {
                        **dict(row),
                        "low_transition_id": low_index,
                        "high_transition_id": high_index,
                        "same_boundary_voltage_max_error_mv": float(
                            np.max(np.abs(low_t - high_t))
                        ),
                        "teacher_distance_rmse_mv": float(
                            np.sqrt(np.mean((low_t1 - high_t1) ** 2))
                        ),
                        "target_peak_mv": peak,
                        "realized_stratum": (
                            "regenerative"
                            if peak >= self.confirmation_config.near_upper_mv
                            else "near_regenerative"
                            if peak >= self.confirmation_config.near_lower_mv
                            else "subthreshold"
                        ),
                    }
                )
        self.pd.DataFrame(pair_rows).to_parquet(
            self.output_dir / "confirmation_pairs.parquet", index=False
        )
        near_count = sum(row["realized_stratum"] == "near_regenerative" for row in pair_rows)
        all_retained = len(pair_rows) == self.confirmation_config.pair_count
        pair_contract_valid = bool(
            all_retained
            and all(row["same_boundary_voltage_max_error_mv"] <= 1e-5 for row in pair_rows)
            and all(row["teacher_distance_rmse_mv"] >= self.confirmation_config.minimum_teacher_distance_mv for row in pair_rows)
        )
        support_sufficient = near_count >= self.confirmation_config.minimum_near_pair_count
        artifact_valid = bool(structural.get("valid") and replay.get("valid") and pair_contract_valid)
        diagnosis = (
            "NEAR_REGENERATIVE_CONFIRMATION_SUPPORT_ACQUIRED"
            if support_sufficient
            else "NEAR_REGENERATIVE_CONFIRMATION_SUPPORT_STILL_INSUFFICIENT"
        )
        report = {
            "schema_version": REGENERATIVE_CONFIRMATION_SCHEMA_VERSION,
            "valid": artifact_valid,
            "artifact_valid": artifact_valid,
            "scientific_support_sufficient": support_sufficient,
            "diagnosis": diagnosis,
            "teacher_commit": PINNED_TEACHER_COMMIT,
            "code_revision": git_commit(self.elm_repo),
            "transition_store_sha256": transition_sha,
            "structural": structural,
            "exhaustive_replay": replay,
            "pair_contract_valid": pair_contract_valid,
            "registered_pair_count": self.confirmation_config.pair_count,
            "retained_pair_count": len(pair_rows),
            "retained_episode_count": len(protocols),
            "all_registered_episodes_retained": all_retained and len(protocols) == 2 * len(pair_rows),
            "minimum_near_pair_count": self.confirmation_config.minimum_near_pair_count,
            "realized_stratum_counts": {
                name: sum(row["realized_stratum"] == name for row in pair_rows)
                for name in ("regenerative", "near_regenerative", "subthreshold")
            },
            "pair_rows": pair_rows,
            "artifact_05jh": self.artifact_05jh_contract,
            "methodology": {
                "pilot_and_acquisition_seeds_disjoint": True,
                "selection_was_outcome_blind_to_acquisition": True,
                "all_acquisition_pairs_retained": True,
                "canonical_netcon_weights_preserved": True,
                "weight_multiplier_sweeps_used": False,
                "causal_realized_release_logged": True,
                "heldout_inputs_extracted": False,
                "candidate_training_performed": False,
                "rollout_performed": False,
            },
            "next_step": (
                "05j_j_confirm_regenerative_state_transition_on_new_support"
                if support_sufficient
                else "05j_i_b_adaptive_near_regenerative_acquisition"
            ),
        }
        write_json(self.output_dir / "validation_report.json", report)
        write_json(self.output_dir / "final_report.json", report)
        self._write_artifact_index({"transition_dataset.h5": transition_sha})
        if not artifact_valid:
            raise RuntimeError("05j-i shard failed structural, replay, or pair validation")
        return report


__all__ = [
    "EXPECTED_05JH_ARCHIVE_SHA256",
    "EXPECTED_05JH_INDEX_SHA256",
    "EXPECTED_05JH_FINAL_SHA256",
    "REGENERATIVE_CONFIRMATION_SCHEMA_VERSION",
    "RegenerativeConfirmationConfig",
    "RegenerativeConfirmationSupportSession",
    "discover_pilot_templates",
    "low_arm_actions",
]
