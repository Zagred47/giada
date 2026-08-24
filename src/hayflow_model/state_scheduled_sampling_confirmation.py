"""06b-g: independent confirmation and STATE scheduled-sampling refinement."""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from . import atomic_state_dynamics_playground as atomic
from .recursive_joint_repair_matrix import (
    RecursiveJointRepairConfig,
    RecursiveJointRepairMatrix,
)


EXPECTED_06BF_ARCHIVE_SHA256 = (
    "ad47fb5f16f4f495cc0dd4641448a6cca565833c1473a87f29b5e1ff81859d43"
)
EXPECTED_06BF_INDEX_SHA256 = (
    "0c0103784aa1da435e43252d29d5b10d65c7e6baf8e94075f4cc0babe03fb799"
)
EXPECTED_06BF_FINAL_SHA256 = (
    "a9b7c1459c08b95064c41f1215717f5ed205180e742a6edc021f51a740ff0a80"
)

SCHEDULED_ARMS: Dict[str, str] = {
    "scalar_continue": "full_feedback",
    "state_linear_curriculum": "state_linear_curriculum",
    "joint_linear_curriculum": "joint_linear_curriculum",
    "state_fixed_25": "state_fixed_probability",
    "shuffled_continue": "shuffled_voltage_to_STATE",
}


def verified_06bf_artifact_root(
    source: Path, cache_dir: Path
) -> Tuple[Path, Dict[str, Any]]:
    """Verify the exact registered 06b-f artifact member by member."""

    source = Path(source).expanduser().resolve()
    cache_dir = Path(cache_dir).resolve()
    if not source.exists():
        raise FileNotFoundError(source)
    archive_hash = "extracted-directory"
    if source.is_file():
        if source.suffix.lower() != ".zip":
            raise RuntimeError("06b-f source must be a ZIP or extracted directory")
        archive_hash = atomic._sha256_file(source)
        stamp = {
            "path": str(source),
            "size": source.stat().st_size,
            "mtime_ns": source.stat().st_mtime_ns,
        }
        marker = cache_dir / ".source.json"
        if not marker.is_file() or json.loads(marker.read_text()) != stamp:
            if cache_dir.exists():
                import shutil

                shutil.rmtree(cache_dir)
            cache_dir.mkdir(parents=True)
            atomic._safe_extract(source, cache_dir)
            marker.write_text(json.dumps(stamp, sort_keys=True), encoding="utf-8")
        search_root = cache_dir
    else:
        search_root = source
    matches = [
        path.parent
        for path in search_root.rglob("artifact_index.json")
        if atomic._sha256_file(path) == EXPECTED_06BF_INDEX_SHA256
    ]
    if len(matches) != 1:
        raise RuntimeError(f"expected one exact 06b-f artifact; found {len(matches)}")
    root = matches[0]
    index = json.loads((root / "artifact_index.json").read_text(encoding="utf-8"))
    failures = []
    for row in index.get("artifacts", []):
        member = root / str(row["path"])
        if (
            not member.is_file()
            or member.stat().st_size != int(row["size_bytes"])
            or atomic._sha256_file(member) != str(row["sha256"])
        ):
            failures.append(str(row["path"]))
    if failures:
        raise RuntimeError(f"06b-f indexed member verification failed: {failures}")
    final_path = root / "final_report.json"
    if atomic._sha256_file(final_path) != EXPECTED_06BF_FINAL_SHA256:
        raise RuntimeError("06b-f final report SHA-256 mismatch")
    final = json.loads(final_path.read_text(encoding="utf-8"))
    if (
        final.get("diagnosis") != "VOLTAGE_REPAIRED_STATE_EXPOSURE_REMAINS"
        or final.get("coupled_06c_canary_authorized") is not False
        or final.get("next_step") != "state_scheduled_sampling_refinement"
    ):
        raise RuntimeError("06b-f result does not authorize 06b-g")
    if source.is_file() and archive_hash != EXPECTED_06BF_ARCHIVE_SHA256:
        archive_hash = "kaggle-repacked"
    return root, {
        "valid": True,
        "source_kind": "zip" if source.is_file() else "directory",
        "archive_sha256": archive_hash,
        "artifact_index_sha256": EXPECTED_06BF_INDEX_SHA256,
        "final_report_sha256": EXPECTED_06BF_FINAL_SHA256,
        "indexed_member_count": len(index.get("artifacts", [])),
        "diagnosis": final["diagnosis"],
    }


@dataclass(frozen=True)
class StateScheduledSamplingConfig(RecursiveJointRepairConfig):
    scheduled_training_steps: int = 400
    scheduled_checkpoints: Tuple[int, ...] = (0, 100, 200, 400)
    scheduled_unroll_horizon_ms: int = 8
    scheduled_batch_window_count: int = 4
    scheduled_learning_rate: float = 0.00015
    scheduled_weight_decay: float = 0.00001
    scheduled_gradient_clip_norm: float = 1.0
    confirmation_components_per_regime: int = 1
    confirmation_window_count: int = 16
    curriculum_initial_teacher_probability: float = 0.50
    curriculum_decay_steps: int = 300
    fixed_STATE_teacher_probability: float = 0.25
    minimum_confirmation_STATE_error_reduction_fraction: float = 0.02
    minimum_confirmation_voltage_error_reduction_fraction: float = 0.10
    minimum_curriculum_STATE_gain_fraction: float = 0.02
    maximum_curriculum_voltage_degradation_fraction: float = 0.02
    minimum_continuation_scaling_fraction: float = 0.02
    maximum_confirmation_one_step_degradation_fraction: float = 0.02
    minimum_confirmation_causal_specificity_fraction: float = 0.01

    def validate(self) -> None:
        super().validate()
        checkpoints = tuple(map(int, self.scheduled_checkpoints))
        if (
            checkpoints[0] != 0
            or checkpoints[-1] != self.scheduled_training_steps
            or tuple(sorted(set(checkpoints))) != checkpoints
        ):
            raise ValueError("06b-g checkpoints must span one continuation")
        if self.scheduled_unroll_horizon_ms != max(self.rollout_horizons_ms):
            raise ValueError("06b-g must train directly at the maximum horizon")
        if not 0 < self.curriculum_initial_teacher_probability < 1:
            raise ValueError("06b-g initial curriculum probability is invalid")
        if not 0 < self.fixed_STATE_teacher_probability < 1:
            raise ValueError("06b-g fixed STATE probability is invalid")
        if self.curriculum_decay_steps > self.scheduled_training_steps:
            raise ValueError("06b-g curriculum decay exceeds continuation budget")
        if min(
            self.scheduled_training_steps,
            self.scheduled_batch_window_count,
            self.confirmation_components_per_regime,
            self.confirmation_window_count,
            self.curriculum_decay_steps,
        ) <= 0:
            raise ValueError("06b-g positive dimensions are invalid")
        if self.scheduled_batch_window_count > self.repair_fit_window_count:
            raise ValueError("06b-g batch exceeds the fit window pool")

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "StateScheduledSamplingConfig":
        payload = dict(values)
        for name in (
            "rollout_horizons_ms",
            "voltage_path_sample_indices",
            "pilot_seeds",
            "scaling_checkpoints",
            "repair_checkpoints",
            "scheduled_checkpoints",
        ):
            if name in payload:
                payload[name] = tuple(map(int, payload[name]))
        result = cls(**payload)
        result.validate()
        return result


class StateScheduledSamplingConfirmation(RecursiveJointRepairMatrix):
    """Continue the registered scalar checkpoint under synchronized curricula."""

    config: StateScheduledSamplingConfig

    def __init__(
        self,
        bundle: Any,
        output_dir: Path,
        config: StateScheduledSamplingConfig,
        artifact_05t_source: Path,
        artifact_06b_source: Path,
        artifact_06bb_source: Path,
        artifact_06bc_source: Path,
        artifact_06bd_source: Path,
        artifact_06be_source: Path,
        artifact_06bf_source: Path,
        *,
        code_revision: str,
    ) -> None:
        super().__init__(
            bundle,
            output_dir,
            config,
            artifact_05t_source,
            artifact_06b_source,
            artifact_06bb_source,
            artifact_06bc_source,
            artifact_06bd_source,
            artifact_06be_source,
            code_revision=code_revision,
        )
        self.artifact_06bf_source = Path(artifact_06bf_source)
        self.source_models: Dict[Tuple[str, int], Tuple[Any, Any]] = {}
        self.scheduled_models: Dict[Tuple[str, int], Tuple[Any, Any]] = {}
        self.scheduled_states: Dict[Tuple[str, int, int], Dict[str, Any]] = {}

    def _build_independent_confirmation_role(self) -> Dict[str, Any]:
        grouped = atomic.disjoint_episode_components_by_regime(
            self.store.episode_rows, role_seed=self.config.role_seed
        )
        consumed = (
            self.config.fit_components_per_regime
            + self.config.calibration_components_per_regime
            + self.config.development_components_per_regime
        )
        rows: List[Dict[str, Any]] = []
        availability = {}
        for regime, components in sorted(grouped.items()):
            availability[regime] = len(components)
            stop = consumed + self.config.confirmation_components_per_regime
            if len(components) < stop:
                raise RuntimeError(
                    f"06b-g regime {regime!r} has {len(components)} components; "
                    f"{stop} required for an unused confirmation role"
                )
            for component in components[consumed:stop]:
                for source in component:
                    row = dict(source)
                    row["06bg_role"] = "independent_confirmation"
                    row["06bg_regime"] = regime
                    rows.append(row)
        previous = {
            str(row["trajectory_id"])
            for role_rows in self.roles.values()
            for row in role_rows
        }
        confirmation = {str(row["trajectory_id"]) for row in rows}
        overlap = sorted(previous & confirmation)
        if overlap:
            raise RuntimeError(f"06b-g confirmation trajectory leak: {overlap}")
        if any(str(row.get("split")) != "train" for row in rows):
            raise RuntimeError("06b-g confirmation leaked outside train")
        self.roles["confirmation"] = rows
        return {
            "valid": True,
            "role_seed": self.config.role_seed,
            "consumed_component_prefix_per_regime": consumed,
            "confirmation_components_per_regime": self.config.confirmation_components_per_regime,
            "available_components_by_regime": availability,
            "confirmation_episode_count": len(rows),
            "confirmation_trajectory_count": len(confirmation),
            "previous_role_overlap": overlap,
            "split": "train",
        }

    def _load_source_06bf_models(self, root: Path, device: Any) -> Dict[str, str]:
        hashes = {}
        for seed in self.config.pilot_seeds:
            for source_arm in ("full_feedback_scalar", "teacherV_teacherS"):
                name = f"repair_{source_arm}_seed{seed}_step600.pt"
                path = root / name
                checkpoint = atomic.torch.load(
                    path, map_location=device, weights_only=False
                )
                if (
                    str(checkpoint.get("arm")) != source_arm
                    or int(checkpoint.get("seed", -1)) != seed
                    or int(checkpoint.get("budget", -1)) != 600
                ):
                    raise RuntimeError(f"06b-g source checkpoint mismatch: {name}")
                pair = self._new_pair(seed, device)
                pair[0].load_state_dict(copy.deepcopy(checkpoint["bridge_state_dict"]))
                pair[1].load_state_dict(copy.deepcopy(checkpoint["STATE_state_dict"]))
                for model in pair:
                    model.eval()
                    for parameter in model.parameters():
                        parameter.requires_grad_(False)
                self.source_models[(source_arm, seed)] = pair
                hashes[name] = atomic._sha256_file(path)
        return hashes

    def prepare_scheduled_sampling_confirmation(self) -> Dict[str, Any]:
        base = self.prepare_recursive_joint_repair()
        source_root, source = verified_06bf_artifact_root(
            self.artifact_06bf_source,
            self.output_dir.parent / ".06bg_artifact_cache" / "06bf",
        )
        role = self._build_independent_confirmation_role()
        self._materialize_window_role(
            "fit",
            self.config.repair_fit_window_count,
            self.config.scheduled_unroll_horizon_ms,
        )
        self._materialize_window_role(
            "confirmation",
            self.config.confirmation_window_count,
            max(self.config.rollout_horizons_ms),
        )
        device = next(iter(self.frozen_state_models.values())).proposal.weight.device
        checkpoint_hashes = self._load_source_06bf_models(source_root, device)
        report = {
            **base,
            "schema_version": "06b-g-scheduled-sampling-contract-v1",
            "experiment": "state_scheduled_sampling_confirmation",
            "source_06bf": source,
            "source_checkpoint_sha256": checkpoint_hashes,
            "source_checkpoint_arm": "full_feedback_scalar_step600",
            "source_reference_arm": "teacherV_teacherS_step600",
            "continuation_arms": dict(SCHEDULED_ARMS),
            "primary_arm": "state_linear_curriculum",
            "fallback_arm": "scalar_continue",
            "hierarchical_decision_rule_preregistered": True,
            "independent_confirmation_role": role,
            "confirmation_window_count": self.config.confirmation_window_count,
            "continuation_horizon_ms": self.config.scheduled_unroll_horizon_ms,
            "continuation_checkpoints": list(self.config.scheduled_checkpoints),
            "same_source_checkpoint_within_seed": True,
            "source_optimizer_state_available": False,
            "optimizer_restart_is_identical_across_arms": True,
            "continuation_claim": "weight continuation with a registered optimizer restart",
            "same_window_stream_within_seed": True,
            "same_teacher_forcing_uniform_draws_within_seed": True,
            "confirmation_used_during_training": False,
            "confirmation_used_for_checkpoint_selection": False,
            "development_used_for_selection": False,
            "validation_state_accessed": False,
            "test_state_accessed": False,
            "full_training_authorized": False,
        }
        for stale_field in (
            "training_stage_at_contract_write",
            "training_planned",
            "feedback_boundary_during_training",
            "joint_objective_backpropagates_through_frozen_state_updater",
            "training_horizon_ms",
        ):
            report.pop(stale_field, None)
        report.update(
            {
                "training_stage_at_contract_write": "not_started",
                "continuation_training_planned": True,
                "feedback_boundary_during_training": "schedule_specific",
                "training_horizon_ms": self.config.scheduled_unroll_horizon_ms,
                "joint_objective_backpropagates_through_trainable_STATE_updater": True,
            }
        )
        atomic._write_json(self.output_dir / "scheduled_sampling_contract.json", report)
        return report

    def _new_continuation_pair(self, seed: int, device: Any) -> Tuple[Any, Any]:
        source = self.source_models[("full_feedback_scalar", seed)]
        pair = self._new_pair(seed, device)
        pair[0].load_state_dict(copy.deepcopy(source[0].state_dict()))
        pair[1].load_state_dict(copy.deepcopy(source[1].state_dict()))
        for model in pair:
            model.train()
            for parameter in model.parameters():
                parameter.requires_grad_(True)
        return pair

    def _teacher_probability(self, arm: str, training_step: int) -> Tuple[float, float]:
        schedule = SCHEDULED_ARMS[arm]
        if schedule in ("full_feedback", "shuffled_voltage_to_STATE"):
            return 0.0, 0.0
        if schedule == "state_fixed_probability":
            return self.config.fixed_STATE_teacher_probability, 0.0
        fraction = max(
            0.0,
            1.0 - float(training_step) / float(self.config.curriculum_decay_steps),
        )
        probability = self.config.curriculum_initial_teacher_probability * fraction
        if schedule == "state_linear_curriculum":
            return probability, 0.0
        if schedule == "joint_linear_curriculum":
            return probability, probability
        raise ValueError(arm)

    def _scheduled_unroll_objectives(
        self,
        bridge: Any,
        state_model: Any,
        batch: Mapping[str, Any],
        arm: str,
        training_step: int,
        state_uniform: Any,
        voltage_uniform: Any,
    ) -> Tuple[Any, Any]:
        current_state = batch["state_t"][:, 0]
        current_voltage = batch["voltage_t"][:, 0]
        state_losses = []
        voltage_losses = []
        state_probability, voltage_probability = self._teacher_probability(
            arm, training_step
        )
        state_center = atomic.torch.as_tensor(
            self.statistics["state_center"], device=current_state.device
        )
        state_scale = atomic.torch.as_tensor(
            self.statistics["state_scale"], device=current_state.device
        )
        delta_scale = atomic.torch.as_tensor(
            self.statistics["delta_scale"], device=current_state.device
        )
        for step in range(self.config.scheduled_unroll_horizon_ms):
            if step == 0:
                state_input, voltage_input = current_state, current_voltage
            else:
                state_mask = (state_uniform[:, step] < state_probability).to(
                    current_state.dtype
                )[:, None]
                voltage_mask = (voltage_uniform[:, step] < voltage_probability).to(
                    current_voltage.dtype
                )[:, None]
                state_input = (
                    state_mask * batch["state_t"][:, step]
                    + (1.0 - state_mask) * current_state
                )
                voltage_input = (
                    voltage_mask * batch["voltage_t"][:, step]
                    + (1.0 - voltage_mask) * current_voltage
                )
            context = atomic.torch.cat(
                (batch["drive"][:, step], batch["held_ions"]), dim=-1
            )
            normalized = (state_input - state_center) / state_scale
            voltage_delta = self._bridge_forward(
                bridge, normalized, voltage_input, context
            )
            current_voltage = voltage_input + voltage_delta
            state_path = (
                atomic.torch.roll(voltage_delta, shifts=1, dims=0)
                if SCHEDULED_ARMS[arm] == "shuffled_voltage_to_STATE"
                else voltage_delta
            )
            state_delta = self._state_forward(
                state_model, normalized, voltage_input, state_path, context
            )
            current_state = state_input + state_delta * delta_scale
            voltage_error = (
                current_voltage - batch["voltage_t1"][:, step]
            ) / self.config.bridge_voltage_scale_mv
            voltage_active = (
                (batch["voltage_t1"][:, step] - batch["voltage_t"][:, step]).abs()
                >= self.config.bridge_active_delta_threshold_mv
            ).float()
            voltage_loss = atomic.torch.mean(
                (1.0 + self.config.bridge_active_weight * voltage_active)
                * atomic.torch_functional.smooth_l1_loss(
                    voltage_error,
                    atomic.torch.zeros_like(voltage_error),
                    reduction="none",
                )
            )
            high = atomic.torch.relu(
                (current_voltage - self.config.physical_voltage_maximum_mv)
                / self.config.bridge_voltage_scale_mv
            )
            low = atomic.torch.relu(
                (self.config.physical_voltage_minimum_mv - current_voltage)
                / self.config.bridge_voltage_scale_mv
            )
            physical = atomic.torch.mean(high * high + low * low)
            drift = atomic.torch.mean(atomic.torch.mean(voltage_error, dim=1) ** 2)
            voltage_losses.append(
                self.config.repair_voltage_loss_weight * voltage_loss
                + self.config.repair_physical_penalty_weight * physical
                + self.config.repair_drift_penalty_weight * drift
            )
            state_error = (
                current_state - batch["state_t1"][:, step]
            ) / state_scale
            target_delta = (
                batch["state_t1"][:, step] - batch["state_t"][:, step]
            ) / delta_scale
            state_active = (
                target_delta.abs() >= self.config.active_delta_threshold
            ).float()
            state_losses.append(
                self.config.repair_state_loss_weight
                * atomic.torch.mean(
                    (1.0 + self.config.active_delta_weight * state_active)
                    * atomic.torch_functional.smooth_l1_loss(
                        state_error,
                        atomic.torch.zeros_like(state_error),
                        reduction="none",
                    )
                )
            )
        return (
            atomic.torch.stack(voltage_losses).mean(),
            atomic.torch.stack(state_losses).mean(),
        )

    def _save_scheduled_checkpoint(
        self, arm: str, seed: int, budget: int, pair: Tuple[Any, Any]
    ) -> Dict[str, Any]:
        payload = {
            "bridge_state_dict": {
                name: value.detach().cpu().clone()
                for name, value in pair[0].state_dict().items()
            },
            "STATE_state_dict": {
                name: value.detach().cpu().clone()
                for name, value in pair[1].state_dict().items()
            },
            "arm": arm,
            "seed": seed,
            "budget": budget,
            "source_arm": "full_feedback_scalar_step600",
            "configuration": asdict(self.config),
        }
        self.scheduled_states[(arm, seed, budget)] = payload
        path = self.output_dir / f"scheduled_{arm}_seed{seed}_step{budget}.pt"
        atomic.torch.save(payload, path)
        return {
            "budget": budget,
            "checkpoint": path.name,
            "checkpoint_sha256": atomic._sha256_file(path),
        }

    def train_synchronized_scheduled_matrix(self) -> Dict[str, Any]:
        device = next(iter(self.frozen_state_models.values())).proposal.weight.device
        reports = {}
        for seed in self.config.pilot_seeds:
            pairs = {
                arm: self._new_continuation_pair(seed, device)
                for arm in SCHEDULED_ARMS
            }
            optimizers = {
                arm: atomic.torch.optim.AdamW(
                    list(pair[0].parameters()) + list(pair[1].parameters()),
                    lr=self.config.scheduled_learning_rate,
                    weight_decay=self.config.scheduled_weight_decay,
                )
                for arm, pair in pairs.items()
            }
            seed_report = {arm: [] for arm in SCHEDULED_ARMS}
            for arm, pair in pairs.items():
                seed_report[arm].append(
                    self._save_scheduled_checkpoint(arm, seed, 0, pair)
                )
            print(
                f"[HayFlow 06b-g][checkpoint seed={seed}] 0/"
                f"{self.config.scheduled_training_steps}: {len(SCHEDULED_ARMS)} arms",
                flush=True,
            )
            rng = np.random.default_rng(seed + 670000)
            progress = atomic._CompactProgress(
                f"06b-g scheduled seed={seed}",
                self.config.scheduled_training_steps,
                max(1, self.config.scheduled_training_steps // 20),
            )
            for step in range(1, self.config.scheduled_training_steps + 1):
                rows = rng.choice(
                    len(self.window_data["fit"]["indices"]),
                    size=self.config.scheduled_batch_window_count,
                    replace=False,
                )
                state_uniform = rng.random(
                    (
                        self.config.scheduled_batch_window_count,
                        self.config.scheduled_unroll_horizon_ms,
                    )
                ).astype(np.float32)
                voltage_uniform = rng.random(state_uniform.shape).astype(np.float32)
                batch = self._batch_tensors("fit", rows, device)
                state_draw = atomic.torch.as_tensor(state_uniform, device=device)
                voltage_draw = atomic.torch.as_tensor(voltage_uniform, device=device)
                losses = []
                for arm, pair in pairs.items():
                    voltage_loss, state_loss = self._scheduled_unroll_objectives(
                        pair[0],
                        pair[1],
                        batch,
                        arm,
                        step,
                        state_draw,
                        voltage_draw,
                    )
                    optimizer = optimizers[arm]
                    optimizer.zero_grad(set_to_none=True)
                    total = voltage_loss + state_loss
                    total.backward()
                    atomic.torch.nn.utils.clip_grad_norm_(
                        list(pair[0].parameters()) + list(pair[1].parameters()),
                        self.config.scheduled_gradient_clip_norm,
                    )
                    optimizer.step()
                    losses.append(float(total.detach().cpu()))
                if step in self.config.scheduled_checkpoints:
                    for arm, pair in pairs.items():
                        seed_report[arm].append(
                            self._save_scheduled_checkpoint(arm, seed, step, pair)
                        )
                    print(
                        f"[HayFlow 06b-g][checkpoint seed={seed}] {step}/"
                        f"{self.config.scheduled_training_steps}: "
                        f"{len(SCHEDULED_ARMS)} arms",
                        flush=True,
                    )
                progress.update(step, f"median_loss={float(np.median(losses)):.4g}")
            for arm, pair in pairs.items():
                pair[0].eval()
                pair[1].eval()
                self.scheduled_models[(arm, seed)] = pair
            reports[str(seed)] = seed_report
        report = {
            "schema_version": "06b-g-synchronized-training-v1",
            "valid": all(
                len(rows[arm]) == len(self.config.scheduled_checkpoints)
                for rows in reports.values()
                for arm in SCHEDULED_ARMS
            ),
            "device": str(device),
            "same_source_checkpoint_within_seed": True,
            "source_optimizer_state_available": False,
            "optimizer_restart_is_identical_across_arms": True,
            "same_window_stream_within_seed": True,
            "same_teacher_forcing_uniform_draws_within_seed": True,
            "confirmation_used_during_training": False,
            "single_trajectory_per_arm_supplies_all_budgets": True,
            "reports": reports,
        }
        atomic._write_json(self.output_dir / "scheduled_training.json", report)
        return report

    def _pair_from_scheduled_checkpoint(
        self, arm: str, seed: int, budget: int, device: Any
    ) -> Tuple[Any, Any]:
        payload = self.scheduled_states[(arm, seed, budget)]
        pair = self._new_pair(seed, device)
        pair[0].load_state_dict(copy.deepcopy(payload["bridge_state_dict"]))
        pair[1].load_state_dict(copy.deepcopy(payload["STATE_state_dict"]))
        return pair

    def evaluate_independent_confirmation(self) -> Dict[str, Any]:
        device = next(iter(self.frozen_state_models.values())).proposal.weight.device
        per_seed = {}
        total = len(self.config.pilot_seeds) * (
            2 + len(SCHEDULED_ARMS) * len(self.config.scheduled_checkpoints)
        )
        progress = atomic._CompactProgress(
            "06b-g independent confirmation", total, max(1, total // 20)
        )
        completed = 0
        for seed in self.config.pilot_seeds:
            source = {}
            for arm in ("full_feedback_scalar", "teacherV_teacherS"):
                pair = self.source_models[(arm, seed)]
                source[arm] = self._evaluate_pair(
                    pair[0], pair[1], "confirmation", device
                )
                completed += 1
                progress.update(completed, f"seed={seed} source={arm}")
            scheduled = {}
            for arm in SCHEDULED_ARMS:
                scheduled[arm] = {}
                for budget in self.config.scheduled_checkpoints:
                    pair = self._pair_from_scheduled_checkpoint(
                        arm, seed, budget, device
                    )
                    scheduled[arm][str(budget)] = self._evaluate_pair(
                        pair[0], pair[1], "confirmation", device
                    )
                    completed += 1
                    progress.update(
                        completed, f"seed={seed} arm={arm} step={budget}"
                    )
            per_seed[str(seed)] = {"source": source, "scheduled": scheduled}
        all_metrics = []
        for seed in per_seed.values():
            for arm in seed["source"].values():
                for boundary in arm.values():
                    all_metrics.extend(boundary.values())
            for arm in seed["scheduled"].values():
                for budget in arm.values():
                    for boundary in budget.values():
                        all_metrics.extend(boundary.values())
        report = {
            "schema_version": "06b-g-independent-confirmation-v1",
            "valid": all(
                metric["nonfinite_state_count"] == 0
                and metric["nonfinite_voltage_count"] == 0
                and metric["state_domain_violation_count"] == 0
                for metric in all_metrics
            ),
            "confirmation_role_is_train_only": True,
            "confirmation_role_is_trajectory_disjoint_from_prior_roles": True,
            "confirmation_used_for_checkpoint_selection": False,
            "validation_state_accessed": False,
            "test_state_accessed": False,
            "per_seed": per_seed,
        }
        atomic._write_json(
            self.output_dir / "independent_confirmation.json", report
        )
        return report

    @staticmethod
    def _median(values: Sequence[float]) -> float:
        return float(np.median(np.asarray(values, dtype=np.float64)))

    def finalize_scheduled_sampling_confirmation(
        self, training: Mapping[str, Any], evaluation: Mapping[str, Any]
    ) -> Dict[str, Any]:
        seeds = list(map(str, self.config.pilot_seeds))
        horizon = f"{max(self.config.rollout_horizons_ms)}_ms"
        one = "1_ms"
        final_budget = str(self.config.scheduled_checkpoints[-1])

        def source(seed: str, arm: str, h: str) -> Mapping[str, Any]:
            return evaluation["per_seed"][seed]["source"][arm]["full_feedback"][h]

        def scheduled(
            seed: str, arm: str, budget: str, h: str
        ) -> Mapping[str, Any]:
            return evaluation["per_seed"][seed]["scheduled"][arm][budget][
                "full_feedback"
            ][h]

        source_rows = {}
        for seed in seeds:
            scalar = source(seed, "full_feedback_scalar", horizon)
            baseline = source(seed, "teacherV_teacherS", horizon)
            scalar_one = source(seed, "full_feedback_scalar", one)
            baseline_one = source(seed, "teacherV_teacherS", one)
            source_rows[seed] = {
                "state_error_reduction": 1.0
                - scalar["normalized_state_rmse"]
                / max(baseline["normalized_state_rmse"], 1e-12),
                "voltage_error_reduction": 1.0
                - scalar["voltage_rmse_mv"] / max(baseline["voltage_rmse_mv"], 1e-12),
                "state_gain": scalar["state_improvement_vs_persistence_fraction"],
                "voltage_gain": scalar["voltage_improvement_vs_persistence_fraction"],
                "physical_violations": scalar["physical_voltage_violation_count"],
                "one_step_degradation": max(
                    scalar_one["normalized_state_rmse"]
                    / max(baseline_one["normalized_state_rmse"], 1e-12)
                    - 1.0,
                    scalar_one["voltage_rmse_mv"]
                    / max(baseline_one["voltage_rmse_mv"], 1e-12)
                    - 1.0,
                ),
            }
        source_median = {
            key: self._median([row[key] for row in source_rows.values()])
            for key in next(iter(source_rows.values()))
        }
        source_confirmed = (
            source_median["state_error_reduction"]
            >= self.config.minimum_confirmation_STATE_error_reduction_fraction
            and source_median["voltage_error_reduction"]
            >= self.config.minimum_confirmation_voltage_error_reduction_fraction
            and source_median["one_step_degradation"]
            <= self.config.maximum_confirmation_one_step_degradation_fraction
            and all(
                row["state_gain"] > 0
                and row["voltage_gain"] > 0
                and row["physical_violations"] == 0
                for row in source_rows.values()
            )
        )

        def joint_error(metric: Mapping[str, Any]) -> float:
            return 0.5 * (
                metric["normalized_state_rmse"]
                / max(metric["persistence_normalized_state_rmse"], 1e-12)
                + metric["voltage_rmse_mv"]
                / max(metric["persistence_voltage_rmse_mv"], 1e-12)
            )

        arm_rows = {}
        for seed in seeds:
            scalar_final = scheduled(seed, "scalar_continue", final_budget, horizon)
            shuffled_final = scheduled(seed, "shuffled_continue", final_budget, horizon)
            baseline = source(seed, "teacherV_teacherS", horizon)
            arm_rows[seed] = {}
            for arm in SCHEDULED_ARMS:
                initial = scheduled(seed, arm, "0", horizon)
                final = scheduled(seed, arm, final_budget, horizon)
                initial_one = scheduled(seed, arm, "0", one)
                final_one = scheduled(seed, arm, final_budget, one)
                scalar_one = scheduled(seed, "scalar_continue", final_budget, one)
                arm_rows[seed][arm] = {
                    "state_gain_over_scalar": 1.0
                    - final["normalized_state_rmse"]
                    / max(scalar_final["normalized_state_rmse"], 1e-12),
                    "voltage_degradation_vs_scalar": final["voltage_rmse_mv"]
                    / max(scalar_final["voltage_rmse_mv"], 1e-12)
                    - 1.0,
                    "state_error_reduction_vs_teacher_reference": 1.0
                    - final["normalized_state_rmse"]
                    / max(baseline["normalized_state_rmse"], 1e-12),
                    "voltage_error_reduction_vs_teacher_reference": 1.0
                    - final["voltage_rmse_mv"]
                    / max(baseline["voltage_rmse_mv"], 1e-12),
                    "state_gain": final["state_improvement_vs_persistence_fraction"],
                    "voltage_gain": final["voltage_improvement_vs_persistence_fraction"],
                    "physical_violations": final["physical_voltage_violation_count"],
                    "one_step_degradation_vs_scalar": max(
                        final_one["normalized_state_rmse"]
                        / max(scalar_one["normalized_state_rmse"], 1e-12)
                        - 1.0,
                        final_one["voltage_rmse_mv"]
                        / max(scalar_one["voltage_rmse_mv"], 1e-12)
                        - 1.0,
                    ),
                    "one_step_degradation_vs_source": max(
                        final_one["normalized_state_rmse"]
                        / max(initial_one["normalized_state_rmse"], 1e-12)
                        - 1.0,
                        final_one["voltage_rmse_mv"]
                        / max(initial_one["voltage_rmse_mv"], 1e-12)
                        - 1.0,
                    ),
                    "scaling_reduction": 1.0
                    - joint_error(final) / max(joint_error(initial), 1e-12),
                    "causal_specificity": scalar_final[
                        "state_improvement_vs_persistence_fraction"
                    ]
                    - shuffled_final["state_improvement_vs_persistence_fraction"],
                }
        median_by_arm = {
            arm: {
                key: self._median(
                    [arm_rows[seed][arm][key] for seed in seeds]
                )
                for key in next(iter(arm_rows[seeds[0]].values()))
            }
            for arm in SCHEDULED_ARMS
        }

        primary = "state_linear_curriculum"
        primary_median = median_by_arm[primary]
        primary_safe = all(
            arm_rows[seed][primary]["state_gain"] > 0
            and arm_rows[seed][primary]["voltage_gain"] > 0
            and arm_rows[seed][primary]["physical_violations"] == 0
            for seed in seeds
        )
        causal_specificity = (
            median_by_arm["scalar_continue"]["causal_specificity"]
            >= self.config.minimum_confirmation_causal_specificity_fraction
        )
        primary_pass = (
            primary_median["state_gain_over_scalar"]
            >= self.config.minimum_curriculum_STATE_gain_fraction
            and primary_median["voltage_degradation_vs_scalar"]
            <= self.config.maximum_curriculum_voltage_degradation_fraction
            and primary_median["state_error_reduction_vs_teacher_reference"]
            >= self.config.minimum_confirmation_STATE_error_reduction_fraction
            and primary_median["voltage_error_reduction_vs_teacher_reference"]
            >= self.config.minimum_confirmation_voltage_error_reduction_fraction
            and primary_median["one_step_degradation_vs_scalar"]
            <= self.config.maximum_confirmation_one_step_degradation_fraction
            and primary_median["scaling_reduction"]
            >= self.config.minimum_continuation_scaling_fraction
            and primary_safe
            and causal_specificity
        )

        fallback = "scalar_continue"
        fallback_median = median_by_arm[fallback]
        fallback_safe = all(
            arm_rows[seed][fallback]["state_gain"] > 0
            and arm_rows[seed][fallback]["voltage_gain"] > 0
            and arm_rows[seed][fallback]["physical_violations"] == 0
            for seed in seeds
        )
        fallback_pass = (
            fallback_median["state_error_reduction_vs_teacher_reference"]
            >= self.config.minimum_confirmation_STATE_error_reduction_fraction
            and fallback_median["voltage_error_reduction_vs_teacher_reference"]
            >= self.config.minimum_confirmation_voltage_error_reduction_fraction
            and fallback_median["scaling_reduction"]
            >= self.config.minimum_continuation_scaling_fraction
            and fallback_median["one_step_degradation_vs_source"]
            <= self.config.maximum_confirmation_one_step_degradation_fraction
            and fallback_safe
            and causal_specificity
        )

        selected: Optional[str] = None
        if source_confirmed and primary_pass:
            selected = primary
            diagnosis = "STATE_CURRICULUM_REPAIR_CONFIRMED"
        elif source_confirmed and fallback_pass:
            selected = fallback
            diagnosis = "SCALAR_FULL_FEEDBACK_CONFIRMED_WITH_CONTINUED_SCALING"
        elif not source_confirmed:
            diagnosis = "SCALAR_SECONDARY_SIGNAL_NOT_INDEPENDENTLY_CONFIRMED"
        else:
            diagnosis = "STATE_EXPOSURE_REFINEMENT_NOT_IDENTIFIED"
        authorize_06c = selected is not None
        report = {
            "schema_version": "06b-g-final-report-v1",
            "valid": bool(training.get("valid") and evaluation.get("valid")),
            "component_decision_grade": True,
            "diagnosis": diagnosis,
            "source_scalar_independently_confirmed": source_confirmed,
            "source_confirmation_median": source_median,
            "source_confirmation_per_seed": source_rows,
            "primary_arm": primary,
            "primary_arm_passed": primary_pass,
            "fallback_arm": fallback,
            "fallback_arm_passed": fallback_pass,
            "hierarchical_decision_rule_preregistered": True,
            "selected_candidate": selected,
            "source_optimizer_state_available": False,
            "optimizer_restart_is_identical_across_arms": True,
            "training_performed": True,
            "bridge_retrained": True,
            "mechanism_STATE_updater_retrained": True,
            "median_by_arm": median_by_arm,
            "per_seed_by_arm": arm_rows,
            "causal_specificity_retained": causal_specificity,
            "registered_thresholds": {
                "minimum_confirmation_STATE_error_reduction_fraction": self.config.minimum_confirmation_STATE_error_reduction_fraction,
                "minimum_confirmation_voltage_error_reduction_fraction": self.config.minimum_confirmation_voltage_error_reduction_fraction,
                "minimum_curriculum_STATE_gain_fraction": self.config.minimum_curriculum_STATE_gain_fraction,
                "maximum_curriculum_voltage_degradation_fraction": self.config.maximum_curriculum_voltage_degradation_fraction,
                "minimum_continuation_scaling_fraction": self.config.minimum_continuation_scaling_fraction,
                "maximum_confirmation_one_step_degradation_fraction": self.config.maximum_confirmation_one_step_degradation_fraction,
                "minimum_confirmation_causal_specificity_fraction": self.config.minimum_confirmation_causal_specificity_fraction,
            },
            "confirmation_used_for_checkpoint_selection": False,
            "development_used_for_selection": False,
            "validation_state_accessed": False,
            "test_state_accessed": False,
            "coupled_06c_canary_authorized": authorize_06c,
            "full_training_authorized": False,
            "fresh_test_generation_authorized": False,
            "mass_dataset_generation_authorized": False,
            "next_step": (
                "06c_coupled_voltage_state_micro_canary"
                if authorize_06c
                else "return_to_atomic_STATE_exposure_playground"
            ),
            "code_revision": self.code_revision,
        }
        atomic._write_json(self.output_dir / "final_report.json", report)
        self._write_artifact_index()
        return report


__all__ = [
    "EXPECTED_06BF_ARCHIVE_SHA256",
    "EXPECTED_06BF_FINAL_SHA256",
    "EXPECTED_06BF_INDEX_SHA256",
    "SCHEDULED_ARMS",
    "StateScheduledSamplingConfig",
    "StateScheduledSamplingConfirmation",
    "verified_06bf_artifact_root",
]
