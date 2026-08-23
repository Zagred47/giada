"""Multi-seed causal canary for the optimized explicit mechanism-state updater."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from . import atomic_state_dynamics_playground as atomic
from . import atomic_voltage_path_identifiability as voltage_path


EXPECTED_06AB_ARCHIVE_SHA256 = (
    "935c5114c553ddc8658032cf8cac10f868f28929055d87c636967de398f01b1f"
)
EXPECTED_06AB_INDEX_SHA256 = (
    "95d348e1bc7d4a7709592f32fc41354544993ca715e284fbb07df498469f52f5"
)
EXPECTED_06AB_FINAL_SHA256 = (
    "4a4bdfa7660fe8f128c7e15a9be148ebd1a8876fa836a86b53a7ee471461ff22"
)

CAUSAL_CANARY_ARMS = ("causal_start_voltage", "linear_endpoint_path")


def verified_06ab_artifact_root(
    source: Path, cache_dir: Path
) -> Tuple[Path, Dict[str, Any]]:
    """Verify the exact 06a-b result that authorizes the causal canary."""

    source = Path(source).expanduser().resolve()
    cache_dir = Path(cache_dir).resolve()
    if not source.exists():
        raise FileNotFoundError(source)
    if source.is_file():
        if source.suffix.lower() != ".zip":
            raise RuntimeError("06a-b source must be a ZIP or extracted directory")
        archive_hash = atomic._sha256_file(source)
        if archive_hash != EXPECTED_06AB_ARCHIVE_SHA256:
            archive_hash = "kaggle-repacked"
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
        archive_hash = "extracted-directory"
        search_root = source
    matches = [
        path.parent
        for path in search_root.rglob("artifact_index.json")
        if atomic._sha256_file(path) == EXPECTED_06AB_INDEX_SHA256
    ]
    if len(matches) != 1:
        raise RuntimeError(f"expected one exact 06a-b artifact root; found {len(matches)}")
    root = matches[0]
    index = json.loads((root / "artifact_index.json").read_text(encoding="utf-8"))
    failures = []
    for record in index.get("artifacts", []):
        member = root / str(record["path"])
        if (
            not member.is_file()
            or member.stat().st_size != int(record["size_bytes"])
            or atomic._sha256_file(member) != str(record["sha256"])
        ):
            failures.append(str(record["path"]))
    if failures:
        raise RuntimeError(f"06a-b indexed member verification failed: {failures}")
    final_path = root / "final_report.json"
    if atomic._sha256_file(final_path) != EXPECTED_06AB_FINAL_SHA256:
        raise RuntimeError("06a-b final report SHA-256 mismatch")
    final = json.loads(final_path.read_text(encoding="utf-8"))
    if (
        final.get("diagnosis") != "ATOMIC_STATE_WAS_OPTIMIZATION_LIMITED"
        or final.get("optimization_budget_identified") is not True
        or final.get("path_information_identified") is not False
    ):
        raise RuntimeError("06a-b source does not authorize the causal canary")
    return root, {
        "valid": True,
        "source_kind": "zip" if source.is_file() else "directory",
        "archive_sha256": archive_hash,
        "artifact_index_sha256": EXPECTED_06AB_INDEX_SHA256,
        "final_report_sha256": EXPECTED_06AB_FINAL_SHA256,
        "indexed_member_count": len(index.get("artifacts", [])),
        "diagnosis": final["diagnosis"],
    }


@dataclass(frozen=True)
class OptimizedExplicitStateCanaryConfig(voltage_path.AtomicVoltagePathConfig):
    pilot_seeds: Tuple[int, ...] = (61017, 61029, 61043)
    training_steps: int = 1200
    evaluation_interval: int = 100
    progress_interval: int = 200
    minimum_per_seed_causal_gain_fraction: float = 0.02
    minimum_median_causal_gain_fraction: float = 0.10
    minimum_median_semantic_macro_gain_fraction: float = 0.03
    minimum_median_active_gain_fraction: float = 0.10
    minimum_positive_semantic_group_fraction: float = 0.70
    minimum_causal_retention_vs_endpoint_fraction: float = 0.70
    minimum_median_eight_ms_rollout_gain_fraction: float = 0.10

    def validate(self) -> None:
        super().validate()
        if len(set(self.pilot_seeds)) < 3 or any(seed <= 0 for seed in self.pilot_seeds):
            raise ValueError("06b requires at least three unique positive seeds")
        fractions = (
            self.minimum_per_seed_causal_gain_fraction,
            self.minimum_median_causal_gain_fraction,
            self.minimum_median_semantic_macro_gain_fraction,
            self.minimum_median_active_gain_fraction,
            self.minimum_positive_semantic_group_fraction,
            self.minimum_causal_retention_vs_endpoint_fraction,
            self.minimum_median_eight_ms_rollout_gain_fraction,
        )
        if any(not 0 < value < 1 for value in fractions):
            raise ValueError("06b registered thresholds must lie in (0, 1)")

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "OptimizedExplicitStateCanaryConfig":
        payload = dict(values)
        for name in (
            "rollout_horizons_ms",
            "voltage_path_sample_indices",
            "pilot_seeds",
        ):
            if name in payload:
                payload[name] = tuple(map(int, payload[name]))
        result = cls(**payload)
        result.validate()
        return result


class OptimizedExplicitStateUpdaterCanary(
    voltage_path.AtomicVoltagePathIdentifiability
):
    """Three-seed causal-versus-endpoint atomic updater canary."""

    config: OptimizedExplicitStateCanaryConfig

    def __init__(
        self,
        bundle: Any,
        output_dir: Path,
        config: OptimizedExplicitStateCanaryConfig,
        artifact_05t_source: Path,
        artifact_06ab_source: Path,
        *,
        code_revision: str,
    ) -> None:
        atomic.AtomicStateDynamicsPlayground.__init__(
            self,
            bundle,
            output_dir,
            config,
            artifact_05t_source,
            code_revision=code_revision,
        )
        self.artifact_06ab_source = Path(artifact_06ab_source)
        self.selected_hidden_width: Optional[int] = None
        self.selected_parameter_count: Optional[int] = None
        self.canary_models: Dict[Tuple[str, int], Any] = {}

    def _materialize_role(self, role: str) -> Dict[str, np.ndarray]:
        # No future microtrace is materialized or read in the causal canary.
        return atomic.AtomicStateDynamicsPlayground._materialize_role(self, role)

    def _path_for_values(
        self,
        values: Mapping[str, np.ndarray],
        rows: np.ndarray,
        segments: np.ndarray,
        arm: str,
    ) -> np.ndarray:
        width = len(self.config.voltage_path_sample_indices)
        if arm == "causal_start_voltage":
            return np.zeros((len(rows), width), dtype=np.float32)
        if arm == "linear_endpoint_path":
            fraction = np.asarray(
                self.config.voltage_path_sample_indices, dtype=np.float32
            ) / float(self.config.expected_microtrace_sample_count - 1)
            delta = (
                values["voltage_t1"][rows, segments]
                - values["voltage_t"][rows, segments]
            )
            return delta[:, None] * fraction[None, :]
        raise ValueError(arm)

    def prepare_causal_canary(self) -> Dict[str, Any]:
        _, source_06ab = verified_06ab_artifact_root(
            self.artifact_06ab_source,
            self.output_dir.parent / ".06b_artifact_cache" / "06ab",
        )
        base = atomic.AtomicStateDynamicsPlayground.prepare_playground(self)
        probe = self._new_capacity_capped_model(atomic.torch.device("cpu"))
        parameter_count = int(sum(value.numel() for value in probe.parameters()))
        contract = {
            **base,
            "schema_version": "06b-causal-state-canary-contract-v1",
            "experiment": "optimized_explicit_state_updater_canary",
            "source_06ab": source_06ab,
            "arms": list(CAUSAL_CANARY_ARMS),
            "primary_arm": "causal_start_voltage",
            "privileged_reference_arm": "linear_endpoint_path",
            "pilot_seeds": list(self.config.pilot_seeds),
            "training_steps_per_seed_and_arm": self.config.training_steps,
            "same_parameter_count_across_arms_and_seeds": True,
            "parameter_count": parameter_count,
            "parameter_ceiling": self.config.maximum_parameter_count,
            "selected_hidden_width": self.selected_hidden_width,
            "future_microtraces_read": False,
            "teacher_endpoint_read_by_primary_arm": False,
            "deployment_compatible_primary_input": True,
            "checkpoint_selection_role": "train-derived calibration only",
            "rollout_windows_nested": True,
            "semantic_group_robustness_required": True,
        }
        contract.pop("teacher_interval_voltage_is_diagnostic_only", None)
        atomic._write_json(self.output_dir / "causal_canary_contract.json", contract)
        return contract

    def _train_seed_arm(self, arm: str, seed: int, device: Any) -> Dict[str, Any]:
        atomic.torch.manual_seed(seed)
        if atomic.torch.cuda.is_available():
            atomic.torch.cuda.manual_seed_all(seed)
        rng = np.random.default_rng(seed)
        model = self._new_capacity_capped_model(device)
        optimizer = atomic.torch.optim.AdamW(
            model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        fit = self.materialized["fit"]
        best_loss = math.inf
        best_state: Optional[Dict[str, Any]] = None
        curve: List[Dict[str, Any]] = []
        progress = atomic._CompactProgress(
            f"06b {arm} seed={seed}",
            self.config.training_steps,
            self.config.progress_interval,
        )
        for step in range(1, self.config.training_steps + 1):
            model.train()
            rows = rng.integers(
                0, len(fit["indices"]), size=self.config.batch_transition_count
            )
            coordinates = self._sample_coordinates(
                rng,
                self.config.batch_transition_count
                * self.config.coordinates_per_transition,
            ).reshape(self.config.batch_transition_count, -1)
            inputs, target = self._batch_path(fit, rows, coordinates, arm, device)
            prediction = model(*inputs)
            weight = 1.0 + self.config.active_delta_weight * (
                target.abs() >= self.config.active_delta_threshold
            ).float()
            loss = atomic.torch.mean(
                weight
                * atomic.torch_functional.smooth_l1_loss(
                    prediction, target, reduction="none"
                )
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            atomic.torch.nn.utils.clip_grad_norm_(
                model.parameters(), self.config.gradient_clip_norm
            )
            optimizer.step()
            if (
                step == 1
                or step % self.config.evaluation_interval == 0
                or step == self.config.training_steps
            ):
                calibration = self._evaluate_one_step_path(
                    model, "calibration", arm, device
                )
                score = calibration["normalized_delta_rmse"]
                curve.append(
                    {
                        "step": step,
                        "train_loss": float(loss.detach().cpu()),
                        "calibration_normalized_delta_rmse": score,
                        "calibration_semantic_macro_rmse": calibration[
                            "semantic_macro_normalized_delta_rmse"
                        ],
                    }
                )
                if score < best_loss:
                    best_loss = score
                    best_state = {
                        key: value.detach().cpu().clone()
                        for key, value in model.state_dict().items()
                    }
            progress.update(
                step,
                f"loss={float(loss.detach().cpu()):.4g} cal={best_loss:.4g}",
            )
        if best_state is None:
            raise RuntimeError(f"06b did not create a checkpoint for {arm}/{seed}")
        selected = self._new_capacity_capped_model(device)
        selected.load_state_dict(best_state)
        self.canary_models[(arm, seed)] = selected
        checkpoint = self.output_dir / f"{arm}_seed{seed}.pt"
        atomic.torch.save(
            {
                "state_dict": best_state,
                "arm": arm,
                "seed": seed,
                "configuration": asdict(self.config),
            },
            checkpoint,
        )
        development = self._evaluate_one_step_path(
            selected, "development", arm, device
        )
        late = [
            row
            for row in curve
            if row["step"] >= int(0.75 * self.config.training_steps)
        ]
        late_start = late[0]["calibration_normalized_delta_rmse"]
        late_best = min(row["calibration_normalized_delta_rmse"] for row in late)
        late_gain = (late_start - late_best) / max(late_start, 1e-12)
        return {
            "arm": arm,
            "seed": seed,
            "parameter_count": self.selected_parameter_count,
            "best_calibration_normalized_delta_rmse": best_loss,
            "development": development,
            "learning_curve": curve,
            "relative_calibration_improvement_last_quarter": late_gain,
            "calibration_plateau_reached": late_gain
            < self.config.plateau_relative_improvement_fraction,
            "checkpoint": checkpoint.name,
            "checkpoint_sha256": atomic._sha256_file(checkpoint),
        }

    def run_causal_canary(self) -> Dict[str, Any]:
        device = atomic.torch.device(
            "cuda" if atomic.torch.cuda.is_available() else "cpu"
        )
        runs: Dict[str, Dict[str, Any]] = {arm: {} for arm in CAUSAL_CANARY_ARMS}
        for seed in self.config.pilot_seeds:
            for arm in CAUSAL_CANARY_ARMS:
                report = self._train_seed_arm(arm, seed, device)
                runs[arm][str(seed)] = report
                atomic._write_json(
                    self.output_dir / f"run_{arm}_seed{seed}.json", report
                )
        counts = {
            run["parameter_count"]
            for arm in runs.values()
            for run in arm.values()
        }
        payload = {
            "schema_version": "06b-causal-state-canary-v1",
            "valid": len(counts) == 1
            and next(iter(counts)) <= self.config.maximum_parameter_count,
            "device": str(device),
            "same_seed_initialization_and_data_order_within_pairs": True,
            "same_parameter_count_across_all_runs": len(counts) == 1,
            "parameter_count": next(iter(counts)),
            "state_and_outcome_splits_read": ["train"],
            "validation_or_test_accessed": False,
            "runs": runs,
        }
        atomic._write_json(self.output_dir / "causal_canary.json", payload)
        return payload

    def evaluate_causal_nested_rollouts(self) -> Dict[str, Any]:
        expected = {
            (arm, seed)
            for arm in CAUSAL_CANARY_ARMS
            for seed in self.config.pilot_seeds
        }
        if set(self.canary_models) != expected:
            raise RuntimeError("run_causal_canary must precede rollout evaluation")
        windows = self._nested_development_windows()
        if not windows:
            raise RuntimeError("06b found no common maximum-horizon windows")
        window_digest = hashlib.sha256(
            json.dumps(
                [list(map(int, row)) for row in windows], separators=(",", ":")
            ).encode()
        ).hexdigest()
        initial_indices = np.asarray([row[0] for row in windows], dtype=np.int64)
        initial = atomic.mechanism_logit(
            self.store.read_state(
                initial_indices, "t", categories=("mechanism_states",)
            )
        ).astype(np.float32)
        device = next(iter(self.canary_models.values())).proposal.weight.device
        reports: Dict[str, Dict[str, Any]] = {arm: {} for arm in CAUSAL_CANARY_ARMS}
        fractions = np.asarray(
            self.config.voltage_path_sample_indices, dtype=np.float32
        ) / float(self.config.expected_microtrace_sample_count - 1)
        for arm in CAUSAL_CANARY_ARMS:
            for seed in self.config.pilot_seeds:
                transformed = initial.copy()
                horizons: Dict[str, Any] = {}
                model = self.canary_models[(arm, seed)]
                for step in range(max(self.config.rollout_horizons_ms)):
                    indices = np.asarray([row[step] for row in windows], dtype=np.int64)
                    voltage_t = self.store.read_state(
                        indices, "t", categories=("voltage",)
                    ).astype(np.float32)
                    voltage_t1 = self.store.read_state(
                        indices, "t_plus_1", categories=("voltage",)
                    ).astype(np.float32)
                    if arm == "causal_start_voltage":
                        path = np.zeros(
                            (
                                len(indices),
                                self.layout.segment_count,
                                len(fractions),
                            ),
                            dtype=np.float32,
                        )
                    else:
                        path = (voltage_t1 - voltage_t)[:, :, None] * fractions[
                            None, None, :
                        ]
                    context = np.concatenate(
                        (
                            atomic.encode_causal_realized_drive(self.store, indices),
                            self._ion_context(indices),
                        ),
                        axis=-1,
                    )
                    normalized = (
                        transformed - self.statistics["state_center"]
                    ) / self.statistics["state_scale"]
                    delta = self._predict_full_delta_path(
                        model,
                        normalized.astype(np.float32),
                        voltage_t,
                        path,
                        context,
                        device,
                    )
                    transformed += delta * self.statistics["delta_scale"]
                    horizon = step + 1
                    if horizon in self.config.rollout_horizons_ms:
                        target_indices = np.asarray(
                            [row[step] for row in windows], dtype=np.int64
                        )
                        target = atomic.mechanism_logit(
                            self.store.read_state(
                                target_indices,
                                "t_plus_1",
                                categories=("mechanism_states",),
                            )
                        ).astype(np.float32)
                        error = (
                            transformed - target
                        ) / self.statistics["state_scale"]
                        persistence_error = (
                            initial - target
                        ) / self.statistics["state_scale"]
                        rmse = float(np.sqrt(np.mean(error * error)))
                        persistence = float(
                            np.sqrt(np.mean(persistence_error * persistence_error))
                        )
                        raw = atomic.inverse_mechanism_logit(transformed)
                        horizons[f"{horizon}_ms"] = {
                            "window_count": len(windows),
                            "window_set_sha256": window_digest,
                            "normalized_state_rmse": rmse,
                            "persistence_normalized_state_rmse": persistence,
                            "improvement_vs_persistence_fraction": 1.0
                            - rmse / max(persistence, 1e-12),
                            "nonfinite_count": int(np.sum(~np.isfinite(raw))),
                            "domain_violation_count": int(
                                np.sum((raw < 0.0) | (raw > 1.0))
                            ),
                        }
                reports[arm][str(seed)] = horizons
        payload = {
            "schema_version": "06b-causal-nested-rollouts-v1",
            "valid": all(
                row["nonfinite_count"] == 0 and row["domain_violation_count"] == 0
                for arm in reports.values()
                for seed in arm.values()
                for row in seed.values()
            ),
            "common_window_count": len(windows),
            "common_window_set_sha256": window_digest,
            "all_horizons_are_prefixes_of_same_windows": True,
            "state_rollout_is_recursive": True,
            "primary_arm_uses_teacher_endpoint": False,
            "validation_or_test_accessed": False,
            "runs": reports,
        }
        atomic._write_json(self.output_dir / "causal_nested_rollouts.json", payload)
        return payload

    @staticmethod
    def _median(values: Sequence[float]) -> float:
        return float(np.median(np.asarray(values, dtype=np.float64)))

    def _plot_canary(
        self, canary: Mapping[str, Any], rollout: Mapping[str, Any]
    ) -> List[str]:
        import matplotlib.pyplot as plt

        figure_dir = self.output_dir / "figures"
        figure_dir.mkdir(exist_ok=True)
        figure, axes = plt.subplots(1, 2, figsize=(12, 4))
        for arm in CAUSAL_CANARY_ARMS:
            for seed_text, run in canary["runs"][arm].items():
                curve = run["learning_curve"]
                axes[0].plot(
                    [row["step"] for row in curve],
                    [row["calibration_normalized_delta_rmse"] for row in curve],
                    alpha=0.65,
                    label=f"{arm} s{seed_text}",
                )
            horizon_values = []
            lower = []
            upper = []
            for horizon in self.config.rollout_horizons_ms:
                values = [
                    rollout["runs"][arm][str(seed)][f"{horizon}_ms"][
                        "improvement_vs_persistence_fraction"
                    ]
                    for seed in self.config.pilot_seeds
                ]
                horizon_values.append(self._median(values))
                lower.append(float(np.min(values)))
                upper.append(float(np.max(values)))
            axes[1].plot(
                self.config.rollout_horizons_ms,
                horizon_values,
                marker="o",
                label=arm,
            )
            axes[1].fill_between(
                self.config.rollout_horizons_ms, lower, upper, alpha=0.15
            )
        axes[0].set(
            xlabel="optimizer step", ylabel="calibration normalized-delta RMSE"
        )
        axes[1].axhline(0.0, color="black", linewidth=1)
        axes[1].set(
            xlabel="nested recursive-state horizon (ms)",
            ylabel="median improvement vs persistence",
            xticks=list(self.config.rollout_horizons_ms),
        )
        for axis in axes:
            axis.grid(alpha=0.25)
            axis.legend(fontsize=7)
        figure.tight_layout()
        path = figure_dir / "optimized_causal_state_canary.png"
        figure.savefig(path, dpi=160)
        plt.close(figure)
        return [str(path.relative_to(self.output_dir))]

    def finalize_causal_canary(
        self, canary: Mapping[str, Any], rollout: Mapping[str, Any]
    ) -> Dict[str, Any]:
        gains: Dict[str, List[float]] = {arm: [] for arm in CAUSAL_CANARY_ARMS}
        macro: Dict[str, List[float]] = {arm: [] for arm in CAUSAL_CANARY_ARMS}
        active: Dict[str, List[float]] = {arm: [] for arm in CAUSAL_CANARY_ARMS}
        positive_groups: Dict[str, List[float]] = {
            arm: [] for arm in CAUSAL_CANARY_ARMS
        }
        per_seed: Dict[str, Any] = {}
        for seed in self.config.pilot_seeds:
            seed_row: Dict[str, Any] = {}
            for arm in CAUSAL_CANARY_ARMS:
                development = canary["runs"][arm][str(seed)]["development"]
                gain = float(development["improvement_vs_persistence_fraction"])
                macro_gain = float(
                    development["semantic_macro_improvement_vs_persistence_fraction"]
                )
                active_gain = float(
                    development["active_improvement_vs_persistence_fraction"]
                )
                group_values = [
                    float(row["improvement_vs_persistence_fraction"])
                    for row in development["semantic_groups"].values()
                ]
                positive_fraction = float(
                    np.mean(np.asarray(group_values, dtype=np.float64) > 0.0)
                )
                gains[arm].append(gain)
                macro[arm].append(macro_gain)
                active[arm].append(active_gain)
                positive_groups[arm].append(positive_fraction)
                seed_row[arm] = {
                    "one_step_gain": gain,
                    "semantic_macro_gain": macro_gain,
                    "active_gain": active_gain,
                    "positive_semantic_group_fraction": positive_fraction,
                    "eight_ms_rollout_gain": rollout["runs"][arm][str(seed)][
                        "8_ms"
                    ]["improvement_vs_persistence_fraction"],
                }
            seed_row["causal_retention_vs_endpoint"] = seed_row[
                "causal_start_voltage"
            ]["one_step_gain"] / max(
                seed_row["linear_endpoint_path"]["one_step_gain"], 1e-12
            )
            per_seed[str(seed)] = seed_row

        causal_rollout = {
            horizon: [
                rollout["runs"]["causal_start_voltage"][str(seed)][f"{horizon}_ms"][
                    "improvement_vs_persistence_fraction"
                ]
                for seed in self.config.pilot_seeds
            ]
            for horizon in self.config.rollout_horizons_ms
        }
        median_causal = self._median(gains["causal_start_voltage"])
        median_endpoint = self._median(gains["linear_endpoint_path"])
        median_macro = self._median(macro["causal_start_voltage"])
        median_active = self._median(active["causal_start_voltage"])
        median_positive_groups = self._median(
            positive_groups["causal_start_voltage"]
        )
        median_retention = self._median(
            [row["causal_retention_vs_endpoint"] for row in per_seed.values()]
        )
        median_eight_ms = self._median(causal_rollout[8])
        all_causal_per_seed = all(
            value >= self.config.minimum_per_seed_causal_gain_fraction
            for value in gains["causal_start_voltage"]
        )
        all_causal_rollouts_positive = all(
            value > 0.0 for values in causal_rollout.values() for value in values
        )
        causal_confirmed = all(
            (
                all_causal_per_seed,
                median_causal >= self.config.minimum_median_causal_gain_fraction,
                median_macro
                >= self.config.minimum_median_semantic_macro_gain_fraction,
                median_active >= self.config.minimum_median_active_gain_fraction,
                median_positive_groups
                >= self.config.minimum_positive_semantic_group_fraction,
                median_retention
                >= self.config.minimum_causal_retention_vs_endpoint_fraction,
                all_causal_rollouts_positive,
                median_eight_ms
                >= self.config.minimum_median_eight_ms_rollout_gain_fraction,
            )
        )
        endpoint_robust = (
            all(
                value >= self.config.minimum_per_seed_causal_gain_fraction
                for value in gains["linear_endpoint_path"]
            )
            and median_endpoint >= self.config.minimum_median_causal_gain_fraction
        )
        if causal_confirmed:
            diagnosis = "CAUSAL_ATOMIC_STATE_UPDATER_CONFIRMED"
            next_step = "06c_coupled_voltage_state_micro_canary"
        elif endpoint_robust:
            diagnosis = "ATOMIC_STATE_REQUIRES_EXPLICIT_VOLTAGE_COUPLING"
            next_step = "06b_b_causal_voltage_state_coupling_forensic"
        elif median_causal >= self.config.minimum_per_seed_causal_gain_fraction:
            diagnosis = "CAUSAL_ATOMIC_STATE_UPDATER_INCONCLUSIVE"
            next_step = "06b_b_causal_state_updater_replication"
        else:
            diagnosis = "OPTIMIZED_ATOMIC_STATE_GAIN_NOT_ROBUST"
            next_step = "inspect_seed_conditioning_and_state_objective"
        figures = self._plot_canary(canary, rollout)
        final = {
            "schema_version": "06b-final-report-v1",
            "valid": bool(canary.get("valid") and rollout.get("valid")),
            "decision_grade": False,
            "component_decision_grade": True,
            "diagnosis": diagnosis,
            "architecture_family": "HayFlow-ESI",
            "causal_updater_confirmed": causal_confirmed,
            "endpoint_reference_robust": endpoint_robust,
            "per_seed": per_seed,
            "aggregate": {
                "median_causal_one_step_gain": median_causal,
                "median_endpoint_one_step_gain": median_endpoint,
                "median_causal_semantic_macro_gain": median_macro,
                "median_causal_active_gain": median_active,
                "median_causal_positive_semantic_group_fraction": median_positive_groups,
                "median_causal_retention_vs_endpoint": median_retention,
                "median_causal_rollout_gain": {
                    f"{horizon}_ms": self._median(values)
                    for horizon, values in causal_rollout.items()
                },
            },
            "registered_thresholds": {
                key: value
                for key, value in asdict(self.config).items()
                if key.startswith("minimum_")
            },
            "all_causal_per_seed_gates_passed": all_causal_per_seed,
            "all_causal_rollout_gains_positive": all_causal_rollouts_positive,
            "parameter_count": canary["parameter_count"],
            "parameter_ceiling": self.config.maximum_parameter_count,
            "same_parameter_count_across_all_runs": canary[
                "same_parameter_count_across_all_runs"
            ],
            "rollout_windows_nested": rollout[
                "all_horizons_are_prefixes_of_same_windows"
            ],
            "future_microtraces_read": False,
            "primary_arm_uses_teacher_endpoint": False,
            "state_and_outcome_splits_read": ["train"],
            "validation_state_accessed": False,
            "test_state_accessed": False,
            "full_neuron_model_trained": False,
            "full_training_authorized": False,
            "fresh_test_generation_authorized": False,
            "mass_dataset_generation_authorized": False,
            "next_step": next_step,
            "figures": figures,
            "code_revision": self.code_revision,
        }
        atomic._write_json(self.output_dir / "final_report.json", final)
        self._write_artifact_index()
        return final

    def _write_artifact_index(self) -> None:
        records = []
        for path in sorted(self.output_dir.rglob("*")):
            if not path.is_file() or path.name == "artifact_index.json":
                continue
            records.append(
                {
                    "path": str(path.relative_to(self.output_dir)).replace("\\", "/"),
                    "size_bytes": path.stat().st_size,
                    "sha256": atomic._sha256_file(path),
                }
            )
        atomic._write_json(
            self.output_dir / "artifact_index.json",
            {
                "schema_version": "06b-artifact-index-v1",
                "artifact_count": len(records),
                "artifacts": records,
            },
        )


__all__ = [
    "EXPECTED_06AB_ARCHIVE_SHA256",
    "EXPECTED_06AB_INDEX_SHA256",
    "EXPECTED_06AB_FINAL_SHA256",
    "CAUSAL_CANARY_ARMS",
    "OptimizedExplicitStateCanaryConfig",
    "OptimizedExplicitStateUpdaterCanary",
    "verified_06ab_artifact_root",
]
