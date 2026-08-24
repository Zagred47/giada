"""06b-e: frozen factorial forensic of the recursive boundary contract.

The experiment does not train or select a model.  It restores the exact final
06b-d bridges and crosses voltage, mechanism-STATE and ion feedback while all
models, windows and realized external inputs remain fixed.  This separates
exposure error from missing recurrent variables and optimizer-trajectory
effects before another training intervention is authorized.
"""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np

from . import atomic_state_dynamics_playground as atomic
from .nested_coupling_optimization_scaling_forensic import (
    SCALING_ARMS,
    NestedCouplingOptimizationScalingConfig,
    NestedCouplingOptimizationScalingForensic,
)


EXPECTED_06BD_ARCHIVE_SHA256 = (
    "a630a4532bef1cce49de733fe12b521de765daa906340a7b48ca99ef7e24291e"
)
EXPECTED_06BD_INDEX_SHA256 = (
    "5b280e0d5aa093f54ce0411e873e8f044354b2e30cb3aa21744cfd1061455882"
)
EXPECTED_06BD_FINAL_SHA256 = (
    "499f2df9c3c9c29d8f75cc523d4d089c245e743720e6eb238c3bd10c29c04b6c"
)

BOUNDARY_CONTRACTS: Dict[str, Tuple[bool, bool, bool]] = {
    "teacherV_teacherS_teacherI": (True, True, True),
    "teacherV_predictedS_teacherI": (True, False, True),
    "predictedV_teacherS_teacherI": (False, True, True),
    "predictedV_predictedS_teacherI": (False, False, True),
    "teacherV_teacherS_heldI": (True, True, False),
    "teacherV_predictedS_heldI": (True, False, False),
    "predictedV_teacherS_heldI": (False, True, False),
    "predictedV_predictedS_heldI": (False, False, False),
}


def verified_06bd_artifact_root(
    source: Path, cache_dir: Path
) -> Tuple[Path, Dict[str, Any]]:
    """Verify the exact registered 06b-d artifact and every indexed member."""

    source = Path(source).expanduser().resolve()
    cache_dir = Path(cache_dir).resolve()
    if not source.exists():
        raise FileNotFoundError(source)
    if source.is_file():
        if source.suffix.lower() != ".zip":
            raise RuntimeError("06b-d source must be a ZIP or extracted directory")
        archive_hash = atomic._sha256_file(source)
        if archive_hash != EXPECTED_06BD_ARCHIVE_SHA256:
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
    roots = [
        path.parent
        for path in search_root.rglob("artifact_index.json")
        if atomic._sha256_file(path) == EXPECTED_06BD_INDEX_SHA256
    ]
    if len(roots) != 1:
        raise RuntimeError(f"expected one exact 06b-d artifact; found {len(roots)}")
    root = roots[0]
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
        raise RuntimeError(f"06b-d indexed member verification failed: {failures}")
    final_path = root / "final_report.json"
    if atomic._sha256_file(final_path) != EXPECTED_06BD_FINAL_SHA256:
        raise RuntimeError("06b-d final report SHA-256 mismatch")
    final = json.loads(final_path.read_text(encoding="utf-8"))
    if (
        final.get("diagnosis") != "ONE_STEP_COUPLING_OBJECTIVE_ONLY"
        or final.get("component_decision_grade") is not True
        or final.get("next_step") != "redesign_recursive_voltage_state_contract"
    ):
        raise RuntimeError("06b-d source does not authorize contract forensics")
    return root, {
        "valid": True,
        "source_kind": "zip" if source.is_file() else "directory",
        "archive_sha256": archive_hash,
        "artifact_index_sha256": EXPECTED_06BD_INDEX_SHA256,
        "final_report_sha256": EXPECTED_06BD_FINAL_SHA256,
        "indexed_member_count": len(index.get("artifacts", [])),
        "diagnosis": final["diagnosis"],
    }


@dataclass(frozen=True)
class RecursiveVoltageStateContractConfig(NestedCouplingOptimizationScalingConfig):
    frozen_checkpoint_budget: int = 1500
    minimum_material_feedback_effect_fraction: float = 0.02
    minimum_material_interaction_fraction: float = 0.01
    minimum_causal_specificity_fraction: float = 0.01
    minimum_positive_seed_count: int = 2
    physical_voltage_minimum_mv: float = -120.0
    physical_voltage_maximum_mv: float = 80.0

    def validate(self) -> None:
        super().validate()
        if self.frozen_checkpoint_budget != self.scaling_training_steps:
            raise ValueError("06b-e must use the preregistered final 06b-d checkpoint")
        for value in (
            self.minimum_material_feedback_effect_fraction,
            self.minimum_material_interaction_fraction,
            self.minimum_causal_specificity_fraction,
        ):
            if not 0 < value < 1:
                raise ValueError("06b-e effect thresholds must lie in (0, 1)")
        if not 1 <= self.minimum_positive_seed_count <= len(self.pilot_seeds):
            raise ValueError("06b-e positive-seed requirement is invalid")
        if self.physical_voltage_minimum_mv >= self.physical_voltage_maximum_mv:
            raise ValueError("06b-e physical voltage bounds are invalid")

    @classmethod
    def from_mapping(
        cls, values: Mapping[str, Any]
    ) -> "RecursiveVoltageStateContractConfig":
        payload = dict(values)
        for name in (
            "rollout_horizons_ms",
            "voltage_path_sample_indices",
            "pilot_seeds",
            "scaling_checkpoints",
        ):
            if name in payload:
                payload[name] = tuple(map(int, payload[name]))
        result = cls(**payload)
        result.validate()
        return result


class RecursiveVoltageStateContractForensic(
    NestedCouplingOptimizationScalingForensic
):
    """Cross frozen boundary substitutions on common nested windows."""

    config: RecursiveVoltageStateContractConfig

    def __init__(
        self,
        bundle: Any,
        output_dir: Path,
        config: RecursiveVoltageStateContractConfig,
        artifact_05t_source: Path,
        artifact_06b_source: Path,
        artifact_06bb_source: Path,
        artifact_06bc_source: Path,
        artifact_06bd_source: Path,
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
            code_revision=code_revision,
        )
        self.artifact_06bd_source = Path(artifact_06bd_source)
        self.contract_models: Dict[Tuple[str, int], Any] = {}

    def _load_frozen_06bd_models(self, root: Path, device: Any) -> Dict[str, str]:
        hashes: Dict[str, str] = {}
        for seed in self.config.pilot_seeds:
            for arm in SCALING_ARMS:
                name = (
                    f"scaling_{arm}_seed{seed}_step"
                    f"{self.config.frozen_checkpoint_budget}.pt"
                )
                path = root / name
                checkpoint = atomic.torch.load(
                    path, map_location=device, weights_only=False
                )
                if (
                    str(checkpoint.get("arm")) != arm
                    or int(checkpoint.get("seed", -1)) != seed
                    or int(checkpoint.get("budget", -1))
                    != self.config.frozen_checkpoint_budget
                ):
                    raise RuntimeError(f"06b-e checkpoint identity mismatch: {name}")
                model = self._new_bridge(device)
                model.load_state_dict(copy.deepcopy(checkpoint["state_dict"]))
                model.eval()
                for parameter in model.parameters():
                    parameter.requires_grad_(False)
                self.contract_models[(arm, seed)] = model
                hashes[name] = atomic._sha256_file(path)
        return hashes

    def prepare_recursive_contract_forensic(self) -> Dict[str, Any]:
        base = self.prepare_scaling_forensic()
        source_root, source = verified_06bd_artifact_root(
            self.artifact_06bd_source,
            self.output_dir.parent / ".06be_artifact_cache" / "06bd",
        )
        device = next(iter(self.frozen_state_models.values())).proposal.weight.device
        checkpoint_hashes = self._load_frozen_06bd_models(source_root, device)
        parameter_counts = {
            int(sum(value.numel() for value in model.parameters()))
            for model in self.contract_models.values()
        }
        blockers = []
        if len(parameter_counts) != 1:
            blockers.append("frozen 06b-d parameter counts differ")
        if len(self.contract_models) != len(SCALING_ARMS) * len(
            self.config.pilot_seeds
        ):
            blockers.append("not all frozen 06b-d models were restored")
        report = {
            **base,
            "schema_version": "06b-e-recursive-contract-v1",
            "valid": bool(base.get("valid")) and not blockers,
            "blockers": blockers,
            "experiment": "recursive_voltage_state_contract_forensic",
            "source_06bd": source,
            "frozen_checkpoint_budget": self.config.frozen_checkpoint_budget,
            "frozen_model_arms": list(SCALING_ARMS),
            "frozen_model_count": len(self.contract_models),
            "frozen_bridge_parameter_count": next(iter(parameter_counts)),
            "checkpoint_sha256": checkpoint_hashes,
            "boundary_contracts": {
                name: {
                    "teacher_voltage_feedback": flags[0],
                    "teacher_mechanism_state_feedback": flags[1],
                    "teacher_ion_feedback": flags[2],
                }
                for name, flags in BOUNDARY_CONTRACTS.items()
            },
            "factorial_axes": [
                "teacher_vs_predicted_voltage_feedback",
                "teacher_vs_predicted_mechanism_STATE_feedback",
                "teacher_vs_held_initial_ion_context",
                "frozen_optimization_and_objective_arm",
            ],
            "realized_external_input_is_step_specific": True,
            "synaptic_internal_state_is_not_closed": True,
            "training_performed": False,
            "candidate_selection_performed": False,
            "development_used_for_selection": False,
            "validation_state_accessed": False,
            "test_state_accessed": False,
            "autonomous_neuron_rollout_claimed": False,
        }
        atomic._write_json(self.output_dir / "recursive_contract.json", report)
        if blockers:
            raise RuntimeError(f"06b-e preflight failed: {blockers}")
        return report

    def _window_cache(self) -> Tuple[List[np.ndarray], str, List[Dict[str, Any]]]:
        windows = self._nested_development_windows()
        if not windows:
            raise RuntimeError("06b-e found no nested development windows")
        digest = hashlib.sha256(
            json.dumps(
                [list(map(int, row)) for row in windows], separators=(",", ":")
            ).encode()
        ).hexdigest()
        cached = []
        for step in range(max(self.config.rollout_horizons_ms)):
            indices = np.asarray([row[step] for row in windows], dtype=np.int64)
            state_t = atomic.mechanism_logit(
                self.store.read_state(
                    indices, "t", categories=("mechanism_states",)
                )
            ).astype(np.float32)
            state_t1 = atomic.mechanism_logit(
                self.store.read_state(
                    indices, "t_plus_1", categories=("mechanism_states",)
                )
            ).astype(np.float32)
            cached.append(
                {
                    "indices": indices,
                    "state_t": state_t,
                    "state_t1": state_t1,
                    "voltage_t": self.store.read_state(
                        indices, "t", categories=("voltage",)
                    ).astype(np.float32),
                    "voltage_t1": self.store.read_state(
                        indices, "t_plus_1", categories=("voltage",)
                    ).astype(np.float32),
                    "drive": atomic.encode_causal_realized_drive(
                        self.store, indices
                    ).astype(np.float32),
                    "ions": self._ion_context(indices).astype(np.float32),
                }
            )
        return windows, digest, cached

    def _horizon_metrics(
        self,
        predicted_state: np.ndarray,
        target_state: np.ndarray,
        initial_state: np.ndarray,
        predicted_voltage: np.ndarray,
        target_voltage: np.ndarray,
        initial_voltage: np.ndarray,
    ) -> Dict[str, Any]:
        state_error = (
            predicted_state - target_state
        ) / self.statistics["state_scale"]
        state_persistence_error = (
            initial_state - target_state
        ) / self.statistics["state_scale"]
        state_rmse = float(np.sqrt(np.mean(state_error * state_error)))
        state_persistence = float(
            np.sqrt(np.mean(state_persistence_error * state_persistence_error))
        )
        voltage_error = predicted_voltage - target_voltage
        voltage_persistence_error = initial_voltage - target_voltage
        voltage_rmse = float(np.sqrt(np.mean(voltage_error * voltage_error)))
        voltage_persistence = float(
            np.sqrt(np.mean(voltage_persistence_error * voltage_persistence_error))
        )
        raw_state = atomic.inverse_mechanism_logit(predicted_state)
        violations = (predicted_voltage < self.config.physical_voltage_minimum_mv) | (
            predicted_voltage > self.config.physical_voltage_maximum_mv
        )
        return {
            "normalized_state_rmse": state_rmse,
            "persistence_normalized_state_rmse": state_persistence,
            "state_improvement_vs_persistence_fraction": 1.0
            - state_rmse / max(state_persistence, 1e-12),
            "voltage_rmse_mv": voltage_rmse,
            "persistence_voltage_rmse_mv": voltage_persistence,
            "voltage_improvement_vs_persistence_fraction": 1.0
            - voltage_rmse / max(voltage_persistence, 1e-12),
            "endpoint_mean_voltage_drift_mv": float(np.mean(voltage_error)),
            "nonfinite_state_count": int(np.sum(~np.isfinite(raw_state))),
            "state_domain_violation_count": int(
                np.sum((raw_state < 0.0) | (raw_state > 1.0))
            ),
            "nonfinite_voltage_count": int(
                np.sum(~np.isfinite(predicted_voltage))
            ),
            "physical_voltage_violation_count": int(np.sum(violations)),
        }

    def evaluate_recursive_contract_matrix(self) -> Dict[str, Any]:
        if not self.contract_models:
            raise RuntimeError("prepare_recursive_contract_forensic must run first")
        windows, digest, cached = self._window_cache()
        fractions = np.asarray(
            self.config.voltage_path_sample_indices, dtype=np.float32
        ) / float(self.config.expected_microtrace_sample_count - 1)
        initial_state = cached[0]["state_t"].copy()
        initial_voltage = cached[0]["voltage_t"].copy()
        held_ions = cached[0]["ions"].copy()
        device = next(iter(self.frozen_state_models.values())).proposal.weight.device
        total = len(self.config.pilot_seeds) * len(SCALING_ARMS) * len(
            BOUNDARY_CONTRACTS
        )
        progress = atomic._CompactProgress(
            "06b-e frozen boundary matrix", total, max(1, total // 12)
        )
        completed = 0
        per_seed: Dict[str, Any] = {}
        for seed in self.config.pilot_seeds:
            endpoint = self.frozen_state_models[("linear_endpoint_path", seed)]
            seed_rows: Dict[str, Any] = {}
            for arm in SCALING_ARMS:
                model = self.contract_models[(arm, seed)]
                arm_rows: Dict[str, Any] = {}
                for contract, flags in BOUNDARY_CONTRACTS.items():
                    teacher_voltage, teacher_state, teacher_ions = flags
                    carried_state = initial_state.copy()
                    carried_voltage = initial_voltage.copy()
                    horizons: Dict[str, Any] = {}
                    for step, values in enumerate(cached):
                        state_input = (
                            values["state_t"] if teacher_state else carried_state
                        )
                        voltage_input = (
                            values["voltage_t"]
                            if teacher_voltage
                            else carried_voltage
                        )
                        ion_input = values["ions"] if teacher_ions else held_ions
                        context = np.concatenate(
                            (values["drive"], ion_input), axis=-1
                        )
                        normalized = (
                            state_input - self.statistics["state_center"]
                        ) / self.statistics["state_scale"]
                        voltage_delta = self._predict_bridge(
                            model,
                            normalized.astype(np.float32),
                            voltage_input,
                            context,
                            device,
                        )
                        carried_voltage = voltage_input + voltage_delta
                        path = voltage_delta[:, :, None] * fractions[None, None, :]
                        state_delta = self._predict_full_delta_path(
                            endpoint,
                            normalized.astype(np.float32),
                            voltage_input,
                            path,
                            context,
                            device,
                        )
                        carried_state = (
                            state_input
                            + state_delta * self.statistics["delta_scale"]
                        )
                        horizon = step + 1
                        if horizon in self.config.rollout_horizons_ms:
                            horizons[f"{horizon}_ms"] = self._horizon_metrics(
                                carried_state,
                                values["state_t1"],
                                initial_state,
                                carried_voltage,
                                values["voltage_t1"],
                                initial_voltage,
                            )
                    arm_rows[contract] = horizons
                    completed += 1
                    progress.update(completed, f"seed={seed} arm={arm}")
                seed_rows[arm] = arm_rows
            per_seed[str(seed)] = seed_rows
        report = {
            "schema_version": "06b-e-recursive-contract-matrix-v1",
            "valid": all(
                metric["nonfinite_state_count"] == 0
                and metric["nonfinite_voltage_count"] == 0
                and metric["state_domain_violation_count"] == 0
                for seed in per_seed.values()
                for arm in seed.values()
                for contract in arm.values()
                for metric in contract.values()
            ),
            "common_window_count": len(windows),
            "common_window_set_sha256": digest,
            "all_horizons_are_prefixes_of_same_windows": True,
            "frozen_model_arms": list(SCALING_ARMS),
            "boundary_contracts": list(BOUNDARY_CONTRACTS),
            "training_performed": False,
            "state_updater_retraining_performed": False,
            "realized_external_input_is_step_specific": True,
            "validation_state_accessed": False,
            "test_state_accessed": False,
            "per_seed": per_seed,
        }
        atomic._write_json(
            self.output_dir / "recursive_contract_matrix.json", report
        )
        return report

    @staticmethod
    def _median(values: Sequence[float]) -> float:
        return float(np.median(np.asarray(values, dtype=np.float64)))

    def summarize_recursive_contract_effects(
        self, matrix: Mapping[str, Any]
    ) -> Dict[str, Any]:
        horizon = f"{max(self.config.rollout_horizons_ms)}_ms"
        primary = "joint_cosine"
        base = "teacherV_teacherS_teacherI"
        state_feedback = "teacherV_predictedS_teacherI"
        voltage_feedback = "predictedV_predictedS_teacherI"
        ion_feedback = "teacherV_predictedS_heldI"
        full = "predictedV_predictedS_heldI"

        def metric(seed: str, arm: str, contract: str, name: str) -> float:
            return float(matrix["per_seed"][seed][arm][contract][horizon][name])

        def penalty(seed: str, contract: str, reference: str = base) -> float:
            numerator = metric(seed, primary, contract, "normalized_state_rmse")
            denominator = metric(seed, primary, reference, "normalized_state_rmse")
            return numerator / max(denominator, 1e-12) - 1.0

        per_seed = {}
        for seed in map(str, self.config.pilot_seeds):
            all_teacher = metric(seed, primary, base, "normalized_state_rmse")
            state_only = metric(
                seed, primary, state_feedback, "normalized_state_rmse"
            )
            voltage_teacher_state = metric(
                seed,
                primary,
                "predictedV_teacherS_teacherI",
                "normalized_state_rmse",
            )
            both = metric(seed, primary, voltage_feedback, "normalized_state_rmse")
            ion_held_teacher_state = metric(
                seed,
                primary,
                "teacherV_teacherS_heldI",
                "normalized_state_rmse",
            )
            ion_state = metric(seed, primary, ion_feedback, "normalized_state_rmse")
            joint_gain = metric(
                seed, "joint_cosine", full, "state_improvement_vs_persistence_fraction"
            )
            shuffled_gain = metric(
                seed,
                "joint_shuffled_cosine",
                full,
                "state_improvement_vs_persistence_fraction",
            )
            constant_gain = metric(
                seed,
                "joint_constant",
                full,
                "state_improvement_vs_persistence_fraction",
            )
            per_seed[seed] = {
                "state_feedback_penalty": state_only / max(all_teacher, 1e-12) - 1.0,
                "voltage_feedback_penalty": both / max(state_only, 1e-12) - 1.0,
                "ion_feedback_penalty": ion_state / max(state_only, 1e-12) - 1.0,
                "full_boundary_penalty": penalty(seed, full),
                "voltage_state_interaction": (
                    both - state_only - voltage_teacher_state + all_teacher
                )
                / max(all_teacher, 1e-12),
                "state_ion_interaction": (
                    ion_state - state_only - ion_held_teacher_state + all_teacher
                )
                / max(all_teacher, 1e-12),
                "constant_over_cosine_full_state_gain": constant_gain - joint_gain,
                "causal_specificity_full_state_gain": joint_gain - shuffled_gain,
                "full_contract_state_gain": joint_gain,
                "full_contract_voltage_gain": metric(
                    seed,
                    primary,
                    full,
                    "voltage_improvement_vs_persistence_fraction",
                ),
            }
        median = {
            name: self._median([row[name] for row in per_seed.values()])
            for name in next(iter(per_seed.values()))
        }

        def identified(name: str, threshold: float) -> bool:
            return median[name] >= threshold and sum(
                row[name] > 0 for row in per_seed.values()
            ) >= self.config.minimum_positive_seed_count

        identified_limits = []
        for label, name in (
            ("mechanism_state_exposure", "state_feedback_penalty"),
            ("voltage_feedback", "voltage_feedback_penalty"),
            ("ion_state_persistence", "ion_feedback_penalty"),
        ):
            if identified(
                name, self.config.minimum_material_feedback_effect_fraction
            ):
                identified_limits.append(label)
        interactions = []
        for label, name in (
            ("voltage_x_state", "voltage_state_interaction"),
            ("state_x_ion", "state_ion_interaction"),
        ):
            if abs(median[name]) >= self.config.minimum_material_interaction_fraction:
                interactions.append(label)
        optimization_trajectory = identified(
            "constant_over_cosine_full_state_gain",
            self.config.minimum_material_feedback_effect_fraction,
        )
        causal_specificity = identified(
            "causal_specificity_full_state_gain",
            self.config.minimum_causal_specificity_fraction,
        )
        if len(identified_limits) > 1:
            diagnosis = "MULTIPLE_RECURSIVE_BOUNDARY_LIMITS_IDENTIFIED"
        elif identified_limits:
            diagnosis = f"{identified_limits[0].upper()}_PRIMARY_LIMIT"
        elif interactions:
            diagnosis = "RECURSIVE_BOUNDARY_INTERACTION_IDENTIFIED"
        elif optimization_trajectory:
            diagnosis = "OPTIMIZATION_TRAJECTORY_RECURSIVE_LIMIT"
        else:
            diagnosis = "RECURSIVE_CONTRACT_LIMIT_UNRESOLVED"
        report = {
            "schema_version": "06b-e-recursive-effects-v1",
            "valid": bool(matrix.get("valid")),
            "primary_frozen_arm": primary,
            "primary_horizon_ms": max(self.config.rollout_horizons_ms),
            "per_seed": per_seed,
            "median_effects": median,
            "identified_limits": identified_limits,
            "identified_interactions": interactions,
            "optimization_trajectory_effect_identified": optimization_trajectory,
            "causal_specificity_retained_under_full_boundary": causal_specificity,
            "diagnosis": diagnosis,
            "registered_thresholds": {
                "minimum_material_feedback_effect_fraction": self.config.minimum_material_feedback_effect_fraction,
                "minimum_material_interaction_fraction": self.config.minimum_material_interaction_fraction,
                "minimum_causal_specificity_fraction": self.config.minimum_causal_specificity_fraction,
                "minimum_positive_seed_count": self.config.minimum_positive_seed_count,
            },
            "development_used_for_selection": False,
            "validation_state_accessed": False,
            "test_state_accessed": False,
        }
        atomic._write_json(self.output_dir / "recursive_effects.json", report)
        return report

    def _plot_effects(
        self, matrix: Mapping[str, Any], effects: Mapping[str, Any]
    ) -> List[str]:
        import matplotlib.pyplot as plt

        figure_dir = self.output_dir / "figures"
        figure_dir.mkdir(exist_ok=True)
        figure, axes = plt.subplots(1, 2, figsize=(12, 4))
        horizons = list(self.config.rollout_horizons_ms)
        contracts = (
            "teacherV_teacherS_teacherI",
            "teacherV_predictedS_teacherI",
            "predictedV_predictedS_teacherI",
            "predictedV_predictedS_heldI",
        )
        for contract in contracts:
            medians = []
            for horizon in horizons:
                values = [
                    matrix["per_seed"][str(seed)]["joint_cosine"][contract][
                        f"{horizon}_ms"
                    ]["state_improvement_vs_persistence_fraction"]
                    for seed in self.config.pilot_seeds
                ]
                medians.append(self._median(values))
            axes[0].plot(horizons, medians, marker="o", label=contract)
        axes[0].set_xlabel("nested horizon (ms)")
        axes[0].set_ylabel("median STATE gain vs persistence")
        axes[0].legend(fontsize=7)
        names = (
            "state_feedback_penalty",
            "voltage_feedback_penalty",
            "ion_feedback_penalty",
            "full_boundary_penalty",
        )
        axes[1].bar(
            np.arange(len(names)),
            [effects["median_effects"][name] for name in names],
        )
        axes[1].axhline(
            self.config.minimum_material_feedback_effect_fraction,
            color="black",
            linestyle="--",
            linewidth=1,
        )
        axes[1].set_xticks(np.arange(len(names)), names, rotation=25, ha="right")
        axes[1].set_ylabel("relative 8 ms STATE-error penalty")
        figure.tight_layout()
        path = figure_dir / "recursive_boundary_factorial.png"
        figure.savefig(path, dpi=160)
        plt.close(figure)
        return [str(path.relative_to(self.output_dir))]

    def finalize_recursive_contract_forensic(
        self,
        matrix: Mapping[str, Any],
        effects: Mapping[str, Any],
    ) -> Dict[str, Any]:
        valid = bool(matrix.get("valid") and effects.get("valid"))
        report = {
            "schema_version": "06b-e-final-report-v1",
            "valid": valid,
            "component_decision_grade": True,
            "diagnosis": effects["diagnosis"],
            "identified_limits": effects["identified_limits"],
            "identified_interactions": effects["identified_interactions"],
            "optimization_trajectory_effect_identified": effects[
                "optimization_trajectory_effect_identified"
            ],
            "causal_specificity_retained_under_full_boundary": effects[
                "causal_specificity_retained_under_full_boundary"
            ],
            "median_effects": effects["median_effects"],
            "per_seed_effects": effects["per_seed"],
            "registered_thresholds": effects["registered_thresholds"],
            "multiple_questions_answered_in_one_frozen_matrix": [
                "mechanism_STATE_exposure_penalty",
                "predicted_voltage_feedback_penalty",
                "held_ion_context_penalty",
                "boundary_interactions",
                "constant_vs_cosine_trajectory_under_full_feedback",
                "causal_specificity_under_full_feedback",
            ],
            "training_performed": False,
            "candidate_selection_performed": False,
            "state_updater_retraining_performed": False,
            "development_used_for_selection": False,
            "validation_state_accessed": False,
            "test_state_accessed": False,
            "realized_external_input_is_step_specific": True,
            "synaptic_internal_state_is_not_closed": True,
            "autonomous_neuron_rollout_claimed": False,
            "bounded_train_only_repair_matrix_authorized": valid
            and effects["diagnosis"] != "RECURSIVE_CONTRACT_LIMIT_UNRESOLVED",
            "coupled_06c_canary_authorized": False,
            "full_training_authorized": False,
            "fresh_test_generation_authorized": False,
            "mass_dataset_generation_authorized": False,
            "next_step": "preregister_training_matrix_for_identified_recursive_limits"
            if effects["diagnosis"] != "RECURSIVE_CONTRACT_LIMIT_UNRESOLVED"
            else "expand_atomic_recursive_boundary_playground",
            "figures": self._plot_effects(matrix, effects),
            "code_revision": self.code_revision,
        }
        atomic._write_json(self.output_dir / "final_report.json", report)
        self._write_artifact_index()
        return report


__all__ = [
    "EXPECTED_06BD_ARCHIVE_SHA256",
    "EXPECTED_06BD_INDEX_SHA256",
    "EXPECTED_06BD_FINAL_SHA256",
    "BOUNDARY_CONTRACTS",
    "RecursiveVoltageStateContractConfig",
    "RecursiveVoltageStateContractForensic",
    "verified_06bd_artifact_root",
]
