"""06b-p: atomic effective-source learnability and boundary-shift matrix.

The preceding 06b-o run established an exact cable-equation decomposition and
a useful passive Hines prior, but almost every learned-source arm selected its
zero-output checkpoint.  This train-only component playground separates target
scaling from objective alignment before any further STATE or memory revision.
Models are selected on one-step calibration transitions and then frozen for a
teacher-boundary versus recursive-voltage boundary comparison.
"""

from __future__ import annotations

import hashlib
import itertools
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from . import atomic_state_dynamics_playground as atomic
from .effective_membrane_source_playground import (
    HINES_SOURCE,
    EffectiveMembraneSourceConfig,
    EffectiveMembraneSourcePlayground,
)


EXPECTED_06BO_ARCHIVE_SHA256 = (
    "12e37d796440b23cff376adf78b4dedab11f59595e75398c487b197247549a80"
)
EXPECTED_06BO_INDEX_SHA256 = (
    "0cdd68d3a18dade1c2d0f0f16b3f1576284f7b71cf8401b87573c1e923b3b919"
)
EXPECTED_06BO_FINAL_SHA256 = (
    "1d942ea703e7c39c903b436f6ad86cce3ed4a303f6ff3e47cc2e2aee764ea89b"
)

RAW_SOURCE = "raw_source"
GLOBAL_P99 = "global_p99"
REGION_P99 = "region_p99"
NATIVE_ONLY = "native_only"
ENDPOINT_ONLY = "endpoint_only"
HYBRID = "hybrid"

SCALING_MODES = (RAW_SOURCE, GLOBAL_P99, REGION_P99)
OBJECTIVES = (NATIVE_ONLY, ENDPOINT_ONLY, HYBRID)


def verified_06bo_artifact_root(
    source: Path, cache_dir: Path
) -> Tuple[Path, Dict[str, Any]]:
    """Verify the exact 06b-o artifact that authorizes the atomic follow-up."""

    source = Path(source).expanduser().resolve()
    cache_dir = Path(cache_dir).resolve()
    if not source.exists():
        raise FileNotFoundError(source)
    archive_hash = "extracted-directory"
    if source.is_file():
        if source.suffix.lower() != ".zip":
            raise RuntimeError("06b-o source must be a ZIP or extracted directory")
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
    roots = [
        path.parent
        for path in search_root.rglob("artifact_index.json")
        if atomic._sha256_file(path) == EXPECTED_06BO_INDEX_SHA256
    ]
    if len(roots) != 1:
        raise RuntimeError(f"expected one exact 06b-o artifact; found {len(roots)}")
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
        raise RuntimeError(f"06b-o indexed member verification failed: {failures}")
    final_path = root / "final_report.json"
    if atomic._sha256_file(final_path) != EXPECTED_06BO_FINAL_SHA256:
        raise RuntimeError("06b-o final report SHA-256 mismatch")
    final = json.loads(final_path.read_text(encoding="utf-8"))
    if (
        final.get("valid") is not True
        or final.get("selected_candidate") is not None
        or final.get("validation_state_accessed") is not False
        or final.get("test_state_accessed") is not False
    ):
        raise RuntimeError("06b-o artifact does not authorize atomic reassessment")
    if source.is_file() and archive_hash != EXPECTED_06BO_ARCHIVE_SHA256:
        archive_hash = "kaggle-repacked"
    return root, {
        "valid": True,
        "archive_sha256": archive_hash,
        "artifact_index_sha256": EXPECTED_06BO_INDEX_SHA256,
        "final_report_sha256": EXPECTED_06BO_FINAL_SHA256,
        "indexed_member_count": len(index.get("artifacts", [])),
        "reported_diagnosis": final.get("diagnosis"),
        "registered_reinterpretation": (
            "PASSIVE_HINES_PRIOR_HELPS_LONG_ROLLOUT_BUT_EFFECTIVE_SOURCE_IS_NOT_LEARNED"
        ),
    }


@dataclass(frozen=True)
class AtomicEffectiveSourceConfig(EffectiveMembraneSourceConfig):
    scaling_modes: Tuple[str, ...] = SCALING_MODES
    objectives: Tuple[str, ...] = OBJECTIVES
    atomic_training_steps: int = 300
    atomic_checkpoints: Tuple[int, ...] = (0, 50, 100, 200, 300)
    atomic_batch_transition_count: int = 32
    source_scale_quantile: float = 0.99
    source_scale_floor: float = 1e-4
    hybrid_native_weight: float = 0.25
    minimum_atomic_gain_over_passive_fraction: float = 0.02
    minimum_recursive_gain_over_passive_fraction: float = 0.02
    minimum_trained_selection_fraction: float = 2.0 / 3.0

    def validate(self) -> None:
        super().validate()
        if tuple(self.scaling_modes) != SCALING_MODES:
            raise ValueError("06b-p scaling matrix changed")
        if tuple(self.objectives) != OBJECTIVES:
            raise ValueError("06b-p objective matrix changed")
        if self.atomic_training_steps <= 0:
            raise ValueError("atomic_training_steps must be positive")
        if self.atomic_checkpoints[0] != 0 or self.atomic_checkpoints[-1] != self.atomic_training_steps:
            raise ValueError("atomic checkpoints must span the training budget")
        if self.atomic_batch_transition_count <= 0:
            raise ValueError("atomic batch size must be positive")
        if not 0.5 < self.source_scale_quantile < 1.0:
            raise ValueError("source_scale_quantile must be in (0.5, 1)")
        if self.source_scale_floor <= 0.0:
            raise ValueError("source_scale_floor must be positive")

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "AtomicEffectiveSourceConfig":
        payload = dict(values)
        for name in (
            "pilot_seeds",
            "output_parameterizations",
            "state_feedback_contracts",
            "temporal_contracts",
            "matrix_checkpoints",
            "rollout_horizons_ms",
            "scaling_modes",
            "objectives",
            "atomic_checkpoints",
        ):
            if name in payload:
                payload[name] = tuple(payload[name])
        result = cls(**payload)
        result.validate()
        return result


class AtomicEffectiveSourceLearnability(EffectiveMembraneSourcePlayground):
    """Paired source-scaling/objective matrix with frozen boundary tests."""

    config: AtomicEffectiveSourceConfig

    def __init__(
        self,
        bundle: Any,
        output_dir: Path,
        config: AtomicEffectiveSourceConfig,
        artifact_05t_source: Path,
        artifact_06bn_source: Path,
        artifact_06bo_source: Path,
        *,
        code_revision: str,
    ) -> None:
        super().__init__(
            bundle,
            output_dir,
            config,
            artifact_05t_source,
            artifact_06bn_source,
            code_revision=code_revision,
        )
        self.artifact_06bo_source = Path(artifact_06bo_source)
        self.source_scales: Dict[str, np.ndarray] = {}
        self.atomic_models: Dict[Tuple[str, int], Any] = {}
        self.atomic_training_valid = False

    def _atomic_specs(self) -> Tuple[Tuple[str, str], ...]:
        return tuple(itertools.product(self.config.scaling_modes, self.config.objectives))

    @staticmethod
    def _atomic_key(spec: Tuple[str, str]) -> str:
        return "|".join(spec)

    def _flat_count(self, role: str) -> int:
        values = self.window_data[role]["voltage_t"]
        return int(values.shape[0] * values.shape[1])

    def _flat_tensors(self, role: str, rows: np.ndarray, device: Any) -> Dict[str, Any]:
        values = self.window_data[role]
        window_count, horizon = values["voltage_t"].shape[:2]
        rows = np.asarray(rows, dtype=np.int64)
        window = rows // horizon
        step = rows % horizon
        held = values["held_ions"][window]
        payload = {
            "state": values["state_t"][window, step],
            "voltage": values["voltage_t"][window, step],
            "target_voltage": values["voltage_t1"][window, step],
            "drive": values["drive"][window, step],
            "held_ions": held,
        }
        return {
            name: atomic.torch.as_tensor(value, device=device)
            for name, value in payload.items()
        }

    def _teacher_features(self, batch: Mapping[str, Any]) -> Tuple[Any, Any, Any]:
        center = atomic.torch.as_tensor(
            self.statistics["state_center"],
            dtype=batch["state"].dtype,
            device=batch["state"].device,
        )
        scale = atomic.torch.as_tensor(
            self.statistics["state_scale"],
            dtype=batch["state"].dtype,
            device=batch["state"].device,
        )
        normalized_state = (batch["state"] - center) / scale
        context = atomic.torch.cat((batch["drive"], batch["held_ions"]), dim=-1)
        features = self._features(
            normalized_state, batch["voltage"], context, "authentic"
        )
        target_source = self._normalized_source_target(
            batch["voltage"], batch["target_voltage"], "authentic"
        )
        return features, context, target_source

    def _compute_source_scales(self, device: Any) -> Dict[str, Any]:
        rows = np.arange(self._flat_count("fit"), dtype=np.int64)
        batch = self._flat_tensors("fit", rows, device)
        with atomic.torch.no_grad():
            _, _, target = self._teacher_features(batch)
        absolute = np.abs(target.detach().cpu().numpy().astype(np.float64))
        segment_count = self.layout.segment_count
        global_scale = max(
            self.config.source_scale_floor,
            float(np.quantile(absolute, self.config.source_scale_quantile)),
        )
        region_scale = np.full(segment_count, global_scale, dtype=np.float64)
        region_report: Dict[str, float] = {}
        for region_id, name in enumerate(self.layout.region_names):
            mask = self.layout.segment_region_ids == region_id
            if bool(mask.any()):
                value = max(
                    self.config.source_scale_floor,
                    float(np.quantile(absolute[:, mask], self.config.source_scale_quantile)),
                )
                region_scale[mask] = value
                region_report[str(name)] = value
        self.source_scales = {
            RAW_SOURCE: np.ones(segment_count, dtype=np.float64),
            GLOBAL_P99: np.full(segment_count, global_scale, dtype=np.float64),
            REGION_P99: region_scale,
        }
        support = {}
        for name, scale_values in self.source_scales.items():
            standardized = absolute / scale_values[None, :]
            support[name] = {
                "scale_minimum": float(scale_values.min()),
                "scale_median": float(np.median(scale_values)),
                "scale_maximum": float(scale_values.max()),
                "standardized_p99": float(np.quantile(standardized, 0.99)),
                "standardized_maximum": float(standardized.max()),
                "beyond_output_limit_fraction": float(
                    np.mean(standardized > self.config.normalized_output_limit)
                ),
            }
        return {
            "valid": all(np.isfinite(values).all() and (values > 0).all() for values in self.source_scales.values()),
            "quantile": self.config.source_scale_quantile,
            "floor": self.config.source_scale_floor,
            "global_scale": global_scale,
            "region_scales": region_report,
            "support": support,
        }

    def prepare_atomic_source_learnability(self) -> Dict[str, Any]:
        base = self.prepare_effective_membrane_source_playground()
        _, source = verified_06bo_artifact_root(
            self.artifact_06bo_source,
            self.output_dir.parent / ".06bp_artifact_cache" / "06bo",
        )
        device = atomic.torch.device("cuda" if atomic.torch.cuda.is_available() else "cpu")
        scale_report = self._compute_source_scales(device)
        report = {
            **base,
            "schema_version": "06b-p-atomic-source-contract-v1",
            "experiment": "atomic_effective_source_learnability",
            "source_06bo": source,
            "scientific_question": (
                "Is the effective membrane source learnable after target rescaling, "
                "and does any one-step gain survive recursive voltage boundaries?"
            ),
            "factorial_axes": {
                "source_scaling": list(self.config.scaling_modes),
                "objective": list(self.config.objectives),
            },
            "factor_arm_count": len(self._atomic_specs()),
            "teacher_boundary_training": True,
            "frozen_recursive_voltage_boundary_test": True,
            "teacher_state_used_during_boundary_test": True,
            "predicted_state_trained": False,
            "temporal_memory_trained": False,
            "same_numeric_input_tensor": True,
            "same_initialization_within_seed": True,
            "same_minibatch_stream_within_seed": True,
            "source_scale_report": scale_report,
            "teacher_endpoint_used_as_model_input": False,
            "teacher_source_oracle_selectable": False,
            "validation_state_accessed": False,
            "test_state_accessed": False,
            "configuration": asdict(self.config),
        }
        atomic._write_json(self.output_dir / "atomic_source_contract.json", report)
        return report

    def _scale_tensor(self, mode: str, reference: Any) -> Any:
        return atomic.torch.as_tensor(
            self.source_scales[mode], dtype=reference.dtype, device=reference.device
        )[None, :]

    def _atomic_forward(
        self, model: Any, mode: str, batch: Mapping[str, Any]
    ) -> Tuple[Any, Any, Any, Any]:
        features, _, target_source = self._teacher_features(batch)
        region_ids = atomic.torch.as_tensor(
            self.layout.segment_region_ids,
            dtype=atomic.torch.long,
            device=batch["voltage"].device,
        )
        hidden = atomic.torch.zeros(
            batch["voltage"].shape[0],
            self.layout.segment_count,
            self.config.matrix_hidden_width,
            dtype=batch["voltage"].dtype,
            device=batch["voltage"].device,
        )
        standardized, _ = model(features, region_ids, hidden, recurrent=False)
        source_scale = self._scale_tensor(mode, standardized)
        decoded_source = standardized * source_scale
        prediction = self._apply_output(
            HINES_SOURCE, decoded_source, batch["voltage"], "authentic"
        )
        return standardized, decoded_source, target_source, prediction

    def _atomic_loss(
        self,
        model: Any,
        spec: Tuple[str, str],
        batch: Mapping[str, Any],
    ) -> Tuple[Any, Dict[str, Any]]:
        mode, objective = spec
        standardized, decoded, target_source, prediction = self._atomic_forward(
            model, mode, batch
        )
        target_standardized = target_source / self._scale_tensor(mode, standardized)
        native = atomic.torch_functional.smooth_l1_loss(
            standardized, target_standardized
        )
        endpoint_error = (
            prediction - batch["target_voltage"]
        ) / self.config.voltage_scale_mv
        target_delta = batch["target_voltage"] - batch["voltage"]
        endpoint = atomic.torch.mean(
            self._activity_weight(target_delta)
            * atomic.torch_functional.smooth_l1_loss(
                endpoint_error,
                atomic.torch.zeros_like(endpoint_error),
                reduction="none",
            )
        )
        if objective == NATIVE_ONLY:
            loss = native
        elif objective == ENDPOINT_ONLY:
            loss = endpoint
        elif objective == HYBRID:
            loss = endpoint + self.config.hybrid_native_weight * native
        else:
            raise ValueError(objective)
        return loss, {
            "native_loss": native,
            "endpoint_loss": endpoint,
            "decoded_source": decoded,
            "target_source": target_source,
            "prediction": prediction,
        }

    @staticmethod
    def _correlation(left: np.ndarray, right: np.ndarray) -> Optional[float]:
        left = np.asarray(left, dtype=np.float64).reshape(-1)
        right = np.asarray(right, dtype=np.float64).reshape(-1)
        if left.std() <= 1e-12 or right.std() <= 1e-12:
            return None
        return float(np.corrcoef(left, right)[0, 1])

    def _masked_gain_against(
        self,
        prediction: np.ndarray,
        baseline: np.ndarray,
        target: np.ndarray,
        masks: Mapping[str, np.ndarray],
    ) -> Dict[str, Any]:
        rows = {}
        for name, mask in masks.items():
            expanded = np.broadcast_to(np.asarray(mask, dtype=bool), target.shape)
            count = int(expanded.sum())
            if not count:
                rows[name] = {"coordinate_count": 0, "gain_over_passive_fraction": None}
                continue
            model_rmse = self._rmse(prediction[expanded], target[expanded])
            baseline_rmse = self._rmse(baseline[expanded], target[expanded])
            rows[name] = {
                "coordinate_count": count,
                "model_rmse_mv": model_rmse,
                "passive_rmse_mv": baseline_rmse,
                "gain_over_passive_fraction": 1.0 - model_rmse / max(baseline_rmse, 1e-12),
            }
        return rows

    def _atomic_metrics(
        self, model: Any, mode: str, role: str, device: Any
    ) -> Dict[str, Any]:
        rows = np.arange(self._flat_count(role), dtype=np.int64)
        batch = self._flat_tensors(role, rows, device)
        model.eval()
        with atomic.torch.no_grad():
            standardized, decoded, target_source, prediction = self._atomic_forward(
                model, mode, batch
            )
            passive = self._apply_output(
                HINES_SOURCE,
                atomic.torch.zeros_like(decoded),
                batch["voltage"],
                "authentic",
            )
        prediction_np = prediction.cpu().numpy()
        passive_np = passive.cpu().numpy()
        target_np = batch["target_voltage"].cpu().numpy()
        current_np = batch["voltage"].cpu().numpy()
        decoded_np = decoded.cpu().numpy()
        source_np = target_source.cpu().numpy()
        activity = np.abs(target_np - current_np)
        masks = {
            "quiescent_lt_1mV": activity < 1.0,
            "moderate_1_to_5mV": (activity >= 1.0) & (activity < 5.0),
            "active_ge_5mV": activity >= 5.0,
            "regenerative_ge_20mV": activity >= 20.0,
        }
        model_rmse = self._rmse(prediction_np, target_np)
        passive_rmse = self._rmse(passive_np, target_np)
        source_rmse = self._rmse(decoded_np, source_np)
        source_rms = float(np.sqrt(np.mean(source_np.astype(np.float64) ** 2)))
        return {
            "endpoint_rmse_mv": model_rmse,
            "passive_endpoint_rmse_mv": passive_rmse,
            "endpoint_gain_over_passive_fraction": 1.0 - model_rmse / max(passive_rmse, 1e-12),
            "source_rmse": source_rmse,
            "source_nrmse": source_rmse / max(source_rms, 1e-12),
            "source_correlation": self._correlation(decoded_np, source_np),
            "standardized_output_saturation_fraction": float(
                np.mean(
                    np.abs(standardized.cpu().numpy())
                    >= 0.99 * self.config.normalized_output_limit
                )
            ),
            "nonfinite_count": int((~np.isfinite(prediction_np)).sum()),
            "physical_voltage_violation_count": int(
                (
                    (prediction_np < self.config.physical_voltage_minimum_mv)
                    | (prediction_np > self.config.physical_voltage_maximum_mv)
                ).sum()
            ),
            "activity": self._masked_gain_against(
                prediction_np, passive_np, target_np, masks
            ),
        }

    @staticmethod
    def _gradient_norm(parameters: Sequence[Any]) -> float:
        squares = []
        for parameter in parameters:
            if parameter.grad is not None:
                squares.append(float(parameter.grad.detach().pow(2).sum().cpu()))
        return float(np.sqrt(sum(squares))) if squares else 0.0

    def train_atomic_source_matrix(self) -> Dict[str, Any]:
        device = atomic.torch.device("cuda" if atomic.torch.cuda.is_available() else "cpu")
        specs = self._atomic_specs()
        reports: Dict[str, Any] = {}
        for seed in self.config.pilot_seeds:
            models = {spec: self._new_source_model(seed, device) for spec in specs}
            optimizers = {
                spec: atomic.torch.optim.AdamW(
                    models[spec].parameters(),
                    lr=self.config.matrix_learning_rate,
                    weight_decay=self.config.matrix_weight_decay,
                )
                for spec in specs
            }
            trajectories = {self._atomic_key(spec): [] for spec in specs}
            best: Dict[Tuple[str, str], Tuple[float, int, Dict[str, Any]]] = {}
            initial_gradients: Dict[str, Any] = {}
            rng = np.random.default_rng(seed + 676000)
            digest = hashlib.sha256()
            progress = atomic._CompactProgress(
                f"06b-p 3x3 seed={seed}",
                self.config.atomic_training_steps,
                max(1, self.config.atomic_training_steps // 25),
            )
            for step in range(self.config.atomic_training_steps + 1):
                if step in self.config.atomic_checkpoints:
                    for spec, model in models.items():
                        calibration = self._atomic_metrics(
                            model, spec[0], "calibration", device
                        )
                        score = calibration["endpoint_rmse_mv"]
                        trajectories[self._atomic_key(spec)].append(
                            {"step": step, "calibration": calibration}
                        )
                        if spec not in best or score < best[spec][0]:
                            best[spec] = (
                                score,
                                step,
                                self._copy_state_dict(model),
                            )
                if step == self.config.atomic_training_steps:
                    break
                rows = rng.choice(
                    self._flat_count("fit"),
                    size=self.config.atomic_batch_transition_count,
                    replace=False,
                )
                digest.update(np.asarray(rows, dtype=np.int64).tobytes())
                batch = self._flat_tensors("fit", rows, device)
                losses = []
                for spec, model in models.items():
                    optimizer = optimizers[spec]
                    optimizer.zero_grad(set_to_none=True)
                    loss, components = self._atomic_loss(model, spec, batch)
                    if not bool(atomic.torch.isfinite(loss)):
                        raise RuntimeError(
                            f"non-finite 06b-p loss seed={seed} step={step} arm={self._atomic_key(spec)}"
                        )
                    loss.backward()
                    if step == 0:
                        initial_gradients[self._atomic_key(spec)] = {
                            "total_gradient_norm": self._gradient_norm(list(model.parameters())),
                            "readout_gradient_norm": self._gradient_norm(list(model.readout.parameters())),
                            "native_loss": float(components["native_loss"].detach().cpu()),
                            "endpoint_loss": float(components["endpoint_loss"].detach().cpu()),
                        }
                    atomic.torch.nn.utils.clip_grad_norm_(
                        model.parameters(), self.config.matrix_gradient_clip_norm
                    )
                    optimizer.step()
                    losses.append(float(loss.detach().cpu()))
                progress.update(
                    step + 1,
                    f"loss[min/median/max]={min(losses):.3g}/{float(np.median(losses)):.3g}/{max(losses):.3g}",
                )
            selected = {}
            for spec, model in models.items():
                score, selected_step, state = best[spec]
                model.load_state_dict(state)
                model.eval()
                key = self._atomic_key(spec)
                self.atomic_models[(key, seed)] = model
                path = self.output_dir / f"atomic_source_{key.replace('|','__')}_seed{seed}.pt"
                atomic.torch.save(
                    {
                        "spec": spec,
                        "seed": seed,
                        "selected_step": selected_step,
                        "state_dict": state,
                    },
                    path,
                )
                selected[key] = {
                    "selected_step": selected_step,
                    "selected_calibration_endpoint_rmse_mv": score,
                    "checkpoint": path.name,
                    "checkpoint_sha256": atomic._sha256_file(path),
                }
            reports[str(seed)] = {
                "batch_stream_sha256": digest.hexdigest(),
                "initial_gradients": initial_gradients,
                "trajectories": trajectories,
                "selected": selected,
            }
        report = {
            "schema_version": "06b-p-atomic-source-training-v1",
            "valid": all(
                len(rows) == len(self.config.atomic_checkpoints)
                for seed in reports.values()
                for rows in seed["trajectories"].values()
            ),
            "factor_arm_count": len(specs),
            "same_minibatch_stream_within_seed": True,
            "same_initialization_within_seed": True,
            "calibration_selected_checkpoint": True,
            "development_used_during_training": False,
            "reports": reports,
        }
        self.atomic_training_valid = bool(report["valid"])
        atomic._write_json(self.output_dir / "atomic_source_training.json", report)
        return report

    def _recursive_metrics(
        self, model: Any, mode: str, role: str, device: Any
    ) -> Dict[str, Any]:
        values = self.window_data[role]
        state = atomic.torch.as_tensor(values["state_t"], device=device)
        voltage_t = atomic.torch.as_tensor(values["voltage_t"], device=device)
        target = atomic.torch.as_tensor(values["voltage_t1"], device=device)
        drive = atomic.torch.as_tensor(values["drive"], device=device)
        held = atomic.torch.as_tensor(values["held_ions"], device=device)
        current = voltage_t[:, 0]
        passive = voltage_t[:, 0]
        center = atomic.torch.as_tensor(
            self.statistics["state_center"], dtype=state.dtype, device=device
        )
        state_scale = atomic.torch.as_tensor(
            self.statistics["state_scale"], dtype=state.dtype, device=device
        )
        region_ids = atomic.torch.as_tensor(
            self.layout.segment_region_ids, dtype=atomic.torch.long, device=device
        )
        hidden = atomic.torch.zeros(
            current.shape[0],
            self.layout.segment_count,
            self.config.matrix_hidden_width,
            dtype=current.dtype,
            device=device,
        )
        horizons = {}
        model.eval()
        with atomic.torch.no_grad():
            for step in range(target.shape[1]):
                normalized_state = (state[:, step] - center) / state_scale
                context = atomic.torch.cat((drive[:, step], held), dim=-1)
                features = self._features(
                    normalized_state, current, context, "authentic"
                )
                standardized, _ = model(features, region_ids, hidden, recurrent=False)
                decoded = standardized * self._scale_tensor(mode, standardized)
                current = self._apply_output(
                    HINES_SOURCE, decoded, current, "authentic"
                )
                passive = self._apply_output(
                    HINES_SOURCE, atomic.torch.zeros_like(decoded), passive, "authentic"
                )
                horizon = step + 1
                if horizon in self.config.rollout_horizons_ms:
                    prediction_np = current.cpu().numpy()
                    passive_np = passive.cpu().numpy()
                    target_np = target[:, step].cpu().numpy()
                    initial_np = voltage_t[:, 0].cpu().numpy()
                    activity = np.abs(target_np - initial_np)
                    masks = {
                        "quiescent_lt_1mV": activity < 1.0,
                        "moderate_1_to_5mV": (activity >= 1.0) & (activity < 5.0),
                        "active_ge_5mV": activity >= 5.0,
                        "regenerative_ge_20mV": activity >= 20.0,
                    }
                    model_rmse = self._rmse(prediction_np, target_np)
                    passive_rmse = self._rmse(passive_np, target_np)
                    horizons[f"{horizon}_ms"] = {
                        "endpoint_rmse_mv": model_rmse,
                        "passive_endpoint_rmse_mv": passive_rmse,
                        "endpoint_gain_over_passive_fraction": 1.0 - model_rmse / max(passive_rmse, 1e-12),
                        "physical_voltage_violation_count": int(
                            (
                                (prediction_np < self.config.physical_voltage_minimum_mv)
                                | (prediction_np > self.config.physical_voltage_maximum_mv)
                            ).sum()
                        ),
                        "nonfinite_voltage_count": int(
                            (~np.isfinite(prediction_np)).sum()
                        ),
                        "activity": self._masked_gain_against(
                            prediction_np, passive_np, target_np, masks
                        ),
                    }
        return {"horizons": horizons}

    def evaluate_atomic_and_recursive_boundaries(self) -> Dict[str, Any]:
        if not self.atomic_training_valid:
            raise RuntimeError("06b-p atomic training is incomplete")
        device = atomic.torch.device("cuda" if atomic.torch.cuda.is_available() else "cpu")
        per_seed = {}
        total = len(self._atomic_specs()) * len(self.config.pilot_seeds)
        progress = atomic._CompactProgress(
            "06b-p frozen boundary evaluation", total, max(1, total // 12)
        )
        completed = 0
        for seed in self.config.pilot_seeds:
            rows = {}
            for spec in self._atomic_specs():
                key = self._atomic_key(spec)
                model = self.atomic_models[(key, seed)]
                rows[key] = {
                    "teacher_boundary_one_step": self._atomic_metrics(
                        model, spec[0], "development", device
                    ),
                    "recursive_voltage_teacher_STATE": self._recursive_metrics(
                        model, spec[0], "development", device
                    ),
                }
                completed += 1
                progress.update(completed, f"seed={seed} {key}")
            per_seed[str(seed)] = rows
        report = {
            "schema_version": "06b-p-boundary-evaluation-v1",
            "valid": all(
                row["teacher_boundary_one_step"]["nonfinite_count"] == 0
                and row["recursive_voltage_teacher_STATE"]["horizons"]["8_ms"]["nonfinite_voltage_count"] == 0
                for seed in per_seed.values()
                for row in seed.values()
            ),
            "role": "historically_reused_train_development",
            "teacher_boundary_used_only_for_atomic_evaluation": True,
            "recursive_boundary_uses_teacher_STATE": True,
            "models_retrained_for_boundary_test": False,
            "validation_state_accessed": False,
            "test_state_accessed": False,
            "per_seed": per_seed,
        }
        atomic._write_json(self.output_dir / "atomic_source_boundary_evaluation.json", report)
        return report

    def _summary(self, key: str, evaluation: Mapping[str, Any]) -> Dict[str, Any]:
        rows = [seed[key] for seed in evaluation["per_seed"].values()]
        atomic_rows = [row["teacher_boundary_one_step"] for row in rows]
        recursive_rows = [
            row["recursive_voltage_teacher_STATE"]["horizons"]["8_ms"]
            for row in rows
        ]
        activity_names = recursive_rows[0]["activity"]
        atomic_gain = self._median(
            [row["endpoint_gain_over_passive_fraction"] for row in atomic_rows]
        )
        recursive_gain = self._median(
            [row["endpoint_gain_over_passive_fraction"] for row in recursive_rows]
        )
        return {
            "median_atomic_endpoint_rmse_mv": self._median(
                [row["endpoint_rmse_mv"] for row in atomic_rows]
            ),
            "median_atomic_gain_over_passive_fraction": atomic_gain,
            "median_source_nrmse": self._median(
                [row["source_nrmse"] for row in atomic_rows]
            ),
            "median_source_correlation": self._available_median(
                [row["source_correlation"] for row in atomic_rows]
            ),
            "median_recursive_8ms_rmse_mv": self._median(
                [row["endpoint_rmse_mv"] for row in recursive_rows]
            ),
            "median_recursive_8ms_gain_over_passive_fraction": recursive_gain,
            "boundary_gain_retention_fraction": (
                recursive_gain / atomic_gain if atomic_gain > 1e-12 else None
            ),
            "physical_voltage_violation_count": int(
                sum(row["physical_voltage_violation_count"] for row in recursive_rows)
            ),
            "recursive_activity_gain_over_passive": {
                name: self._available_median(
                    [
                        row["activity"][name]["gain_over_passive_fraction"]
                        for row in recursive_rows
                    ]
                )
                for name in activity_names
            },
        }

    def _main_effect(
        self,
        summaries: Mapping[str, Any],
        axis: int,
        positive: str,
        negative: str,
        metric: str,
    ) -> float:
        contrasts = []
        for spec in self._atomic_specs():
            if spec[axis] != positive:
                continue
            reference = list(spec)
            reference[axis] = negative
            positive_value = summaries[self._atomic_key(spec)][metric]
            negative_value = summaries[self._atomic_key(tuple(reference))][metric]
            contrasts.append(1.0 - positive_value / max(negative_value, 1e-12))
        return self._median(contrasts)

    def _scaling_report(self, training: Mapping[str, Any]) -> Dict[str, Any]:
        report = {}
        for spec in self._atomic_specs():
            key = self._atomic_key(spec)
            by_step = {int(step): [] for step in self.config.atomic_checkpoints}
            selected_steps = []
            for seed in training["reports"].values():
                selected_steps.append(int(seed["selected"][key]["selected_step"]))
                for row in seed["trajectories"][key]:
                    by_step[int(row["step"])].append(
                        float(row["calibration"]["endpoint_rmse_mv"])
                    )
            medians = {
                str(step): self._median(values) for step, values in by_step.items()
            }
            ordered = [medians[str(step)] for step in self.config.atomic_checkpoints]
            report[key] = {
                "median_calibration_endpoint_rmse_by_step": medians,
                "relative_gain_first_to_last": 1.0 - ordered[-1] / max(ordered[0], 1e-12),
                "selected_steps": selected_steps,
                "trained_selection_fraction": float(np.mean(np.asarray(selected_steps) > 0)),
            }
        return report

    def finalize_atomic_source_learnability(
        self,
        contract: Mapping[str, Any],
        training: Mapping[str, Any],
        evaluation: Mapping[str, Any],
    ) -> Dict[str, Any]:
        summaries = {
            self._atomic_key(spec): self._summary(self._atomic_key(spec), evaluation)
            for spec in self._atomic_specs()
        }
        scaling = self._scaling_report(training)
        best_key = min(
            summaries,
            key=lambda key: summaries[key]["median_atomic_endpoint_rmse_mv"],
        )
        best = summaries[best_key]
        activity = best["recursive_activity_gain_over_passive"]
        activity_safe = all(
            activity.get(name) is not None and activity[name] >= 0.0
            for name in ("quiescent_lt_1mV", "moderate_1_to_5mV", "active_ge_5mV")
        )
        trained = scaling[best_key]["trained_selection_fraction"] >= self.config.minimum_trained_selection_fraction
        atomic_signal = (
            best["median_atomic_gain_over_passive_fraction"]
            >= self.config.minimum_atomic_gain_over_passive_fraction
            and trained
        )
        recursive_signal = (
            best["median_recursive_8ms_gain_over_passive_fraction"]
            >= self.config.minimum_recursive_gain_over_passive_fraction
        )
        safe = bool(
            atomic_signal
            and recursive_signal
            and activity_safe
            and best["physical_voltage_violation_count"] == 0
        )
        if safe:
            diagnosis = "SCALED_EFFECTIVE_SOURCE_SURVIVES_RECURSIVE_BOUNDARY"
            next_step = "independent_train_support_scaled_source_confirmation"
        elif atomic_signal and not recursive_signal:
            diagnosis = "SOURCE_LEARNABLE_ON_TEACHER_BOUNDARY_BUT_FAILS_RECURSIVE_BOUNDARY"
            next_step = "causal_recursive_boundary_exposure_or_scheduled_sampling"
        elif atomic_signal:
            diagnosis = "SOURCE_LEARNABLE_BUT_REGIME_SAFETY_FAILS"
            next_step = "source_regime_decomposition_with_explicit_quiet_residual"
        else:
            diagnosis = "CURRENT_CAUSAL_REPRESENTATION_DOES_NOT_LEARN_EFFECTIVE_SOURCE"
            next_step = "revise_source_observability_or_supervised_physical_quantity"
        report = {
            "schema_version": "06b-p-final-report-v1",
            "valid": bool(contract.get("valid") and training.get("valid") and evaluation.get("valid")),
            "component_playground_grade": True,
            "diagnosis": diagnosis,
            "best_observed_arm": best_key,
            "selected_candidate": best_key if safe else None,
            "best_observed_arm_metrics": best,
            "factor_main_effects": {
                "global_p99_over_raw_atomic": self._main_effect(
                    summaries, 0, GLOBAL_P99, RAW_SOURCE, "median_atomic_endpoint_rmse_mv"
                ),
                "region_p99_over_raw_atomic": self._main_effect(
                    summaries, 0, REGION_P99, RAW_SOURCE, "median_atomic_endpoint_rmse_mv"
                ),
                "native_over_endpoint_atomic": self._main_effect(
                    summaries, 1, NATIVE_ONLY, ENDPOINT_ONLY, "median_atomic_endpoint_rmse_mv"
                ),
                "hybrid_over_endpoint_atomic": self._main_effect(
                    summaries, 1, HYBRID, ENDPOINT_ONLY, "median_atomic_endpoint_rmse_mv"
                ),
            },
            "checkpoint_scaling": scaling,
            "atomic_source_signal": atomic_signal,
            "recursive_boundary_signal": recursive_signal,
            "regime_safety_passed": activity_safe,
            "summaries": summaries,
            "teacher_source_oracle_selectable": False,
            "teacher_endpoint_used_as_model_input": False,
            "predicted_state_trained": False,
            "temporal_memory_trained": False,
            "new_independent_confirmation_claimed": False,
            "validation_state_accessed": False,
            "test_state_accessed": False,
            "coupled_06c_canary_authorized": False,
            "full_training_authorized": False,
            "mass_dataset_generation_authorized": False,
            "next_step": next_step,
            "code_revision": self.code_revision,
        }
        atomic._write_json(self.output_dir / "final_report.json", report)
        self._write_artifact_index()
        return report


__all__ = [
    "EXPECTED_06BO_ARCHIVE_SHA256",
    "EXPECTED_06BO_INDEX_SHA256",
    "EXPECTED_06BO_FINAL_SHA256",
    "RAW_SOURCE",
    "GLOBAL_P99",
    "REGION_P99",
    "NATIVE_ONLY",
    "ENDPOINT_ONLY",
    "HYBRID",
    "AtomicEffectiveSourceConfig",
    "AtomicEffectiveSourceLearnability",
    "verified_06bo_artifact_root",
]
