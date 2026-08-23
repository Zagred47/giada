"""Frozen-state forensic for a causal voltage-to-state coupling bridge."""

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
from . import optimized_explicit_state_updater_canary as causal_canary


EXPECTED_06B_ARCHIVE_SHA256 = (
    "0d44d0f6aeb90c7df67a65cd2f92ffbdad9c9163acc11af92ab90f1c52d785ec"
)
EXPECTED_06B_INDEX_SHA256 = (
    "0fe4985566f7276c333bd288280b3756751e82f02353d2e43c791374f126a612"
)
EXPECTED_06B_FINAL_SHA256 = (
    "89512fc5cd37a06c21d59d9d5d74d6418f40afa0d09a3bbaf7a8bf2ff1e4ccc7"
)

COUPLING_MODES = (
    "frozen_causal",
    "predicted_endpoint",
    "shuffled_predicted_endpoint",
    "teacher_endpoint_oracle",
)


def verified_06b_artifact_root(
    source: Path, cache_dir: Path
) -> Tuple[Path, Dict[str, Any]]:
    """Verify the exact component-decision-grade 06b artifact."""

    source = Path(source).expanduser().resolve()
    cache_dir = Path(cache_dir).resolve()
    if not source.exists():
        raise FileNotFoundError(source)
    if source.is_file():
        if source.suffix.lower() != ".zip":
            raise RuntimeError("06b source must be a ZIP or extracted directory")
        archive_hash = atomic._sha256_file(source)
        if archive_hash != EXPECTED_06B_ARCHIVE_SHA256:
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
        if atomic._sha256_file(path) == EXPECTED_06B_INDEX_SHA256
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"expected one exact 06b artifact root; found {len(matches)}"
        )
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
        raise RuntimeError(f"06b indexed member verification failed: {failures}")
    final_path = root / "final_report.json"
    if atomic._sha256_file(final_path) != EXPECTED_06B_FINAL_SHA256:
        raise RuntimeError("06b final report SHA-256 mismatch")
    final = json.loads(final_path.read_text(encoding="utf-8"))
    if (
        final.get("diagnosis") != "ATOMIC_STATE_REQUIRES_EXPLICIT_VOLTAGE_COUPLING"
        or final.get("component_decision_grade") is not True
        or final.get("causal_updater_confirmed") is not False
    ):
        raise RuntimeError("06b source does not authorize coupling forensics")
    return root, {
        "valid": True,
        "source_kind": "zip" if source.is_file() else "directory",
        "archive_sha256": archive_hash,
        "artifact_index_sha256": EXPECTED_06B_INDEX_SHA256,
        "final_report_sha256": EXPECTED_06B_FINAL_SHA256,
        "indexed_member_count": len(index.get("artifacts", [])),
        "diagnosis": final["diagnosis"],
    }


@dataclass(frozen=True)
class CausalVoltageStateCouplingConfig(
    causal_canary.OptimizedExplicitStateCanaryConfig
):
    bridge_training_steps: int = 800
    bridge_hidden_width: int = 64
    bridge_region_embedding_width: int = 8
    bridge_batch_transition_count: int = 8
    bridge_segments_per_transition: int = 256
    bridge_evaluation_interval: int = 100
    bridge_progress_interval: int = 100
    bridge_learning_rate: float = 0.001
    bridge_weight_decay: float = 0.00001
    bridge_gradient_clip_norm: float = 1.0
    bridge_voltage_scale_mv: float = 20.0
    bridge_delta_limit_mv: float = 100.0
    bridge_active_delta_threshold_mv: float = 5.0
    bridge_active_weight: float = 4.0
    maximum_bridge_parameter_count: int = 12000
    minimum_median_voltage_gain_fraction: float = 0.10
    minimum_median_state_gain_over_causal_fraction: float = 0.02
    minimum_median_state_gain_over_shuffled_fraction: float = 0.02
    minimum_median_oracle_gap_recovery_fraction: float = 0.20
    minimum_median_eight_ms_gain_over_causal_fraction: float = 0.02

    def validate(self) -> None:
        super().validate()
        positive = (
            self.bridge_training_steps,
            self.bridge_hidden_width,
            self.bridge_region_embedding_width,
            self.bridge_batch_transition_count,
            self.bridge_segments_per_transition,
            self.bridge_evaluation_interval,
            self.bridge_progress_interval,
            self.maximum_bridge_parameter_count,
        )
        if any(int(value) <= 0 for value in positive):
            raise ValueError("06b-b bridge integer configuration is invalid")
        if not 0 < self.bridge_learning_rate < 1 or self.bridge_weight_decay < 0:
            raise ValueError("06b-b bridge optimizer configuration is invalid")
        if (
            self.bridge_voltage_scale_mv <= 0
            or self.bridge_delta_limit_mv <= 0
            or self.bridge_active_delta_threshold_mv < 0
            or self.bridge_active_weight <= 0
        ):
            raise ValueError("06b-b voltage objective configuration is invalid")
        thresholds = (
            self.minimum_median_voltage_gain_fraction,
            self.minimum_median_state_gain_over_causal_fraction,
            self.minimum_median_state_gain_over_shuffled_fraction,
            self.minimum_median_oracle_gap_recovery_fraction,
            self.minimum_median_eight_ms_gain_over_causal_fraction,
        )
        if any(not 0 < value < 1 for value in thresholds):
            raise ValueError("06b-b registered thresholds must lie in (0, 1)")

    @classmethod
    def from_mapping(
        cls, values: Mapping[str, Any]
    ) -> "CausalVoltageStateCouplingConfig":
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


if atomic.nn is not None:

    class CausalVoltageBridge(atomic.nn.Module):
        """Shared local/morphological predictor of one-ms voltage change."""

        def __init__(
            self,
            *,
            state_width: int,
            presence_width: int,
            context_width: int,
            static_width: int,
            region_count: int,
            region_embedding_width: int,
            hidden_width: int,
            normalized_delta_limit: float,
        ) -> None:
            super().__init__()
            self.normalized_delta_limit = float(normalized_delta_limit)
            self.region_embedding = atomic.nn.Embedding(
                region_count, region_embedding_width
            )
            width = (
                3
                + state_width
                + presence_width
                + context_width
                + static_width
                + region_embedding_width
            )
            self.network = atomic.nn.Sequential(
                atomic.nn.Linear(width, hidden_width),
                atomic.nn.SiLU(),
                atomic.nn.Linear(hidden_width, hidden_width),
                atomic.nn.SiLU(),
                atomic.nn.Linear(hidden_width, 1),
            )
            atomic.nn.init.zeros_(self.network[-1].weight)
            atomic.nn.init.zeros_(self.network[-1].bias)

        def forward(
            self,
            axial_voltage: Any,
            mechanism_state: Any,
            mechanism_presence: Any,
            context: Any,
            static: Any,
            region_id: Any,
        ) -> Any:
            features = atomic.torch.cat(
                (
                    axial_voltage,
                    mechanism_state,
                    mechanism_presence,
                    context,
                    static,
                    self.region_embedding(region_id),
                ),
                dim=-1,
            )
            return self.normalized_delta_limit * atomic.torch.tanh(
                self.network(features).squeeze(-1)
            )

else:  # pragma: no cover

    class CausalVoltageBridge:
        def __init__(self, *_: Any, **__: Any) -> None:
            raise RuntimeError("06b-b requires PyTorch")


class CausalVoltageStateCouplingForensic(
    causal_canary.OptimizedExplicitStateUpdaterCanary
):
    """Train only voltage bridges while all 06b state updaters stay frozen."""

    config: CausalVoltageStateCouplingConfig

    def __init__(
        self,
        bundle: Any,
        output_dir: Path,
        config: CausalVoltageStateCouplingConfig,
        artifact_05t_source: Path,
        artifact_06b_source: Path,
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
        self.artifact_06b_source = Path(artifact_06b_source)
        self.selected_hidden_width: Optional[int] = None
        self.selected_parameter_count: Optional[int] = None
        self.frozen_state_models: Dict[Tuple[str, int], Any] = {}
        self.bridge_models: Dict[int, Any] = {}
        self.segment_state: Dict[str, np.ndarray] = {}
        self.semantic_presence: Optional[np.ndarray] = None
        self.child_ids: Optional[np.ndarray] = None
        self.child_mask: Optional[np.ndarray] = None

    def _materialize_role(self, role: str) -> Dict[str, np.ndarray]:
        return atomic.AtomicStateDynamicsPlayground._materialize_role(self, role)

    def _prepare_segment_state(self) -> None:
        group_count = len(self.coordinate_groups)
        presence = np.zeros(
            (self.layout.segment_count, group_count), dtype=np.float32
        )
        for group, coordinates in self.coordinate_groups.items():
            segments = self.coordinate["segment"][coordinates]
            if len(np.unique(segments)) != len(segments):
                raise RuntimeError("06b-b found duplicate semantic state per segment")
            presence[segments, int(group)] = 1.0
            for role, values in self.materialized.items():
                if role not in self.segment_state:
                    self.segment_state[role] = np.zeros(
                        (
                            len(values["indices"]),
                            self.layout.segment_count,
                            group_count,
                        ),
                        dtype=np.float32,
                    )
                self.segment_state[role][:, segments, int(group)] = values["state"][
                    :, coordinates
                ]
        self.semantic_presence = presence
        max_children = max(1, max(map(len, self.layout.children)))
        child_ids = np.zeros(
            (self.layout.segment_count, max_children), dtype=np.int64
        )
        child_mask = np.zeros_like(child_ids, dtype=np.float32)
        for segment, children in enumerate(self.layout.children):
            if children:
                child_ids[segment, : len(children)] = children
                child_mask[segment, : len(children)] = 1.0
        self.child_ids = child_ids
        self.child_mask = child_mask

    def _new_bridge(self, device: Any) -> Any:
        model = CausalVoltageBridge(
            state_width=len(self.coordinate_groups),
            presence_width=len(self.coordinate_groups),
            context_width=len(atomic.CAUSAL_DRIVE_FEATURES)
            + len(self.ion_feature_names),
            static_width=self.layout.segment_static.shape[1],
            region_count=len(self.layout.region_names),
            region_embedding_width=self.config.bridge_region_embedding_width,
            hidden_width=self.config.bridge_hidden_width,
            normalized_delta_limit=self.config.bridge_delta_limit_mv
            / self.config.bridge_voltage_scale_mv,
        ).to(device)
        count = int(sum(value.numel() for value in model.parameters()))
        if count > self.config.maximum_bridge_parameter_count:
            raise RuntimeError(
                f"06b-b voltage bridge has {count} parameters; ceiling is "
                f"{self.config.maximum_bridge_parameter_count}"
            )
        return model

    def _load_frozen_state_models(self, root: Path, device: Any) -> None:
        for arm in causal_canary.CAUSAL_CANARY_ARMS:
            for seed in self.config.pilot_seeds:
                model = self._new_capacity_capped_model(device)
                checkpoint = atomic.torch.load(
                    root / f"{arm}_seed{seed}.pt",
                    map_location=device,
                    weights_only=False,
                )
                if str(checkpoint.get("arm")) != arm or int(
                    checkpoint.get("seed", -1)
                ) != seed:
                    raise RuntimeError(
                        f"06b-b checkpoint identity mismatch {arm}/{seed}"
                    )
                model.load_state_dict(checkpoint["state_dict"])
                model.eval()
                for parameter in model.parameters():
                    parameter.requires_grad_(False)
                self.frozen_state_models[(arm, seed)] = model

    def prepare_coupling_forensic(self) -> Dict[str, Any]:
        source_root, source_report = verified_06b_artifact_root(
            self.artifact_06b_source,
            self.output_dir.parent / ".06bb_artifact_cache" / "06b",
        )
        base = atomic.AtomicStateDynamicsPlayground.prepare_playground(self)
        self._prepare_segment_state()
        device = atomic.torch.device(
            "cuda" if atomic.torch.cuda.is_available() else "cpu"
        )
        self._load_frozen_state_models(source_root, device)
        bridge = self._new_bridge(device)
        bridge_parameter_count = int(
            sum(value.numel() for value in bridge.parameters())
        )
        contract = {
            **base,
            "schema_version": "06b-b-coupling-contract-v1",
            "experiment": "causal_voltage_state_coupling_forensic",
            "source_06b": source_report,
            "frozen_state_checkpoint_count": len(self.frozen_state_models),
            "state_updater_retraining_performed": False,
            "coupling_modes": list(COUPLING_MODES),
            "trainable_component": "causal_voltage_bridge_only",
            "bridge_parameter_count": bridge_parameter_count,
            "bridge_parameter_ceiling": self.config.maximum_bridge_parameter_count,
            "bridge_inputs": [
                "voltage_t",
                "parent_and_child_voltage_differences_t",
                "mechanism_STATE_t",
                "mechanism_presence",
                "local_ions_t",
                "U_realized",
                "segment_static",
                "region_id",
            ],
            "shuffled_control_uses_same_bridge": True,
            "future_microtraces_read": False,
            "primary_coupled_mode_uses_teacher_endpoint": False,
            "recursive_quantity": "mechanism_STATE_only",
            "voltage_boundary_condition": "teacher_V_t_at_each_ms",
            "autonomous_voltage_rollout_claimed": False,
            "rollout_windows_nested": True,
        }
        contract.pop("teacher_interval_voltage_is_diagnostic_only", None)
        atomic._write_json(self.output_dir / "coupling_contract.json", contract)
        return contract

    def _axial_voltage_features(
        self, voltage: np.ndarray, rows: np.ndarray, segments: np.ndarray
    ) -> np.ndarray:
        parent = self.layout.parent_ids[segments]
        parent_delta = voltage[rows, parent] - voltage[rows, segments]
        children = self.child_ids[segments]
        mask = self.child_mask[segments]
        child_values = voltage[rows[:, None], children]
        child_delta = (
            (child_values - voltage[rows, segments, None]) * mask
        ).sum(axis=1) / np.maximum(mask.sum(axis=1), 1.0)
        return np.stack(
            (
                voltage[rows, segments] / 100.0,
                parent_delta / 100.0,
                child_delta / 100.0,
            ),
            axis=-1,
        ).astype(np.float32)

    def _bridge_batch(
        self,
        role: str,
        transition_rows: np.ndarray,
        segments: np.ndarray,
        device: Any,
        *,
        normalized_state: Optional[np.ndarray] = None,
        voltage_t: Optional[np.ndarray] = None,
        context: Optional[np.ndarray] = None,
    ) -> Tuple[Tuple[Any, ...], Any]:
        values = self.materialized[role]
        rows = np.repeat(transition_rows, segments.shape[1])
        flat_segments = segments.reshape(-1)
        state = (
            self.segment_state[role]
            if normalized_state is None
            else self._state_by_segment(normalized_state)
        )
        voltage = values["voltage_t"] if voltage_t is None else voltage_t
        drive = values["context"] if context is None else context
        tensor = lambda value, dtype=None: atomic.torch.as_tensor(
            value, dtype=dtype, device=device
        )
        inputs = (
            tensor(
                self._axial_voltage_features(voltage, rows, flat_segments),
                atomic.torch.float32,
            ),
            tensor(state[rows, flat_segments], atomic.torch.float32),
            tensor(self.semantic_presence[flat_segments], atomic.torch.float32),
            tensor(drive[rows, flat_segments], atomic.torch.float32),
            tensor(self.layout.segment_static[flat_segments], atomic.torch.float32),
            tensor(self.layout.segment_region_ids[flat_segments], atomic.torch.long),
        )
        target = (
            values["voltage_t1"][rows, flat_segments]
            - values["voltage_t"][rows, flat_segments]
        ) / self.config.bridge_voltage_scale_mv
        return inputs, tensor(target, atomic.torch.float32)

    def _state_by_segment(self, normalized_state: np.ndarray) -> np.ndarray:
        output = np.zeros(
            (
                len(normalized_state),
                self.layout.segment_count,
                len(self.coordinate_groups),
            ),
            dtype=np.float32,
        )
        for group, coordinates in self.coordinate_groups.items():
            segments = self.coordinate["segment"][coordinates]
            output[:, segments, int(group)] = normalized_state[:, coordinates]
        return output

    def _predict_bridge(
        self,
        model: Any,
        normalized_state: np.ndarray,
        voltage_t: np.ndarray,
        context: np.ndarray,
        device: Any,
    ) -> np.ndarray:
        state = self._state_by_segment(normalized_state)
        output = np.empty_like(voltage_t, dtype=np.float32)
        model.eval()
        with atomic.torch.no_grad():
            for start in range(0, self.layout.segment_count, 256):
                stop = min(self.layout.segment_count, start + 256)
                segments = np.arange(start, stop, dtype=np.int64)
                rows = np.repeat(np.arange(len(voltage_t)), len(segments))
                flat_segments = np.tile(segments, len(voltage_t))
                tensor = lambda value, dtype=None: atomic.torch.as_tensor(
                    value, dtype=dtype, device=device
                )
                prediction = model(
                    tensor(
                        self._axial_voltage_features(
                            voltage_t, rows, flat_segments
                        ),
                        atomic.torch.float32,
                    ),
                    tensor(state[rows, flat_segments], atomic.torch.float32),
                    tensor(
                        self.semantic_presence[flat_segments], atomic.torch.float32
                    ),
                    tensor(context[rows, flat_segments], atomic.torch.float32),
                    tensor(
                        self.layout.segment_static[flat_segments], atomic.torch.float32
                    ),
                    tensor(
                        self.layout.segment_region_ids[flat_segments], atomic.torch.long
                    ),
                ).reshape(len(voltage_t), len(segments))
                output[:, start:stop] = (
                    prediction.cpu().numpy() * self.config.bridge_voltage_scale_mv
                )
        return output

    def _evaluate_bridge(
        self, model: Any, role: str, device: Any
    ) -> Dict[str, Any]:
        values = self.materialized[role]
        predicted = self._predict_bridge(
            model,
            values["state"],
            values["voltage_t"],
            values["context"],
            device,
        )
        target = values["voltage_t1"] - values["voltage_t"]
        error = predicted - target
        active = np.abs(target) >= self.config.bridge_active_delta_threshold_mv
        rmse = float(np.sqrt(np.mean(error * error)))
        persistence = float(np.sqrt(np.mean(target * target)))
        active_rmse = (
            float(np.sqrt(np.mean(error[active] ** 2))) if np.any(active) else 0.0
        )
        active_persistence = (
            float(np.sqrt(np.mean(target[active] ** 2))) if np.any(active) else 0.0
        )
        return {
            "voltage_delta_rmse_mv": rmse,
            "persistence_voltage_delta_rmse_mv": persistence,
            "improvement_vs_persistence_fraction": 1.0
            - rmse / max(persistence, 1e-12),
            "active_voltage_delta_rmse_mv": active_rmse,
            "active_persistence_voltage_delta_rmse_mv": active_persistence,
            "active_improvement_vs_persistence_fraction": 1.0
            - active_rmse / max(active_persistence, 1e-12),
            "active_example_count": int(np.sum(active)),
            "nonfinite_count": int(np.sum(~np.isfinite(predicted))),
            "prediction_min_mv": float(np.min(predicted)),
            "prediction_max_mv": float(np.max(predicted)),
        }

    def _train_bridge(self, seed: int, device: Any) -> Dict[str, Any]:
        atomic.torch.manual_seed(seed)
        if atomic.torch.cuda.is_available():
            atomic.torch.cuda.manual_seed_all(seed)
        rng = np.random.default_rng(seed)
        model = self._new_bridge(device)
        optimizer = atomic.torch.optim.AdamW(
            model.parameters(),
            lr=self.config.bridge_learning_rate,
            weight_decay=self.config.bridge_weight_decay,
        )
        best_loss = math.inf
        best_state: Optional[Dict[str, Any]] = None
        curve: List[Dict[str, Any]] = []
        progress = atomic._CompactProgress(
            f"06b-b voltage bridge seed={seed}",
            self.config.bridge_training_steps,
            self.config.bridge_progress_interval,
        )
        fit = self.materialized["fit"]
        for step in range(1, self.config.bridge_training_steps + 1):
            model.train()
            rows = rng.integers(
                0,
                len(fit["indices"]),
                size=self.config.bridge_batch_transition_count,
            )
            segments = rng.integers(
                0,
                self.layout.segment_count,
                size=(
                    self.config.bridge_batch_transition_count,
                    self.config.bridge_segments_per_transition,
                ),
            )
            inputs, target = self._bridge_batch("fit", rows, segments, device)
            prediction = model(*inputs)
            active_threshold = (
                self.config.bridge_active_delta_threshold_mv
                / self.config.bridge_voltage_scale_mv
            )
            weight = 1.0 + self.config.bridge_active_weight * (
                target.abs() >= active_threshold
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
                model.parameters(), self.config.bridge_gradient_clip_norm
            )
            optimizer.step()
            if (
                step == 1
                or step % self.config.bridge_evaluation_interval == 0
                or step == self.config.bridge_training_steps
            ):
                calibration = self._evaluate_bridge(model, "calibration", device)
                score = calibration["voltage_delta_rmse_mv"]
                curve.append(
                    {
                        "step": step,
                        "train_loss": float(loss.detach().cpu()),
                        "calibration_voltage_delta_rmse_mv": score,
                        "calibration_voltage_gain": calibration[
                            "improvement_vs_persistence_fraction"
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
                f"loss={float(loss.detach().cpu()):.4g} calV={best_loss:.4g}",
            )
        if best_state is None:
            raise RuntimeError(f"06b-b did not create voltage bridge seed {seed}")
        selected = self._new_bridge(device)
        selected.load_state_dict(best_state)
        self.bridge_models[seed] = selected
        checkpoint = self.output_dir / f"causal_voltage_bridge_seed{seed}.pt"
        atomic.torch.save(
            {
                "state_dict": best_state,
                "seed": seed,
                "configuration": asdict(self.config),
            },
            checkpoint,
        )
        return {
            "seed": seed,
            "parameter_count": int(
                sum(value.numel() for value in selected.parameters())
            ),
            "best_calibration_voltage_delta_rmse_mv": best_loss,
            "development": self._evaluate_bridge(selected, "development", device),
            "learning_curve": curve,
            "checkpoint": checkpoint.name,
            "checkpoint_sha256": atomic._sha256_file(checkpoint),
        }

    def train_voltage_bridges(self) -> Dict[str, Any]:
        device = next(iter(self.frozen_state_models.values())).proposal.weight.device
        runs = {}
        for seed in self.config.pilot_seeds:
            report = self._train_bridge(seed, device)
            runs[str(seed)] = report
            atomic._write_json(
                self.output_dir / f"voltage_bridge_seed{seed}.json", report
            )
        counts = {row["parameter_count"] for row in runs.values()}
        payload = {
            "schema_version": "06b-b-voltage-bridge-pilot-v1",
            "valid": len(counts) == 1
            and next(iter(counts)) <= self.config.maximum_bridge_parameter_count,
            "device": str(device),
            "state_updater_retraining_performed": False,
            "bridge_parameter_count": next(iter(counts)),
            "runs": runs,
        }
        atomic._write_json(self.output_dir / "voltage_bridge_pilot.json", payload)
        return payload

    def _deterministic_permutation(self, count: int, seed: int, tag: str) -> np.ndarray:
        return np.asarray(
            sorted(
                range(count),
                key=lambda value: hashlib.sha256(
                    f"06b-b|{seed}|{tag}|{value}".encode()
                ).hexdigest(),
            ),
            dtype=np.int64,
        )

    def _state_metrics_from_path(
        self,
        model: Any,
        role: str,
        path_delta_mv: np.ndarray,
        device: Any,
    ) -> Dict[str, Any]:
        values = self.materialized[role]
        fractions = np.asarray(
            self.config.voltage_path_sample_indices, dtype=np.float32
        ) / float(self.config.expected_microtrace_sample_count - 1)
        squared_error = persistence_error = 0.0
        active_squared_error = active_persistence_error = 0.0
        active_count = count = 0
        group_count = len(self.coordinate_groups)
        group_sse = np.zeros(group_count, dtype=np.float64)
        group_persistence = np.zeros(group_count, dtype=np.float64)
        group_examples = np.zeros(group_count, dtype=np.int64)
        model.eval()
        with atomic.torch.no_grad():
            for start in range(
                0, len(self.mechanism_records), self.config.evaluation_coordinate_chunk
            ):
                stop = min(
                    len(self.mechanism_records),
                    start + self.config.evaluation_coordinate_chunk,
                )
                cols = np.arange(start, stop, dtype=np.int64)
                rows = np.repeat(np.arange(len(values["indices"])), len(cols))
                flat_cols = np.tile(cols, len(values["indices"]))
                segments = self.coordinate["segment"][flat_cols]
                path = path_delta_mv[rows, segments, None] * fractions[None, :]
                tensor = lambda value, dtype=None: atomic.torch.as_tensor(
                    value, dtype=dtype, device=device
                )
                prediction = model(
                    tensor(values["state"][rows, flat_cols], atomic.torch.float32),
                    tensor(values["voltage_t"][rows, segments], atomic.torch.float32),
                    tensor(path, atomic.torch.float32),
                    tensor(values["context"][rows, segments], atomic.torch.float32),
                    tensor(self.layout.segment_static[segments], atomic.torch.float32),
                    tensor(self.coordinate["mechanism"][flat_cols], atomic.torch.long),
                    tensor(self.coordinate["variable"][flat_cols], atomic.torch.long),
                    tensor(self.coordinate["kind"][flat_cols], atomic.torch.long),
                    tensor(self.coordinate["region"][flat_cols], atomic.torch.long),
                )
                target = tensor(values["delta"][rows, flat_cols], atomic.torch.float32)
                error = prediction - target
                active = target.abs() >= self.config.active_delta_threshold
                error_np = error.cpu().numpy().reshape(
                    len(values["indices"]), len(cols)
                )
                target_np = target.cpu().numpy().reshape(
                    len(values["indices"]), len(cols)
                )
                per_coordinate = np.sum(error_np * error_np, axis=0)
                per_persistence = np.sum(target_np * target_np, axis=0)
                groups = self.coordinate["semantic_group"][cols]
                np.add.at(group_sse, groups, per_coordinate)
                np.add.at(group_persistence, groups, per_persistence)
                np.add.at(group_examples, groups, len(values["indices"]))
                squared_error += float(np.sum(per_coordinate))
                persistence_error += float(np.sum(per_persistence))
                if bool(active.any()):
                    active_squared_error += float(
                        atomic.torch.sum(error[active] ** 2).cpu()
                    )
                    active_persistence_error += float(
                        atomic.torch.sum(target[active] ** 2).cpu()
                    )
                    active_count += int(active.sum().item())
                count += int(target.numel())
        rmse = math.sqrt(squared_error / max(count, 1))
        persistence = math.sqrt(persistence_error / max(count, 1))
        group_rmse = np.sqrt(group_sse / np.maximum(group_examples, 1))
        group_persistence_rmse = np.sqrt(
            group_persistence / np.maximum(group_examples, 1)
        )
        group_gains = 1.0 - group_rmse / np.maximum(group_persistence_rmse, 1e-12)
        macro_rmse = float(np.mean(group_rmse))
        macro_persistence = float(np.mean(group_persistence_rmse))
        active_rmse = math.sqrt(active_squared_error / max(active_count, 1))
        active_persistence = math.sqrt(
            active_persistence_error / max(active_count, 1)
        )
        return {
            "normalized_delta_rmse": rmse,
            "persistence_normalized_delta_rmse": persistence,
            "improvement_vs_persistence_fraction": 1.0
            - rmse / max(persistence, 1e-12),
            "semantic_macro_improvement_vs_persistence_fraction": 1.0
            - macro_rmse / max(macro_persistence, 1e-12),
            "active_improvement_vs_persistence_fraction": 1.0
            - active_rmse / max(active_persistence, 1e-12),
            "positive_semantic_group_fraction": float(np.mean(group_gains > 0.0)),
            "nonfinite_count": int(np.sum(~np.isfinite(group_gains))),
        }

    def evaluate_one_step_coupling(self) -> Dict[str, Any]:
        role = "development"
        values = self.materialized[role]
        device = next(iter(self.frozen_state_models.values())).proposal.weight.device
        report = {}
        for seed in self.config.pilot_seeds:
            bridge = self.bridge_models[seed]
            predicted = self._predict_bridge(
                bridge,
                values["state"],
                values["voltage_t"],
                values["context"],
                device,
            )
            permutation = self._deterministic_permutation(
                len(predicted), seed, "development"
            )
            actual = values["voltage_t1"] - values["voltage_t"]
            causal_model = self.frozen_state_models[("causal_start_voltage", seed)]
            endpoint_model = self.frozen_state_models[("linear_endpoint_path", seed)]
            modes = {
                "frozen_causal": self._state_metrics_from_path(
                    causal_model, role, np.zeros_like(predicted), device
                ),
                "predicted_endpoint": self._state_metrics_from_path(
                    endpoint_model, role, predicted, device
                ),
                "shuffled_predicted_endpoint": self._state_metrics_from_path(
                    endpoint_model, role, predicted[permutation], device
                ),
                "teacher_endpoint_oracle": self._state_metrics_from_path(
                    endpoint_model, role, actual, device
                ),
            }
            report[str(seed)] = {
                "voltage_bridge": self._evaluate_bridge(bridge, role, device),
                "modes": modes,
                "shuffle_permutation_sha256": hashlib.sha256(
                    permutation.tobytes()
                ).hexdigest(),
            }
        payload = {
            "schema_version": "06b-b-one-step-coupling-v1",
            "valid": all(
                row["voltage_bridge"]["nonfinite_count"] == 0
                and all(mode["nonfinite_count"] == 0 for mode in row["modes"].values())
                for row in report.values()
            ),
            "state_updater_retraining_performed": False,
            "shuffled_control_uses_same_predictions": True,
            "validation_or_test_accessed": False,
            "per_seed": report,
        }
        atomic._write_json(self.output_dir / "one_step_coupling.json", payload)
        return payload

    def evaluate_coupled_nested_rollouts(self) -> Dict[str, Any]:
        windows = self._nested_development_windows()
        if not windows:
            raise RuntimeError("06b-b found no nested development windows")
        digest = hashlib.sha256(
            json.dumps(
                [list(map(int, row)) for row in windows], separators=(",", ":")
            ).encode()
        ).hexdigest()
        first = np.asarray([row[0] for row in windows], dtype=np.int64)
        initial = atomic.mechanism_logit(
            self.store.read_state(first, "t", categories=("mechanism_states",))
        ).astype(np.float32)
        fractions = np.asarray(
            self.config.voltage_path_sample_indices, dtype=np.float32
        ) / float(self.config.expected_microtrace_sample_count - 1)
        device = next(iter(self.frozen_state_models.values())).proposal.weight.device
        report: Dict[str, Any] = {}
        for seed in self.config.pilot_seeds:
            seed_report = {}
            for mode in COUPLING_MODES:
                transformed = initial.copy()
                model = self.frozen_state_models[
                    (
                        "causal_start_voltage"
                        if mode == "frozen_causal"
                        else "linear_endpoint_path",
                        seed,
                    )
                ]
                horizons = {}
                for step in range(max(self.config.rollout_horizons_ms)):
                    indices = np.asarray([row[step] for row in windows], dtype=np.int64)
                    voltage_t = self.store.read_state(
                        indices, "t", categories=("voltage",)
                    ).astype(np.float32)
                    voltage_t1 = self.store.read_state(
                        indices, "t_plus_1", categories=("voltage",)
                    ).astype(np.float32)
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
                    if mode == "frozen_causal":
                        voltage_delta = np.zeros_like(voltage_t)
                    elif mode == "teacher_endpoint_oracle":
                        voltage_delta = voltage_t1 - voltage_t
                    else:
                        voltage_delta = self._predict_bridge(
                            self.bridge_models[seed],
                            normalized.astype(np.float32),
                            voltage_t,
                            context,
                            device,
                        )
                        if mode == "shuffled_predicted_endpoint":
                            permutation = self._deterministic_permutation(
                                len(voltage_delta), seed, f"rollout-step{step}"
                            )
                            voltage_delta = voltage_delta[permutation]
                    path = voltage_delta[:, :, None] * fractions[None, None, :]
                    predicted_delta = self._predict_full_delta_path(
                        model,
                        normalized.astype(np.float32),
                        voltage_t,
                        path,
                        context,
                        device,
                    )
                    transformed += predicted_delta * self.statistics["delta_scale"]
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
                            "normalized_state_rmse": rmse,
                            "persistence_normalized_state_rmse": persistence,
                            "improvement_vs_persistence_fraction": 1.0
                            - rmse / max(persistence, 1e-12),
                            "window_count": len(windows),
                            "window_set_sha256": digest,
                            "nonfinite_count": int(np.sum(~np.isfinite(raw))),
                            "domain_violation_count": int(
                                np.sum((raw < 0.0) | (raw > 1.0))
                            ),
                        }
                seed_report[mode] = horizons
            report[str(seed)] = seed_report
        payload = {
            "schema_version": "06b-b-coupled-rollouts-v1",
            "valid": all(
                row["nonfinite_count"] == 0 and row["domain_violation_count"] == 0
                for seed in report.values()
                for mode in seed.values()
                for row in mode.values()
            ),
            "common_window_count": len(windows),
            "common_window_set_sha256": digest,
            "all_horizons_are_prefixes_of_same_windows": True,
            "state_updater_retraining_performed": False,
            "validation_or_test_accessed": False,
            "per_seed": report,
        }
        atomic._write_json(self.output_dir / "coupled_nested_rollouts.json", payload)
        return payload

    @staticmethod
    def _median(values: Sequence[float]) -> float:
        return float(np.median(np.asarray(values, dtype=np.float64)))

    def _plot(
        self,
        bridge: Mapping[str, Any],
        one_step: Mapping[str, Any],
        rollout: Mapping[str, Any],
    ) -> List[str]:
        import matplotlib.pyplot as plt

        figure_dir = self.output_dir / "figures"
        figure_dir.mkdir(exist_ok=True)
        figure, axes = plt.subplots(1, 3, figsize=(15, 4))
        for seed, row in bridge["runs"].items():
            curve = row["learning_curve"]
            axes[0].plot(
                [point["step"] for point in curve],
                [point["calibration_voltage_delta_rmse_mv"] for point in curve],
                label=f"seed {seed}",
            )
        labels = list(COUPLING_MODES)
        positions = np.arange(len(labels))
        medians = [
            self._median(
                [
                    one_step["per_seed"][str(seed)]["modes"][mode][
                        "improvement_vs_persistence_fraction"
                    ]
                    for seed in self.config.pilot_seeds
                ]
            )
            for mode in labels
        ]
        axes[1].bar(positions, medians)
        axes[1].set_xticks(positions, labels, rotation=25, ha="right")
        for mode in COUPLING_MODES:
            values = []
            for horizon in self.config.rollout_horizons_ms:
                values.append(
                    self._median(
                        [
                            rollout["per_seed"][str(seed)][mode][f"{horizon}_ms"][
                                "improvement_vs_persistence_fraction"
                            ]
                            for seed in self.config.pilot_seeds
                        ]
                    )
                )
            axes[2].plot(
                self.config.rollout_horizons_ms, values, marker="o", label=mode
            )
        axes[0].set(
            xlabel="bridge optimizer step", ylabel="calibration delta-V RMSE (mV)"
        )
        axes[1].set(ylabel="median one-step STATE gain")
        axes[2].axhline(0.0, color="black", linewidth=1)
        axes[2].set(
            xlabel="nested horizon (ms)",
            ylabel="median recursive STATE gain",
            xticks=list(self.config.rollout_horizons_ms),
        )
        for axis in axes:
            axis.grid(alpha=0.25)
        axes[0].legend(fontsize=8)
        axes[2].legend(fontsize=7)
        figure.tight_layout()
        path = figure_dir / "causal_voltage_state_coupling.png"
        figure.savefig(path, dpi=160)
        plt.close(figure)
        return [str(path.relative_to(self.output_dir))]

    def finalize_coupling_forensic(
        self,
        bridge: Mapping[str, Any],
        one_step: Mapping[str, Any],
        rollout: Mapping[str, Any],
    ) -> Dict[str, Any]:
        per_seed = {}
        for seed in self.config.pilot_seeds:
            row = one_step["per_seed"][str(seed)]
            gains = {
                mode: row["modes"][mode]["improvement_vs_persistence_fraction"]
                for mode in COUPLING_MODES
            }
            oracle_gap = gains["teacher_endpoint_oracle"] - gains["frozen_causal"]
            recovery = (
                gains["predicted_endpoint"] - gains["frozen_causal"]
            ) / max(oracle_gap, 1e-12)
            rollout_eight = {
                mode: rollout["per_seed"][str(seed)][mode]["8_ms"][
                    "improvement_vs_persistence_fraction"
                ]
                for mode in COUPLING_MODES
            }
            per_seed[str(seed)] = {
                "voltage_gain": row["voltage_bridge"][
                    "improvement_vs_persistence_fraction"
                ],
                "state_gains": gains,
                "predicted_gain_over_causal": gains["predicted_endpoint"]
                - gains["frozen_causal"],
                "predicted_gain_over_shuffled": gains["predicted_endpoint"]
                - gains["shuffled_predicted_endpoint"],
                "oracle_gap_recovery": recovery,
                "eight_ms_gains": rollout_eight,
                "eight_ms_predicted_gain_over_causal": rollout_eight[
                    "predicted_endpoint"
                ]
                - rollout_eight["frozen_causal"],
            }
        median = {
            "voltage_gain": self._median(
                [row["voltage_gain"] for row in per_seed.values()]
            ),
            "predicted_gain_over_causal": self._median(
                [row["predicted_gain_over_causal"] for row in per_seed.values()]
            ),
            "predicted_gain_over_shuffled": self._median(
                [row["predicted_gain_over_shuffled"] for row in per_seed.values()]
            ),
            "oracle_gap_recovery": self._median(
                [row["oracle_gap_recovery"] for row in per_seed.values()]
            ),
            "eight_ms_predicted_gain_over_causal": self._median(
                [
                    row["eight_ms_predicted_gain_over_causal"]
                    for row in per_seed.values()
                ]
            ),
        }
        bridge_predictive = (
            all(row["voltage_gain"] > 0.0 for row in per_seed.values())
            and median["voltage_gain"]
            >= self.config.minimum_median_voltage_gain_fraction
        )
        predicted_beats_causal = (
            all(row["predicted_gain_over_causal"] > 0.0 for row in per_seed.values())
            and median["predicted_gain_over_causal"]
            >= self.config.minimum_median_state_gain_over_causal_fraction
        )
        predicted_beats_shuffled = (
            median["predicted_gain_over_shuffled"]
            >= self.config.minimum_median_state_gain_over_shuffled_fraction
        )
        gap_recovered = (
            median["oracle_gap_recovery"]
            >= self.config.minimum_median_oracle_gap_recovery_fraction
        )
        rollout_recovered = (
            median["eight_ms_predicted_gain_over_causal"]
            >= self.config.minimum_median_eight_ms_gain_over_causal_fraction
        )
        coupling_identified = all(
            (
                bridge_predictive,
                predicted_beats_causal,
                predicted_beats_shuffled,
                gap_recovered,
                rollout_recovered,
            )
        )
        if coupling_identified:
            diagnosis = "CAUSAL_VOLTAGE_BRIDGE_IDENTIFIED"
            next_step = "06c_coupled_voltage_state_micro_canary"
        elif not bridge_predictive:
            diagnosis = "CAUSAL_VOLTAGE_BRIDGE_NOT_PREDICTIVE"
            next_step = "inspect_voltage_bridge_representation"
        elif not predicted_beats_shuffled:
            diagnosis = "VOLTAGE_BRIDGE_STATE_GAIN_NOT_CAUSALLY_SPECIFIC"
            next_step = "inspect_bridge_control_and_state_sensitivity"
        else:
            diagnosis = "FROZEN_STATE_UPDATER_INCOMPATIBLE_WITH_PREDICTED_VOLTAGE"
            next_step = "06b_c_joint_voltage_state_training_forensic"
        figures = self._plot(bridge, one_step, rollout)
        final = {
            "schema_version": "06b-b-final-report-v1",
            "valid": bool(
                bridge.get("valid") and one_step.get("valid") and rollout.get("valid")
            ),
            "decision_grade": False,
            "component_decision_grade": True,
            "diagnosis": diagnosis,
            "coupling_identified": coupling_identified,
            "gate_checks": {
                "bridge_predictive": bridge_predictive,
                "predicted_beats_frozen_causal": predicted_beats_causal,
                "predicted_beats_shuffled_control": predicted_beats_shuffled,
                "oracle_gap_recovered": gap_recovered,
                "eight_ms_rollout_recovered": rollout_recovered,
            },
            "per_seed": per_seed,
            "median": median,
            "registered_thresholds": {
                key: value
                for key, value in asdict(self.config).items()
                if key.startswith("minimum_median_")
            },
            "state_updater_retraining_performed": False,
            "frozen_state_checkpoint_count": len(self.frozen_state_models),
            "bridge_parameter_count": bridge["bridge_parameter_count"],
            "bridge_parameter_ceiling": self.config.maximum_bridge_parameter_count,
            "shuffled_control_uses_same_bridge": True,
            "rollout_windows_nested": rollout[
                "all_horizons_are_prefixes_of_same_windows"
            ],
            "recursive_quantity": "mechanism_STATE_only",
            "voltage_boundary_condition": "teacher_V_t_at_each_ms",
            "autonomous_voltage_rollout_claimed": False,
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
                "schema_version": "06b-b-artifact-index-v1",
                "artifact_count": len(records),
                "artifacts": records,
            },
        )


__all__ = [
    "EXPECTED_06B_ARCHIVE_SHA256",
    "EXPECTED_06B_INDEX_SHA256",
    "EXPECTED_06B_FINAL_SHA256",
    "COUPLING_MODES",
    "CausalVoltageBridge",
    "CausalVoltageStateCouplingConfig",
    "CausalVoltageStateCouplingForensic",
    "verified_06b_artifact_root",
]
