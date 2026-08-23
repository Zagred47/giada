"""Train-only forensic separating optimization from intra-ms voltage path.

This experiment follows the failed 06a technical gate.  It keeps the atomic
mechanism-state problem, the role partition and the parameter ceiling fixed,
then crosses two factors:

* checkpoint budget: 300 versus 1200 optimizer steps;
* voltage context: a linear endpoint interpolation versus the authentic
  teacher microtrace sampled at eight fixed offsets.

Both voltage arms are privileged diagnostics.  They make no deployment claim,
never read validation/test state, and are evaluated on common nested rollout
windows so horizon comparisons are meaningful.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from . import atomic_state_dynamics_playground as atomic


EXPECTED_06A_ARCHIVE_SHA256 = (
    "b2aaa071925c0eea4c34f7e116d491faeea52f9fc166a8b6370e085c6532983d"
)
EXPECTED_06A_INDEX_SHA256 = (
    "ad28ed4666e8bd99fb0be5f5d2230e7b731868e20740e94fdd7537aa56e96cb5"
)
EXPECTED_06A_FINAL_SHA256 = (
    "fa72141a9ca50ceb2582d6eec1852dcd5af7d83e6e34b7e408750a48bf518fa2"
)

VOLTAGE_CONTEXT_ARMS = ("linear_endpoint_path", "teacher_microtrace_path")
BUDGET_LABELS = ("short", "long")


def verified_06a_artifact_root(
    source: Path, cache_dir: Path
) -> Tuple[Path, Dict[str, Any]]:
    """Verify the exact failed 06a artifact that authorizes this forensic."""

    source = Path(source).expanduser().resolve()
    cache_dir = Path(cache_dir).resolve()
    if not source.exists():
        raise FileNotFoundError(source)
    if source.is_file():
        if source.suffix.lower() != ".zip":
            raise RuntimeError("06a source must be a ZIP or extracted directory")
        archive_hash = atomic._sha256_file(source)
        if archive_hash != EXPECTED_06A_ARCHIVE_SHA256:
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

    matching = [
        path.parent
        for path in search_root.rglob("artifact_index.json")
        if atomic._sha256_file(path) == EXPECTED_06A_INDEX_SHA256
    ]
    if len(matching) != 1:
        raise RuntimeError(f"expected one exact 06a artifact root; found {len(matching)}")
    root = matching[0]
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
        raise RuntimeError(f"06a indexed member verification failed: {failures}")
    final_path = root / "final_report.json"
    if atomic._sha256_file(final_path) != EXPECTED_06A_FINAL_SHA256:
        raise RuntimeError("06a final report SHA-256 mismatch")
    final = json.loads(final_path.read_text(encoding="utf-8"))
    if (
        final.get("diagnosis") != "ATOMIC_STATE_UPDATE_NOT_YET_LEARNABLE"
        or final.get("technical_gate_passed") is not False
    ):
        raise RuntimeError("06a source is not the registered failed atomic pilot")
    return root, {
        "valid": True,
        "source_kind": "zip" if source.is_file() else "directory",
        "archive_sha256": archive_hash,
        "artifact_index_sha256": EXPECTED_06A_INDEX_SHA256,
        "final_report_sha256": EXPECTED_06A_FINAL_SHA256,
        "indexed_member_count": len(index.get("artifacts", [])),
        "diagnosis": final["diagnosis"],
        "technical_gate_passed": False,
    }


@dataclass(frozen=True)
class AtomicVoltagePathConfig(atomic.AtomicStateDynamicsConfig):
    training_steps: int = 1200
    short_training_steps: int = 300
    voltage_path_sample_indices: Tuple[int, ...] = (5, 10, 15, 20, 25, 30, 35, 40)
    expected_microtrace_sample_count: int = 41
    maximum_parameter_count: int = 7238
    minimum_factor_effect_fraction: float = 0.01
    plateau_relative_improvement_fraction: float = 0.002
    progress_interval: int = 100
    evaluation_interval: int = 50

    def validate(self) -> None:
        super().validate()
        if self.short_training_steps <= 0 or self.short_training_steps >= self.training_steps:
            raise ValueError("06a-b short budget must be positive and below long budget")
        samples = self.voltage_path_sample_indices
        if tuple(sorted(set(samples))) != samples or not samples:
            raise ValueError("06a-b voltage path indices must be unique and increasing")
        if samples[-1] != self.expected_microtrace_sample_count - 1:
            raise ValueError("06a-b voltage path must include the one-ms endpoint")
        if samples[0] <= 0 or self.expected_microtrace_sample_count < 3:
            raise ValueError("06a-b voltage path grid is invalid")
        if self.maximum_parameter_count <= 0:
            raise ValueError("06a-b parameter ceiling is invalid")
        if not 0 < self.minimum_factor_effect_fraction < 1:
            raise ValueError("06a-b factor-effect threshold is invalid")
        if not 0 < self.plateau_relative_improvement_fraction < 1:
            raise ValueError("06a-b plateau threshold is invalid")

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "AtomicVoltagePathConfig":
        payload = dict(values)
        for name in ("rollout_horizons_ms", "voltage_path_sample_indices"):
            if name in payload:
                payload[name] = tuple(map(int, payload[name]))
        result = cls(**payload)
        result.validate()
        return result


def voltage_path_features(
    teacher_trace: np.ndarray,
    voltage_t: np.ndarray,
    voltage_t_plus_1: np.ndarray,
    sample_indices: Sequence[int],
    mode: str,
) -> np.ndarray:
    """Return equally wide path deltas for the endpoint and teacher arms."""

    trace = np.asarray(teacher_trace, dtype=np.float32)
    start = np.asarray(voltage_t, dtype=np.float32)
    end = np.asarray(voltage_t_plus_1, dtype=np.float32)
    indices = np.asarray(sample_indices, dtype=np.int64)
    if trace.ndim != 3 or start.ndim != 2 or end.shape != start.shape:
        raise ValueError("06a-b voltage path arrays have incompatible shapes")
    if trace.shape[0] != start.shape[0] or trace.shape[2] != start.shape[1]:
        raise ValueError("06a-b voltage path trace does not match boundary voltage")
    if indices[0] <= 0 or indices[-1] >= trace.shape[1]:
        raise ValueError("06a-b voltage path sample index is outside the trace")
    if mode == "teacher_microtrace_path":
        return np.transpose(trace[:, indices, :], (0, 2, 1)) - start[:, :, None]
    if mode == "linear_endpoint_path":
        fraction = indices.astype(np.float32) / float(trace.shape[1] - 1)
        return (end - start)[:, :, None] * fraction[None, None, :]
    raise ValueError(f"unknown 06a-b voltage context {mode!r}")


if atomic.nn is not None:

    class VoltagePathMechanismStateUpdater(atomic.nn.Module):
        """Capacity-capped shared updater with a fixed-width voltage path."""

        def __init__(
            self,
            *,
            mechanism_count: int,
            variable_count: int,
            kind_count: int,
            region_count: int,
            static_width: int,
            drive_width: int,
            path_width: int,
            hidden_width: int,
            embedding_width: int,
            normalized_delta_limit: float,
        ) -> None:
            super().__init__()
            self.normalized_delta_limit = float(normalized_delta_limit)
            self.mechanism_embedding = atomic.nn.Embedding(mechanism_count, embedding_width)
            self.variable_embedding = atomic.nn.Embedding(variable_count, embedding_width)
            self.kind_embedding = atomic.nn.Embedding(kind_count, embedding_width)
            self.region_embedding = atomic.nn.Embedding(region_count, embedding_width)
            width = 2 + path_width + static_width + drive_width + 4 * embedding_width
            self.encoder = atomic.nn.Sequential(
                atomic.nn.Linear(width, hidden_width),
                atomic.nn.SiLU(),
                atomic.nn.Linear(hidden_width, hidden_width),
                atomic.nn.SiLU(),
            )
            self.proposal = atomic.nn.Linear(hidden_width, 1)
            self.relaxation = atomic.nn.Linear(hidden_width, 1)
            atomic.nn.init.zeros_(self.proposal.weight)
            atomic.nn.init.zeros_(self.proposal.bias)

        def forward(
            self,
            state_value: Any,
            voltage_t: Any,
            voltage_path: Any,
            drive: Any,
            static: Any,
            mechanism_id: Any,
            variable_id: Any,
            kind_id: Any,
            region_id: Any,
        ) -> Any:
            embedded = atomic.torch.cat(
                (
                    self.mechanism_embedding(mechanism_id),
                    self.variable_embedding(variable_id),
                    self.kind_embedding(kind_id),
                    self.region_embedding(region_id),
                ),
                dim=-1,
            )
            hidden = self.encoder(
                atomic.torch.cat(
                    (
                        state_value.unsqueeze(-1),
                        (voltage_t / 100.0).unsqueeze(-1),
                        voltage_path / 100.0,
                        drive,
                        static,
                        embedded,
                    ),
                    dim=-1,
                )
            )
            proposal = self.normalized_delta_limit * atomic.torch.tanh(
                self.proposal(hidden).squeeze(-1)
            )
            return atomic.torch.sigmoid(
                self.relaxation(hidden).squeeze(-1)
            ) * proposal

else:  # pragma: no cover

    class VoltagePathMechanismStateUpdater:
        def __init__(self, *_: Any, **__: Any) -> None:
            raise RuntimeError("06a-b requires PyTorch")


class AtomicVoltagePathIdentifiability(atomic.AtomicStateDynamicsPlayground):
    """End-to-end paired-factorial session for notebook 06a-b."""

    config: AtomicVoltagePathConfig

    def __init__(
        self,
        bundle: Any,
        output_dir: Path,
        config: AtomicVoltagePathConfig,
        artifact_05t_source: Path,
        artifact_06a_source: Path,
        *,
        code_revision: str,
    ) -> None:
        super().__init__(
            bundle, output_dir, config, artifact_05t_source, code_revision=code_revision
        )
        self.artifact_06a_source = Path(artifact_06a_source)
        self.factor_models: Dict[Tuple[str, str], Any] = {}
        self.microtrace_boundary_error_mv: Dict[str, float] = {}
        self.selected_hidden_width: Optional[int] = None
        self.selected_parameter_count: Optional[int] = None

    def _read_teacher_path(self, indices: Sequence[int], voltage_t: np.ndarray) -> np.ndarray:
        sample_indices = np.asarray(self.config.voltage_path_sample_indices, dtype=np.int64)
        output = np.empty(
            (len(indices), self.layout.segment_count, len(sample_indices)), dtype=np.float32
        )
        maximum_error = 0.0
        for row, index in enumerate(indices):
            trace = np.asarray(self.store.microtrace(int(index)), dtype=np.float32)
            if trace.shape != (
                self.config.expected_microtrace_sample_count,
                self.layout.segment_count,
            ):
                raise RuntimeError(f"06a-b unexpected microtrace shape {trace.shape}")
            maximum_error = max(
                maximum_error, float(np.max(np.abs(trace[0] - voltage_t[row])))
            )
            output[row] = trace[sample_indices].T - voltage_t[row, :, None]
        if maximum_error > 1e-4:
            raise RuntimeError(f"06a-b microtrace/start boundary mismatch {maximum_error}")
        return output

    def _materialize_role(self, role: str) -> Dict[str, np.ndarray]:
        values = super()._materialize_role(role)
        values["teacher_voltage_path"] = self._read_teacher_path(
            values["indices"], values["voltage_t"]
        )
        final_delta = values["teacher_voltage_path"][:, :, -1]
        expected = values["voltage_t1"] - values["voltage_t"]
        error = float(np.max(np.abs(final_delta - expected)))
        self.microtrace_boundary_error_mv[role] = error
        if error > 1e-4:
            raise RuntimeError(f"06a-b microtrace/end boundary mismatch in {role}: {error}")
        return values

    def prepare_voltage_path_forensic(self) -> Dict[str, Any]:
        _, source_06a = verified_06a_artifact_root(
            self.artifact_06a_source,
            self.output_dir.parent / ".06ab_artifact_cache" / "06a",
        )
        base = super().prepare_playground()
        probe = self._new_capacity_capped_model(atomic.torch.device("cpu"))
        parameter_count = int(sum(value.numel() for value in probe.parameters()))
        contract = {
            **base,
            "schema_version": "06a-b-voltage-path-contract-v1",
            "experiment": "atomic_voltage_path_identifiability",
            "source_06a": source_06a,
            "arms": list(VOLTAGE_CONTEXT_ARMS),
            "voltage_context_arms": list(VOLTAGE_CONTEXT_ARMS),
            "budget_steps": {
                "short": self.config.short_training_steps,
                "long": self.config.training_steps,
            },
            "voltage_path_sample_indices": list(
                self.config.voltage_path_sample_indices
            ),
            "voltage_path_offsets_ms": [
                index * 0.025 for index in self.config.voltage_path_sample_indices
            ],
            "teacher_voltage_context_is_diagnostic_only": True,
            "deployment_input_claimed": False,
            "checkpoint_selection_role": "train-derived calibration only",
            "architecture_selection_performed": False,
            "parameter_ceiling": self.config.maximum_parameter_count,
            "selected_hidden_width": self.selected_hidden_width,
            "parameter_count": parameter_count,
            "same_parameter_count_across_arms": True,
            "microtrace_boundary_error_mv": self.microtrace_boundary_error_mv,
            "rollout_windows_nested": True,
        }
        contract.pop("teacher_interval_voltage_is_diagnostic_only", None)
        atomic._write_json(self.output_dir / "voltage_path_contract.json", contract)
        return contract

    def _model_at_width(self, hidden_width: int, device: Any) -> Any:
        return VoltagePathMechanismStateUpdater(
            mechanism_count=len(self.layout.mechanism_names),
            variable_count=len(self.layout.variable_names),
            kind_count=len(self.layout.kind_names),
            region_count=len(self.layout.region_names),
            static_width=self.layout.segment_static.shape[1],
            drive_width=len(atomic.CAUSAL_DRIVE_FEATURES) + len(self.ion_feature_names),
            path_width=len(self.config.voltage_path_sample_indices),
            hidden_width=hidden_width,
            embedding_width=self.config.embedding_width,
            normalized_delta_limit=self.config.normalized_delta_limit,
        ).to(device)

    def _new_capacity_capped_model(self, device: Any) -> Any:
        if atomic.torch is None:
            raise RuntimeError("06a-b requires PyTorch")
        for width in range(self.config.hidden_width, 3, -1):
            model = self._model_at_width(width, device)
            count = int(sum(value.numel() for value in model.parameters()))
            if count <= self.config.maximum_parameter_count:
                if self.selected_hidden_width is not None and width != self.selected_hidden_width:
                    raise RuntimeError("06a-b capacity selection changed across arms")
                self.selected_hidden_width = width
                self.selected_parameter_count = count
                return model
        raise RuntimeError("06a-b could not satisfy the registered parameter ceiling")

    def _path_for_values(
        self, values: Mapping[str, np.ndarray], rows: np.ndarray, segments: np.ndarray, arm: str
    ) -> np.ndarray:
        if arm == "teacher_microtrace_path":
            return values["teacher_voltage_path"][rows, segments]
        if arm == "linear_endpoint_path":
            fraction = np.asarray(self.config.voltage_path_sample_indices, dtype=np.float32)
            fraction /= float(self.config.expected_microtrace_sample_count - 1)
            delta = values["voltage_t1"][rows, segments] - values["voltage_t"][rows, segments]
            return delta[:, None] * fraction[None, :]
        raise ValueError(arm)

    def _batch_path(
        self,
        values: Mapping[str, np.ndarray],
        transition_rows: np.ndarray,
        coordinates: np.ndarray,
        arm: str,
        device: Any,
    ) -> Tuple[Tuple[Any, ...], Any]:
        rows = np.repeat(transition_rows, coordinates.shape[1])
        cols = coordinates.reshape(-1)
        segments = self.coordinate["segment"][cols]
        tensor = lambda value, dtype=None: atomic.torch.as_tensor(
            value, dtype=dtype, device=device
        )
        inputs = (
            tensor(values["state"][rows, cols], atomic.torch.float32),
            tensor(values["voltage_t"][rows, segments], atomic.torch.float32),
            tensor(self._path_for_values(values, rows, segments, arm), atomic.torch.float32),
            tensor(values["context"][rows, segments], atomic.torch.float32),
            tensor(self.layout.segment_static[segments], atomic.torch.float32),
            tensor(self.coordinate["mechanism"][cols], atomic.torch.long),
            tensor(self.coordinate["variable"][cols], atomic.torch.long),
            tensor(self.coordinate["kind"][cols], atomic.torch.long),
            tensor(self.coordinate["region"][cols], atomic.torch.long),
        )
        return inputs, tensor(values["delta"][rows, cols], atomic.torch.float32)

    def _evaluate_one_step_path(
        self, model: Any, role: str, arm: str, device: Any
    ) -> Dict[str, Any]:
        values = self.materialized[role]
        group_count = len(self.coordinate_groups)
        group_sse = np.zeros(group_count, dtype=np.float64)
        group_persistence_sse = np.zeros(group_count, dtype=np.float64)
        group_examples = np.zeros(group_count, dtype=np.int64)
        squared_error = persistence_error = 0.0
        active_squared_error = active_persistence_error = 0.0
        count = active_count = 0
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
                coordinates = np.broadcast_to(cols, (len(values["indices"]), len(cols)))
                rows = np.arange(len(values["indices"]), dtype=np.int64)
                inputs, target = self._batch_path(values, rows, coordinates, arm, device)
                prediction = model(*inputs)
                error = prediction - target
                active = target.abs() >= self.config.active_delta_threshold
                error_np = error.detach().cpu().numpy().reshape(len(rows), len(cols))
                target_np = target.detach().cpu().numpy().reshape(len(rows), len(cols))
                per_coordinate_sse = np.sum(error_np * error_np, axis=0)
                per_coordinate_persistence = np.sum(target_np * target_np, axis=0)
                groups = self.coordinate["semantic_group"][cols]
                np.add.at(group_sse, groups, per_coordinate_sse)
                np.add.at(group_persistence_sse, groups, per_coordinate_persistence)
                np.add.at(group_examples, groups, len(rows))
                squared_error += float(np.sum(per_coordinate_sse))
                persistence_error += float(np.sum(per_coordinate_persistence))
                if bool(active.any()):
                    active_squared_error += float(atomic.torch.sum(error[active] ** 2).cpu())
                    active_persistence_error += float(atomic.torch.sum(target[active] ** 2).cpu())
                    active_count += int(active.sum().item())
                count += int(target.numel())
        rmse = math.sqrt(squared_error / max(count, 1))
        persistence = math.sqrt(persistence_error / max(count, 1))
        group_rmse = np.sqrt(group_sse / np.maximum(group_examples, 1))
        group_persistence = np.sqrt(
            group_persistence_sse / np.maximum(group_examples, 1)
        )
        semantic_names = sorted(
            {
                f"{row['mechanism']}|{row['variable']}|{row['kind']}"
                for row in self.mechanism_records
            }
        )
        if len(semantic_names) != group_count:
            raise RuntimeError("06a-b semantic group vocabulary changed during evaluation")
        semantic_groups = {
            name: {
                "normalized_delta_rmse": float(group_rmse[group]),
                "persistence_normalized_delta_rmse": float(
                    group_persistence[group]
                ),
                "improvement_vs_persistence_fraction": 1.0
                - float(group_rmse[group])
                / max(float(group_persistence[group]), 1e-12),
                "coordinate_example_count": int(group_examples[group]),
            }
            for group, name in enumerate(semantic_names)
        }
        macro_rmse = float(np.mean(group_rmse))
        macro_persistence = float(np.mean(group_persistence))
        active_rmse = math.sqrt(active_squared_error / max(active_count, 1))
        active_persistence = math.sqrt(active_persistence_error / max(active_count, 1))
        return {
            "normalized_delta_rmse": rmse,
            "persistence_normalized_delta_rmse": persistence,
            "improvement_vs_persistence_fraction": 1.0 - rmse / max(persistence, 1e-12),
            "semantic_macro_normalized_delta_rmse": macro_rmse,
            "semantic_macro_persistence_normalized_delta_rmse": macro_persistence,
            "semantic_macro_improvement_vs_persistence_fraction": 1.0
            - macro_rmse / max(macro_persistence, 1e-12),
            "active_normalized_delta_rmse": active_rmse,
            "active_persistence_normalized_delta_rmse": active_persistence,
            "active_improvement_vs_persistence_fraction": 1.0
            - active_rmse / max(active_persistence, 1e-12),
            "coordinate_example_count": count,
            "active_coordinate_example_count": active_count,
            "semantic_group_count": group_count,
            "semantic_groups": semantic_groups,
        }

    def _train_context(self, arm: str, device: Any) -> Dict[str, Any]:
        atomic.torch.manual_seed(self.config.pilot_seed)
        if atomic.torch.cuda.is_available():
            atomic.torch.cuda.manual_seed_all(self.config.pilot_seed)
        rng = np.random.default_rng(self.config.pilot_seed)
        model = self._new_capacity_capped_model(device)
        optimizer = atomic.torch.optim.AdamW(
            model.parameters(), lr=self.config.learning_rate, weight_decay=self.config.weight_decay
        )
        best_loss = {label: math.inf for label in BUDGET_LABELS}
        best_state: Dict[str, Optional[Dict[str, Any]]] = {
            label: None for label in BUDGET_LABELS
        }
        curve: List[Dict[str, Any]] = []
        fit = self.materialized["fit"]
        progress = atomic._CompactProgress(
            f"06a-b {arm}", self.config.training_steps, self.config.progress_interval
        )
        for step in range(1, self.config.training_steps + 1):
            model.train()
            rows = rng.integers(0, len(fit["indices"]), size=self.config.batch_transition_count)
            coordinates = self._sample_coordinates(
                rng, self.config.batch_transition_count * self.config.coordinates_per_transition
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
            evaluate = (
                step == 1
                or step % self.config.evaluation_interval == 0
                or step in (self.config.short_training_steps, self.config.training_steps)
            )
            if evaluate:
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
                eligible = ["long"]
                if step <= self.config.short_training_steps:
                    eligible.append("short")
                for label in eligible:
                    if score < best_loss[label]:
                        best_loss[label] = score
                        best_state[label] = {
                            key: value.detach().cpu().clone()
                            for key, value in model.state_dict().items()
                        }
            progress.update(step, f"loss={float(loss.detach().cpu()):.4g} cal={best_loss['long']:.4g}")

        budget_reports: Dict[str, Any] = {}
        for label in BUDGET_LABELS:
            if best_state[label] is None:
                raise RuntimeError(f"06a-b missing {label} checkpoint")
            selected = self._new_capacity_capped_model(device)
            selected.load_state_dict(best_state[label])
            self.factor_models[(arm, label)] = selected
            checkpoint = self.output_dir / f"{arm}_{label}.pt"
            atomic.torch.save(
                {
                    "state_dict": best_state[label],
                    "voltage_context": arm,
                    "budget": label,
                    "configuration": asdict(self.config),
                },
                checkpoint,
            )
            development = self._evaluate_one_step_path(
                selected, "development", arm, device
            )
            budget_reports[label] = {
                "maximum_optimizer_steps": (
                    self.config.short_training_steps
                    if label == "short"
                    else self.config.training_steps
                ),
                "best_calibration_normalized_delta_rmse": best_loss[label],
                "development": development,
                "checkpoint": checkpoint.name,
                "checkpoint_sha256": atomic._sha256_file(checkpoint),
            }

        late = [
            row
            for row in curve
            if row["step"] >= int(0.75 * self.config.training_steps)
        ]
        late_start = late[0]["calibration_normalized_delta_rmse"]
        late_best = min(row["calibration_normalized_delta_rmse"] for row in late)
        relative_late_gain = (late_start - late_best) / max(late_start, 1e-12)
        report = {
            "voltage_context": arm,
            "paired_seed": self.config.pilot_seed,
            "parameter_count": self.selected_parameter_count,
            "hidden_width": self.selected_hidden_width,
            "learning_curve": curve,
            "relative_calibration_improvement_last_quarter": relative_late_gain,
            "calibration_plateau_reached": relative_late_gain
            < self.config.plateau_relative_improvement_fraction,
            "budgets": budget_reports,
        }
        atomic._write_json(self.output_dir / f"factorial_{arm}.json", report)
        return report

    def run_factorial_pilot(self) -> Dict[str, Any]:
        device = atomic.torch.device(
            "cuda" if atomic.torch.cuda.is_available() else "cpu"
        )
        arms = {arm: self._train_context(arm, device) for arm in VOLTAGE_CONTEXT_ARMS}
        counts = {
            arms[arm]["parameter_count"] for arm in VOLTAGE_CONTEXT_ARMS
        }
        payload = {
            "schema_version": "06a-b-factorial-pilot-v1",
            "valid": len(counts) == 1 and next(iter(counts)) <= self.config.maximum_parameter_count,
            "device": str(device),
            "same_initialization_seed": True,
            "same_data_order": True,
            "same_optimizer": True,
            "same_parameter_count": len(counts) == 1,
            "parameter_count": next(iter(counts)),
            "checkpoint_selection_role": "train-derived calibration only",
            "validation_or_test_accessed": False,
            "arms": arms,
        }
        atomic._write_json(self.output_dir / "factorial_pilot.json", payload)
        return payload

    def _nested_development_windows(self) -> List[np.ndarray]:
        horizon = max(self.config.rollout_horizons_ms)
        allowed = {str(row["trajectory_id"]) for row in self.roles["development"]}
        windows: List[np.ndarray] = []
        for trajectory in sorted(allowed):
            indices = self.store.trajectory_indices[trajectory]
            for start in range(max(0, len(indices) - horizon + 1)):
                candidate = indices[start : start + horizon]
                steps = self.store.metadata["step_index"][candidate]
                if np.array_equal(steps, np.arange(steps[0], steps[0] + horizon)):
                    windows.append(candidate)
        windows = sorted(
            windows,
            key=lambda row: hashlib.sha256(
                f"{self.config.role_seed}|nested|max{horizon}|{','.join(map(str, row))}".encode()
            ).hexdigest(),
        )
        return windows[: self.config.rollout_windows_per_horizon]

    def _predict_full_delta_path(
        self,
        model: Any,
        normalized_state: np.ndarray,
        voltage_t: np.ndarray,
        path: np.ndarray,
        context: np.ndarray,
        device: Any,
    ) -> np.ndarray:
        batch = len(normalized_state)
        output = np.empty_like(normalized_state, dtype=np.float32)
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
                rows = np.repeat(np.arange(batch), len(cols))
                flat_cols = np.tile(cols, batch)
                segments = self.coordinate["segment"][flat_cols]
                tensor = lambda value, dtype=None: atomic.torch.as_tensor(
                    value, dtype=dtype, device=device
                )
                prediction = model(
                    tensor(normalized_state[rows, flat_cols], atomic.torch.float32),
                    tensor(voltage_t[rows, segments], atomic.torch.float32),
                    tensor(path[rows, segments], atomic.torch.float32),
                    tensor(context[rows, segments], atomic.torch.float32),
                    tensor(self.layout.segment_static[segments], atomic.torch.float32),
                    tensor(self.coordinate["mechanism"][flat_cols], atomic.torch.long),
                    tensor(self.coordinate["variable"][flat_cols], atomic.torch.long),
                    tensor(self.coordinate["kind"][flat_cols], atomic.torch.long),
                    tensor(self.coordinate["region"][flat_cols], atomic.torch.long),
                ).reshape(batch, len(cols))
                output[:, start:stop] = prediction.cpu().numpy()
        return output

    def evaluate_nested_rollouts(self) -> Dict[str, Any]:
        expected = {
            (arm, budget) for arm in VOLTAGE_CONTEXT_ARMS for budget in BUDGET_LABELS
        }
        if set(self.factor_models) != expected:
            raise RuntimeError("run_factorial_pilot must precede rollout evaluation")
        windows = self._nested_development_windows()
        if not windows:
            raise RuntimeError("06a-b found no common maximum-horizon windows")
        window_digest = hashlib.sha256(
            json.dumps([list(map(int, row)) for row in windows], separators=(",", ":")).encode()
        ).hexdigest()
        initial_indices = np.asarray([row[0] for row in windows], dtype=np.int64)
        initial = atomic.mechanism_logit(
            self.store.read_state(initial_indices, "t", categories=("mechanism_states",))
        ).astype(np.float32)
        device = next(iter(self.factor_models.values())).proposal.weight.device
        arms: Dict[str, Any] = {}
        for arm in VOLTAGE_CONTEXT_ARMS:
            arms[arm] = {}
            for budget in BUDGET_LABELS:
                model = self.factor_models[(arm, budget)]
                transformed = initial.copy()
                horizon_report: Dict[str, Any] = {}
                for step in range(max(self.config.rollout_horizons_ms)):
                    indices = np.asarray([row[step] for row in windows], dtype=np.int64)
                    voltage_t = self.store.read_state(
                        indices, "t", categories=("voltage",)
                    ).astype(np.float32)
                    voltage_t1 = self.store.read_state(
                        indices, "t_plus_1", categories=("voltage",)
                    ).astype(np.float32)
                    if arm == "teacher_microtrace_path":
                        path = self._read_teacher_path(indices, voltage_t)
                    else:
                        fraction = np.asarray(
                            self.config.voltage_path_sample_indices, dtype=np.float32
                        ) / float(self.config.expected_microtrace_sample_count - 1)
                        path = (voltage_t1 - voltage_t)[:, :, None] * fraction[None, None, :]
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
                        model, normalized.astype(np.float32), voltage_t, path, context, device
                    )
                    transformed += delta * self.statistics["delta_scale"]
                    horizon = step + 1
                    if horizon in self.config.rollout_horizons_ms:
                        targets = np.asarray([row[step] for row in windows], dtype=np.int64)
                        target = atomic.mechanism_logit(
                            self.store.read_state(
                                targets, "t_plus_1", categories=("mechanism_states",)
                            )
                        ).astype(np.float32)
                        error = (transformed - target) / self.statistics["state_scale"]
                        persistence_error = (initial - target) / self.statistics["state_scale"]
                        rmse = float(np.sqrt(np.mean(error * error)))
                        persistence = float(np.sqrt(np.mean(persistence_error * persistence_error)))
                        raw = atomic.inverse_mechanism_logit(transformed)
                        horizon_report[f"{horizon}_ms"] = {
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
                arms[arm][budget] = horizon_report
        payload = {
            "schema_version": "06a-b-nested-rollouts-v1",
            "valid": all(
                row["nonfinite_count"] == 0 and row["domain_violation_count"] == 0
                for arm in arms.values()
                for budget in arm.values()
                for row in budget.values()
            ),
            "common_maximum_horizon_window_count": len(windows),
            "common_window_set_sha256": window_digest,
            "all_horizons_are_prefixes_of_same_windows": True,
            "state_rollout_is_recursive": True,
            "membrane_voltage_path_is_teacher_forced": True,
            "validation_or_test_accessed": False,
            "arms": arms,
        }
        atomic._write_json(self.output_dir / "nested_rollouts.json", payload)
        return payload

    def _plot_factorial(
        self, pilot: Mapping[str, Any], rollout: Mapping[str, Any]
    ) -> List[str]:
        import matplotlib.pyplot as plt

        figure_dir = self.output_dir / "figures"
        figure_dir.mkdir(exist_ok=True)
        figure, axes = plt.subplots(1, 2, figsize=(12, 4))
        for arm in VOLTAGE_CONTEXT_ARMS:
            curve = pilot["arms"][arm]["learning_curve"]
            axes[0].plot(
                [row["step"] for row in curve],
                [row["calibration_normalized_delta_rmse"] for row in curve],
                label=arm,
            )
            long_rollout = rollout["arms"][arm]["long"]
            axes[1].plot(
                list(self.config.rollout_horizons_ms),
                [
                    long_rollout[f"{horizon}_ms"]["improvement_vs_persistence_fraction"]
                    for horizon in self.config.rollout_horizons_ms
                ],
                marker="o",
                label=arm,
            )
        axes[0].axvline(
            self.config.short_training_steps,
            color="black",
            linestyle="--",
            linewidth=1,
            label="short budget",
        )
        axes[0].set(xlabel="optimizer step", ylabel="calibration normalized-delta RMSE")
        axes[1].axhline(0.0, color="black", linewidth=1)
        axes[1].set(
            xlabel="nested recursive-state horizon (ms)",
            ylabel="improvement vs persistence",
            xticks=list(self.config.rollout_horizons_ms),
        )
        for axis in axes:
            axis.grid(alpha=0.25)
            axis.legend(fontsize=8)
        figure.tight_layout()
        path = figure_dir / "atomic_voltage_path_factorial.png"
        figure.savefig(path, dpi=160)
        plt.close(figure)
        return [str(path.relative_to(self.output_dir))]

    def finalize_voltage_path_forensic(
        self, pilot: Mapping[str, Any], rollout: Mapping[str, Any]
    ) -> Dict[str, Any]:
        gains = {
            arm: {
                budget: pilot["arms"][arm]["budgets"][budget]["development"][
                    "improvement_vs_persistence_fraction"
                ]
                for budget in BUDGET_LABELS
            }
            for arm in VOLTAGE_CONTEXT_ARMS
        }
        endpoint_budget_effect = (
            gains["linear_endpoint_path"]["long"]
            - gains["linear_endpoint_path"]["short"]
        )
        teacher_budget_effect = (
            gains["teacher_microtrace_path"]["long"]
            - gains["teacher_microtrace_path"]["short"]
        )
        path_effect = (
            gains["teacher_microtrace_path"]["long"]
            - gains["linear_endpoint_path"]["long"]
        )
        gate = self.config.minimum_pilot_improvement_fraction
        factor = self.config.minimum_factor_effect_fraction
        path_identified = (
            gains["teacher_microtrace_path"]["long"] >= gate and path_effect >= factor
        )
        optimization_identified = (
            max(gains[arm]["long"] for arm in VOLTAGE_CONTEXT_ARMS) >= gate
            and max(endpoint_budget_effect, teacher_budget_effect) >= factor
        )
        if path_identified and optimization_identified:
            diagnosis = "ATOMIC_STATE_REQUIRES_VOLTAGE_PATH_AND_LONGER_OPTIMIZATION"
            next_step = "06b_voltage_substep_optimized_state_updater_canary"
        elif path_identified:
            diagnosis = "ATOMIC_STATE_REQUIRES_INTRA_MS_VOLTAGE_PATH"
            next_step = "06b_voltage_substep_state_updater_canary"
        elif optimization_identified:
            diagnosis = "ATOMIC_STATE_WAS_OPTIMIZATION_LIMITED"
            next_step = "06b_optimized_explicit_state_updater_canary"
        elif max(gains[arm]["long"] for arm in VOLTAGE_CONTEXT_ARMS) >= gate:
            diagnosis = "ATOMIC_STATE_LEARNABLE_BUT_FACTOR_UNRESOLVED"
            next_step = "06a_c_atomic_factor_replication"
        else:
            diagnosis = "ATOMIC_STATE_REMAINS_UNIDENTIFIABLE"
            next_step = "06a_c_local_state_context_forensics"
        figures = self._plot_factorial(pilot, rollout)
        final = {
            "schema_version": "06a-b-final-report-v1",
            "valid": bool(pilot.get("valid") and rollout.get("valid")),
            "decision_grade": False,
            "diagnosis": diagnosis,
            "architecture_family": "HayFlow-ESI",
            "one_step_gain_vs_persistence": gains,
            "factor_effects": {
                "endpoint_optimization_budget_effect": endpoint_budget_effect,
                "teacher_path_optimization_budget_effect": teacher_budget_effect,
                "teacher_path_over_linear_endpoint_effect": path_effect,
                "minimum_registered_factor_effect": factor,
            },
            "technical_gate": gate,
            "path_information_identified": path_identified,
            "optimization_budget_identified": optimization_identified,
            "calibration_plateau": {
                arm: pilot["arms"][arm]["calibration_plateau_reached"]
                for arm in VOLTAGE_CONTEXT_ARMS
            },
            "same_parameter_count_across_arms": pilot["same_parameter_count"],
            "parameter_count": pilot["parameter_count"],
            "parameter_ceiling": self.config.maximum_parameter_count,
            "rollout_windows_nested": rollout[
                "all_horizons_are_prefixes_of_same_windows"
            ],
            "state_and_outcome_splits_read": ["train"],
            "teacher_microtrace_is_diagnostic_only": True,
            "deployment_input_claimed": False,
            "architecture_selection_performed": False,
            "validation_state_accessed": False,
            "test_state_accessed": False,
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
                "schema_version": "06a-b-artifact-index-v1",
                "artifact_count": len(records),
                "artifacts": records,
            },
        )


__all__ = [
    "EXPECTED_06A_ARCHIVE_SHA256",
    "EXPECTED_06A_INDEX_SHA256",
    "EXPECTED_06A_FINAL_SHA256",
    "VOLTAGE_CONTEXT_ARMS",
    "BUDGET_LABELS",
    "AtomicVoltagePathConfig",
    "AtomicVoltagePathIdentifiability",
    "VoltagePathMechanismStateUpdater",
    "verified_06a_artifact_root",
    "voltage_path_features",
]
