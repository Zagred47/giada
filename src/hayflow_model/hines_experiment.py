"""Notebook-05 runner for the recurrent HayFlow-Hines prototype.

The runner is deliberately fail-fast.  A balanced overfit canary is executed
before any generalisation experiment; if neither HayFlow-Hines nor the
conventional ConvGRU control can memorise it, the expensive curriculum is not
started and the saved report classifies the failure as data/target/input/code.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from ..hayflow_data.composite_flowmap import (
    EVENT_KINDS,
    CompositeFlowmapBundle,
    CompositeTransitionStore,
    classify_regime,
)
from ..hayflow_data.hines_inputs import (
    canonical_anchor_segment_ids,
    encode_realized_synaptic_drive,
    explicit_teacher_views,
)
from ..hayflow_data.reconditioned_flowmap import (
    ReconditionedStateNormalizer,
    ReconditioningConfig,
)
from ..hayflow_eval.flowmap_metrics import write_parquet
from ..hayflow_eval.release_flowmap_metrics import (
    branching_metrics,
    pooled_event_metrics,
    voltage_fidelity_rows,
)
from .hayflow_hines import (
    HayFlowHines,
    HayFlowHinesConfig,
    OrderedSegmentConvGRU,
    hayflow_hines_arrays,
    model_parameter_count,
)
from .hines_layer import DifferentiableHinesSolve, require_torch


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    def safe(value: Any) -> Any:
        if isinstance(value, Mapping):
            return {str(key): safe(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [safe(item) for item in value]
        if isinstance(value, np.ndarray):
            return safe(value.tolist())
        if isinstance(value, np.generic):
            value = value.item()
        if isinstance(value, float) and not math.isfinite(value):
            return None
        return value

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(safe(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _fingerprint(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[2],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


class Progress:
    def __init__(self, label: str, total: int) -> None:
        self.label = label
        self.total = max(1, int(total))
        self.started = time.monotonic()

    def update(self, value: int, detail: str = "") -> None:
        elapsed = time.monotonic() - self.started
        rate = value / elapsed if value and elapsed else 0.0
        eta = (self.total - value) / rate if rate else math.inf
        eta_text = "?" if not math.isfinite(eta) else f"{eta / 60:.1f} min"
        print(
            f"[HayFlow 05][{self.label}] {value}/{self.total} "
            f"({100.0 * value / self.total:.1f}%) ETA {eta_text} {detail}",
            flush=True,
        )


@dataclass(frozen=True)
class HinesPrototypeExperimentConfig:
    profile: str = "diagnostic_full"
    seeds: Tuple[int, ...] = (17, 29, 43)
    batch_size: int = 2
    sequence_batch_size: int = 2
    learning_rate: float = 2e-4
    canary_learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    gradient_clip_norm: float = 1.0
    canary_epochs: int = 300
    canary_patience: int = 80
    canary_voltage_epochs: int = 120
    canary_event_epochs: int = 120
    canary_joint_epochs: int = 60
    canary_evaluation_interval: int = 10
    canary_episode_per_class: int = 2
    canary_max_transitions: int = 128
    phase1_epochs: int = 35
    phase2_epochs: int = 18
    phase3_epochs: int = 12
    phase1_transitions_per_episode: int = 4
    rollout_windows_per_epoch: int = 48
    evaluation_transitions: int = 2048
    evaluation_windows_per_regime: int = 8
    normalization_transitions: int = 2048
    lambda_voltage: float = 1.0
    lambda_peak: float = 0.25
    lambda_event: float = 1.0
    lambda_timing: float = 0.05
    lambda_state: float = 0.10
    lambda_rollout: float = 0.50
    lambda_branching: float = 0.25
    lambda_privileged: float = 0.01
    lambda_microtrace: float = 0.02
    canary_voltage_rmse_mv: float = 1.0
    canary_event_f1: float = 0.90
    canary_peak_error_mv: float = 5.0
    canary_branching_retention: float = 0.50
    rollout_horizons_ms: Tuple[int, ...] = (2, 4, 8, 16, 32)
    model: HayFlowHinesConfig = field(default_factory=HayFlowHinesConfig)

    def effective(self) -> "HinesPrototypeExperimentConfig":
        if self.profile not in {"smoke", "diagnostic_full"}:
            raise ValueError("profile must be smoke or diagnostic_full")
        self.model.validate()
        if min(
            self.canary_voltage_epochs,
            self.canary_event_epochs,
            self.canary_joint_epochs,
            self.canary_evaluation_interval,
            self.canary_episode_per_class,
        ) <= 0:
            raise ValueError("canary stage lengths, interval and support must be positive")
        if self.profile == "diagnostic_full" and (
            self.canary_voltage_epochs
            + self.canary_event_epochs
            + self.canary_joint_epochs
            != self.canary_epochs
        ):
            raise ValueError("canary stage epochs must sum to canary_epochs")
        if self.profile == "diagnostic_full":
            return self
        values = asdict(self)
        values.update(
            seeds=(self.seeds[0],),
            canary_epochs=5,
            canary_patience=5,
            canary_voltage_epochs=2,
            canary_event_epochs=2,
            canary_joint_epochs=1,
            canary_evaluation_interval=1,
            canary_episode_per_class=1,
            canary_max_transitions=24,
            phase1_epochs=2,
            phase2_epochs=1,
            phase3_epochs=1,
            rollout_windows_per_epoch=4,
            evaluation_transitions=64,
            evaluation_windows_per_regime=1,
            normalization_transitions=64,
            rollout_horizons_ms=(2, 4),
        )
        values["model"] = HayFlowHinesConfig(**values["model"])
        return HinesPrototypeExperimentConfig(**values)

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "HinesPrototypeExperimentConfig":
        payload = dict(values)
        model = payload.pop("model", {})
        for name in ("seeds", "rollout_horizons_ms"):
            if name in payload:
                payload[name] = tuple(int(item) for item in payload[name])
        payload["model"] = HayFlowHinesConfig(**model)
        return cls(**payload).effective()


class HayFlowHinesExperiment:
    """End-to-end canary-first experiment over the immutable composite store."""

    def __init__(
        self,
        bundle: CompositeFlowmapBundle,
        output_dir: Path,
        config: HinesPrototypeExperimentConfig,
        *,
        b3_report: Optional[Mapping[str, Any]] = None,
    ) -> None:
        self.bundle = bundle
        self.output_dir = Path(output_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.figure_dir = self.output_dir / "figures"
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.figure_dir.mkdir(exist_ok=True)
        self.config = config.effective()
        self.store = CompositeTransitionStore(bundle)
        self.layout = self.store.layout
        self.arrays = hayflow_hines_arrays(self.layout)
        self.anchors = canonical_anchor_segment_ids(self.layout)
        self.code_commit = _git_commit()
        self.normalizer: Optional[ReconditionedStateNormalizer] = None
        self.privileged_center: Optional[np.ndarray] = None
        self.privileged_scale: Optional[np.ndarray] = None
        self.b3_report = dict(b3_report or {})
        self.rows: Dict[str, List[Dict[str, Any]]] = {
            "one_step": [], "rollout": [], "events": [], "branching": [],
            "recovery": [], "drift": [], "attenuation": [], "comparison": [],
            "ood": [],
        }
        self.train_voltage_min: Optional[np.ndarray] = None
        self.train_voltage_max: Optional[np.ndarray] = None
        self.train_branch_pair: Optional[Tuple[int, int]] = None
        self.training_contract_blockers: List[str] = []
        self.registry: Dict[str, Dict[str, Any]] = {}

    def close(self) -> None:
        self.store.close()

    def _episode_id(self, index: int) -> str:
        trajectory = str(self.store.metadata["trajectory_id"][int(index)])
        row = self.store.episode_by_trajectory.get(trajectory, {})
        return str(row.get("episode_id", trajectory))

    def _regime(self, index: int) -> str:
        trajectory = str(self.store.metadata["trajectory_id"][int(index)])
        row = self.store.episode_by_trajectory.get(trajectory, {})
        return classify_regime(row, self.store.metadata["category"][int(index)])

    def _stratified_indices(
        self,
        split: str,
        *,
        seed: int,
        per_episode: Optional[int],
        limit: int = 0,
    ) -> np.ndarray:
        rng = np.random.default_rng(seed)
        groups: Dict[str, List[np.ndarray]] = {}
        accepted = (
            set(self.store.split_indices) - {"train", "validation", "test"}
            if split == "test" else {split}
        )
        for trajectory, indices in self.store.trajectory_indices.items():
            if str(self.store.metadata["split"][indices[0]]) not in accepted:
                continue
            episode = self.store.episode_by_trajectory.get(trajectory, {})
            regime = classify_regime(episode, self.store.metadata["category"][indices[0]])
            groups.setdefault(regime, []).append(indices)
        selected: List[int] = []
        for regime in sorted(groups):
            trajectories = groups[regime]
            order = rng.permutation(len(trajectories))
            for position in order:
                indices = trajectories[int(position)]
                if per_episode and len(indices) > per_episode:
                    picks = np.linspace(0, len(indices) - 1, per_episode, dtype=int)
                    indices = indices[picks]
                selected.extend(int(value) for value in indices)
        values = np.asarray(sorted(set(selected)), dtype=np.int64)
        if limit and len(values) > limit:
            values = np.sort(rng.choice(values, size=limit, replace=False))
        return values

    def prepare(self) -> Dict[str, Any]:
        loader = self.store.report()
        _write_json(self.output_dir / "composite_loader_report.json", loader)
        sample = self._stratified_indices(
            "train", seed=501, per_episode=4, limit=self.config.normalization_transitions
        )
        raw_t = self.store.read_state(sample, "t")
        raw_t1 = self.store.read_state(sample, "t_plus_1")
        voltage_sample = np.concatenate(
            [raw_t[:, : self.layout.segment_count], raw_t1[:, : self.layout.segment_count]],
            axis=0,
        )
        self.train_voltage_min = voltage_sample.min(0).astype(np.float32)
        self.train_voltage_max = voltage_sample.max(0).astype(np.float32)
        self.normalizer = ReconditionedStateNormalizer(
            self.layout,
            ReconditioningConfig(
                activity_epsilon=1e-9,
                sparse_update_fraction=0.1,
                minimum_scale=1e-8,
                gate_transform="logit",
            ),
        ).fit(raw_t, raw_t1)
        selected_privileged = self.arrays["selected_privileged_indices"]
        if len(selected_privileged):
            privileged = self.store.read_privileged(sample)[:, selected_privileged]
            self.privileged_center = privileged.mean(0).astype(np.float32)
            self.privileged_scale = np.maximum(privileged.std(0), 1e-8).astype(np.float32)
        else:
            self.privileged_center = np.zeros(0, dtype=np.float32)
            self.privileged_scale = np.ones(0, dtype=np.float32)
        self.train_branch_pair = self._find_counterfactual_pair(("train",))
        self.training_contract_blockers = []
        if self.train_branch_pair is None:
            self.training_contract_blockers.append(
                "no numerically identical S_t pair with different U_realized exists in "
                "the training split; branching supervision cannot be used without test leakage"
            )
        normalization = {
            "schema_version": "05-normalization-v1",
            "fit_split": "train",
            "fit_transition_count": len(sample),
            "sample_sha256": hashlib.sha256(sample.tobytes()).hexdigest(),
            "state": self.normalizer.to_dict(),
            "selected_privileged_center": self.privileged_center,
            "selected_privileged_scale": self.privileged_scale,
            "voltage_training_min_mv": self.train_voltage_min,
            "voltage_training_max_mv": self.train_voltage_max,
        }
        _write_json(self.output_dir / "normalization_schema.json", normalization)
        model_config = {
            "schema_version": "05-model-config-v1",
            "architecture": "HayFlow-Hines recurrent morphology-aware prototype",
            "forward_path_reuses_B3": False,
            "primary_input": "U_realized",
            "ablation": ["H0", "H1", "H2"],
            "conventional_control": "fixed-segment-order ConvGRU",
            "experiment": asdict(self.config),
            "selected_core_target_count": int(len(self.arrays["selected_core_indices"])),
            "selected_privileged_target_count": int(len(self.arrays["selected_privileged_indices"])),
            "training_branch_pair": (
                list(self.train_branch_pair) if self.train_branch_pair is not None else None
            ),
            "training_contract_blockers": self.training_contract_blockers,
            "not_implemented": [
                "morphology reduction", "massive dataset", "S4", "Mamba",
                "custom CUDA kernels", "mixed precision", "state pruning",
                "aggressive compression",
            ],
        }
        _write_json(self.output_dir / "model_configurations.json", model_config)
        return {
            "loader": loader,
            "normalizer_fingerprint": self.normalizer.fingerprint(),
            "anchor_segment_ids": self.anchors.tolist(),
            "selected_core_target_count": int(len(self.arrays["selected_core_indices"])),
            "selected_privileged_target_count": int(len(self.arrays["selected_privileged_indices"])),
            "training_branch_pair": (
                list(self.train_branch_pair) if self.train_branch_pair is not None else None
            ),
            "training_contract_blockers": self.training_contract_blockers,
        }

    def run_hines_layer_tests(self) -> Dict[str, Any]:
        require_torch()
        import torch

        cases = []
        parents = [0, 0, 0, 1, 1, 2]
        solver = DifferentiableHinesSolve(parents).double()
        diagonal = torch.tensor([[2.2, 2.1, 1.8, 1.2, 1.1, 1.5]], dtype=torch.double)
        coupling = torch.tensor([[0.0, 0.7, 0.4, 0.3, 0.2, 0.5]], dtype=torch.double)
        rhs = torch.tensor([[1.0, -0.2, 0.5, 0.9, -0.7, 0.1]], dtype=torch.double)
        actual, diagnostics = solver(diagonal, coupling, rhs, return_diagnostics=True)
        expected = torch.linalg.solve(solver.dense_matrix(diagonal, coupling), rhs.unsqueeze(-1)).squeeze(-1)
        dense_error = float((actual - expected).abs().max())
        cases.append({"name": "synthetic_dense_identity", "maximum_error": dense_error, "passed": dense_error < 1e-10})
        d = diagonal.clone().requires_grad_(True)
        g = coupling.clone().requires_grad_(True)
        b = rhs.clone().requires_grad_(True)
        gradcheck = bool(torch.autograd.gradcheck(lambda x, y, z: solver(x, y, z), (d, g, b), eps=1e-6, atol=1e-5, rtol=1e-4))
        cases.append({"name": "autograd_gradcheck", "passed": gradcheck})
        physical_solver = DifferentiableHinesSolve(self.arrays["parent_ids"])
        mass = 1000.0 * torch.as_tensor(self.arrays["capacitance_uf"], dtype=torch.float32)
        physical_diagonal = mass + torch.as_tensor(self.arrays["leak_conductance_us"] + self.arrays["axial_total_us"], dtype=torch.float32)
        resting = torch.full((1, self.layout.segment_count), -76.0)
        physical_rhs = mass.unsqueeze(0) * resting + torch.as_tensor(self.arrays["leak_conductance_us"] * self.arrays["leak_reversal_mv"], dtype=torch.float32).unsqueeze(0)
        physical, physical_diagnostics = physical_solver(
            physical_diagonal.unsqueeze(0),
            torch.as_tensor(self.arrays["axial_conductance_to_parent_us"], dtype=torch.float32).unsqueeze(0),
            physical_rhs,
            return_diagnostics=True,
        )
        physical_safe = bool(torch.isfinite(physical).all() and physical_diagnostics["positive_diagonal"] and physical_diagnostics["well_conditioned"])
        cases.append({"name": "canonical_morphology_safe", "passed": physical_safe, "diagnostics": physical_diagnostics})
        report = {
            "schema_version": "05-hines-tests-v1",
            "valid": all(row["passed"] for row in cases),
            "cases": cases,
            "synthetic_diagnostics": diagnostics,
        }
        _write_json(self.output_dir / "hines_layer_tests.json", report)
        if not report["valid"]:
            raise RuntimeError(f"Hines layer tests failed: {report}")
        return report

    def _event_amplitude_targets(self, indices: Sequence[int]) -> Tuple[np.ndarray, np.ndarray]:
        amplitude = np.zeros((len(indices), len(EVENT_KINDS)), dtype=np.float32)
        segment = np.zeros((len(indices), len(EVENT_KINDS)), dtype=np.int64)
        lookup = {name: index for index, name in enumerate(EVENT_KINDS)}
        for row, logical_index in enumerate(indices):
            for event in self.store.events(int(logical_index)):
                kind = str(event.get("kind", ""))
                if kind not in lookup:
                    continue
                column = lookup[kind]
                amplitude[row, column] = float(event.get("amplitude_mv", event.get("amplitude", 0.0)) or 0.0)
                segment[row, column] = int(event.get("segment_id", event.get("peak_segment_id", 0)) or 0)
        return amplitude, segment

    def _microtrace_targets(self, indices: Sequence[int]) -> Tuple[np.ndarray, np.ndarray]:
        output = np.zeros((len(indices), 5, 41), dtype=np.float32)
        mask = np.zeros(len(indices), dtype=bool)
        for row, logical_index in enumerate(indices):
            try:
                values = np.asarray(self.store.microtrace(int(logical_index)), dtype=np.float32)
            except (KeyError, OSError, ValueError):
                continue
            if values.ndim != 2:
                continue
            if values.shape[0] == self.layout.segment_count:
                selected = values[self.anchors]
            elif values.shape[1] == self.layout.segment_count:
                selected = values[:, self.anchors].T
            else:
                continue
            grid = np.linspace(0, selected.shape[1] - 1, 41)
            source = np.arange(selected.shape[1])
            output[row] = np.stack([np.interp(grid, source, trace) for trace in selected])
            mask[row] = True
        return output, mask

    def _batch(self, indices: Sequence[int], *, include_targets: bool = True, include_microtrace: bool = False) -> Dict[str, Any]:
        if self.normalizer is None:
            raise RuntimeError("prepare() must be called first")
        indices = np.asarray(indices, dtype=np.int64)
        raw_t = self.store.read_state(indices, "t")
        normalized_t = self.normalizer.normalize_state(raw_t).astype(np.float32)
        voltage_t = raw_t[:, : self.layout.segment_count].astype(np.float32)
        calcium_t, synapse_t = explicit_teacher_views(
            normalized_t,
            self.arrays,
            segment_count=self.layout.segment_count,
            calcium_dim=self.config.model.calcium_state_dim,
            synapse_dim=self.config.model.synapse_state_dim,
        )
        result: Dict[str, Any] = {
            "indices": indices,
            "teacher_state_t": normalized_t,
            "raw_state_t": raw_t,
            "voltage_t": voltage_t,
            "calcium_t": calcium_t,
            "synapse_state_t": synapse_t,
            "anchor_voltage_t": voltage_t[:, self.anchors],
            "anchor_segment_ids": self.anchors,
        }
        result.update(
            encode_realized_synaptic_drive(
                self.store, indices, voltage_t,
                dt_ms=self.config.model.dt_ms,
                raw_state_t=raw_t,
            )
        )
        result.update(self.store.event_targets(indices))
        amplitude, segment = self._event_amplitude_targets(indices)
        result.update(event_amplitude=amplitude, event_segment=segment)
        if include_targets:
            raw_t1 = self.store.read_state(indices, "t_plus_1")
            normalized_t1 = self.normalizer.normalize_state(raw_t1).astype(np.float32)
            normalized_delta, state_activity = self.normalizer.delta_and_activity(
                raw_t, raw_t1
            )
            calcium_t1, synapse_t1 = explicit_teacher_views(
                normalized_t1,
                self.arrays,
                segment_count=self.layout.segment_count,
                calcium_dim=self.config.model.calcium_state_dim,
                synapse_dim=self.config.model.synapse_state_dim,
            )
            selected_core = self.arrays["selected_core_indices"]
            selected_sparse = self.normalizer.sparse_mask[selected_core]
            selected_state_mask = (
                ~selected_sparse.reshape(1, -1)
                | state_activity[:, selected_core]
            )
            selected_privileged = self.arrays["selected_privileged_indices"]
            privileged = self.store.read_privileged(indices)[:, selected_privileged].astype(np.float32)
            if len(selected_privileged):
                privileged = (privileged - self.privileged_center) / self.privileged_scale
            result.update(
                raw_state_t_plus_1=raw_t1,
                voltage_target=raw_t1[:, : self.layout.segment_count].astype(np.float32),
                calcium_target=calcium_t1,
                synapse_state_target=synapse_t1,
                # The decoder predicts the 1 ms transition, not the absolute
                # transformed state.  Sparse variables contribute regression
                # loss only when they actually change.
                selected_state_target=normalized_delta[:, selected_core],
                selected_state_mask=selected_state_mask,
                selected_privileged_target=privileged,
            )
            if include_microtrace:
                trace, trace_mask = self._microtrace_targets(indices)
                result.update(probe_microtrace_target=trace, probe_microtrace_mask=trace_mask)
        return result

    @staticmethod
    def _torch_batch(raw: Mapping[str, Any], device: Any) -> Dict[str, Any]:
        import torch

        integer = {"indices", "event_region", "event_segment", "anchor_segment_ids"}
        boolean = {
            "event_timing_mask", "event_region_mask", "probe_microtrace_mask",
            "selected_state_mask",
        }
        excluded = {"raw_state_t", "raw_state_t_plus_1"}
        result: Dict[str, Any] = {}
        for key, value in raw.items():
            if key in excluded:
                result[key] = value
            elif isinstance(value, np.ndarray):
                dtype = torch.long if key in integer else torch.bool if key in boolean else torch.float32
                result[key] = torch.as_tensor(value, dtype=dtype, device=device)
            else:
                result[key] = value
        return result

    def _models(self, device: Any) -> Dict[str, Any]:
        metadata = self.layout.to_model_metadata()
        return {
            "H0": HayFlowHines(self.config.model, metadata, self.arrays).to(device),
            "H1": HayFlowHines(self.config.model, metadata, self.arrays).to(device),
            "H2": HayFlowHines(self.config.model, metadata, self.arrays).to(device),
            "ConvGRU": OrderedSegmentConvGRU(self.config.model, metadata, self.arrays).to(device),
        }

    @staticmethod
    def _f1(probability: np.ndarray, target: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        prediction = probability >= threshold
        truth = target > 0.5
        tp = (prediction & truth).sum(0)
        fp = (prediction & ~truth).sum(0)
        fn = (~prediction & truth).sum(0)
        return 2.0 * tp / np.maximum(1, 2 * tp + fp + fn)

    def _loss(
        self,
        output: Mapping[str, Any],
        batch: Mapping[str, Any],
        *,
        event_pos_weight: Any,
        include_privileged: bool,
        rollout_factor: float = 0.0,
        objective_weights: Optional[Mapping[str, float]] = None,
    ) -> Tuple[Any, Dict[str, float]]:
        import torch
        import torch.nn.functional as F

        voltage_error = (output["voltage"] - batch["voltage_target"]) / 10.0
        voltage = F.smooth_l1_loss(voltage_error, torch.zeros_like(voltage_error))
        peak = F.smooth_l1_loss(
            output["voltage"].amax(1) / 10.0,
            batch["voltage_target"].amax(1) / 10.0,
        )
        event = F.binary_cross_entropy_with_logits(
            output["event_logits"], batch["event_presence"], pos_weight=event_pos_weight
        )
        timing_mask = batch["event_timing_mask"]
        timing = F.l1_loss(
            output["event_timing"][timing_mask], batch["event_timing"][timing_mask]
        ) if bool(timing_mask.any()) else voltage.new_zeros(())
        region_mask = batch["event_region_mask"]
        region = F.cross_entropy(
            output["event_region_logits"][region_mask], batch["event_region"][region_mask]
        ) if bool(region_mask.any()) else voltage.new_zeros(())
        segment_mask = batch["event_presence"] > 0.5
        segment = F.cross_entropy(
            output["event_segment_logits"][segment_mask],
            batch["event_segment"][segment_mask],
        ) if bool(segment_mask.any()) else voltage.new_zeros(())
        amplitude_mask = batch["event_presence"] > 0.5
        amplitude = F.smooth_l1_loss(
            output["event_amplitude"][amplitude_mask] / 20.0,
            batch["event_amplitude"][amplitude_mask] / 20.0,
        ) if bool(amplitude_mask.any()) else voltage.new_zeros(())
        state = voltage.new_zeros(())
        if "selected_state" in output:
            state_mask = batch.get("selected_state_mask")
            if state_mask is None or bool(state_mask.any()):
                prediction = output["selected_state"]
                target = batch["selected_state_target"]
                if state_mask is not None:
                    prediction = prediction[state_mask]
                    target = target[state_mask]
                state = state + F.smooth_l1_loss(prediction, target)
            state = state + F.smooth_l1_loss(output["calcium"], batch["calcium_target"])
            state = state + 0.25 * F.smooth_l1_loss(output["synapse"], batch["synapse_state_target"])
        privileged = voltage.new_zeros(())
        if include_privileged and "selected_privileged" in output and output["selected_privileged"].numel():
            privileged = F.smooth_l1_loss(output["selected_privileged"], batch["selected_privileged_target"])
        microtrace = voltage.new_zeros(())
        if include_privileged and "probe_microtrace" in output and "probe_microtrace_target" in batch:
            mask = batch["probe_microtrace_mask"]
            if bool(mask.any()):
                microtrace = F.smooth_l1_loss(
                    output["probe_microtrace"][mask] / 10.0,
                    batch["probe_microtrace_target"][mask] / 10.0,
                )
        weights = {
            "voltage": self.config.lambda_voltage,
            "peak": self.config.lambda_peak,
            "event": self.config.lambda_event,
            "timing": self.config.lambda_timing,
            "state": self.config.lambda_state,
            "privileged": self.config.lambda_privileged,
            "microtrace": self.config.lambda_microtrace,
        }
        if objective_weights is not None:
            weights.update({key: float(value) for key, value in objective_weights.items()})
        total = (
            weights["voltage"] * voltage
            + weights["peak"] * peak
            + weights["event"] * event
            + weights["timing"] * (
                timing + 0.25 * region + 0.25 * segment + 0.25 * amplitude
            )
            + weights["state"] * state
            + weights["privileged"] * privileged
            + weights["microtrace"] * microtrace
        )
        total = total * (1.0 + self.config.lambda_rollout * float(rollout_factor))
        return total, {
            "voltage": float(voltage.detach()), "peak": float(peak.detach()),
            "event": float(event.detach()), "timing": float(timing.detach()),
            "region": float(region.detach()), "segment": float(segment.detach()),
            "amplitude": float(amplitude.detach()), "state": float(state.detach()),
            "privileged": float(privileged.detach()),
            "microtrace": float(microtrace.detach()),
        }

    def _canary_indices(self) -> Tuple[np.ndarray, Optional[Tuple[int, int]]]:
        selected: List[int] = []
        for kind in EVENT_KINDS:
            episodes = self.store.episode_indices(split="train", event_kind=kind)
            for indices in episodes[: self.config.canary_episode_per_class]:
                targets = self.store.event_targets(indices)["event_presence"]
                column = EVENT_KINDS.index(kind)
                positives = np.flatnonzero(targets[:, column] > 0.5)
                centers = positives[:4] if len(positives) else np.asarray([len(indices) // 2])
                for center in centers:
                    lo = max(0, int(center) - 2)
                    hi = min(len(indices), int(center) + 3)
                    selected.extend(int(value) for value in indices[lo:hi])
        hard_rows = [
            row for row in self.store.episode_rows
            if str(row.get("split")) == "train"
            and str(row.get("hard_negative_for", "")) not in {"", "[]", "nan"}
        ]
        for row in hard_rows[:6]:
            indices = self.store.trajectory_indices[str(row["trajectory_id"])]
            selected.extend(int(value) for value in indices[: min(5, len(indices))])
        values = np.asarray(sorted(set(selected)), dtype=np.int64)
        if len(values) > self.config.canary_max_transitions:
            event_presence = self.store.event_targets(values)["event_presence"].sum(1)
            order = np.argsort(-event_presence, kind="stable")
            values = np.sort(values[order[: self.config.canary_max_transitions]])
        branch_pair = self.train_branch_pair
        if branch_pair is None:
            # A held-out pair is acceptable only for the disposable canary.  It
            # never reaches the weights used by the full experiment.
            branch_pair = self._find_counterfactual_pair(("release_identifiability_test",))
        if branch_pair is not None:
            values = np.asarray(sorted(set(values.tolist() + list(branch_pair))), dtype=np.int64)
        return values, branch_pair

    def _find_counterfactual_pair(
        self, splits: Sequence[str] = ("train", "release_identifiability_test")
    ) -> Optional[Tuple[int, int]]:
        groups: Dict[Tuple[str, int], List[int]] = {}
        for split in splits:
            for indices in self.store.episode_indices(split=split):
                trajectory = str(self.store.metadata["trajectory_id"][indices[0]])
                episode = self.store.episode_by_trajectory.get(trajectory, {})
                snapshot = str(episode.get("snapshot_id", episode.get("snapshot_source", "")))
                if not snapshot:
                    continue
                for index in indices:
                    groups.setdefault((snapshot, int(self.store.metadata["step_index"][index])), []).append(int(index))
        for candidates in groups.values():
            for left_position, left in enumerate(candidates):
                for right in candidates[left_position + 1 :]:
                    state_error = np.max(np.abs(self.store.read_state([left], "t") - self.store.read_state([right], "t")))
                    if state_error > 1e-5:
                        continue
                    left_actions = json.dumps(self.store.actions(left, "U_realized"), sort_keys=True)
                    right_actions = json.dumps(self.store.actions(right, "U_realized"), sort_keys=True)
                    if left_actions != right_actions:
                        return left, right
        # Snapshot identifiers are provenance, not part of S_t.  Independent
        # episodes can therefore form a legitimate training counterfactual
        # when the *complete explicit boundary state* is numerically identical.
        # Search the first few macro-steps in a bounded, vectorised pass and
        # verify the full state after bucketing by voltage.
        accepted = set(splits)
        by_local_step: Dict[int, List[int]] = {}
        for trajectory, indices in self.store.trajectory_indices.items():
            if not len(indices):
                continue
            split = str(self.store.metadata["split"][indices[0]])
            if split not in accepted:
                continue
            for local_step, index in enumerate(indices[:16]):
                by_local_step.setdefault(local_step, []).append(int(index))
        for local_step in sorted(by_local_step):
            indices = np.asarray(by_local_step[local_step], dtype=np.int64)
            if len(indices) < 2:
                continue
            states = self.store.read_state(indices, "t")
            voltage = np.round(states[:, : self.layout.segment_count], decimals=5)
            buckets: Dict[bytes, List[int]] = {}
            for position, row in enumerate(voltage):
                signature = hashlib.sha256(
                    np.ascontiguousarray(row, dtype=np.float32).tobytes()
                ).digest()
                buckets.setdefault(signature, []).append(position)
            for positions in buckets.values():
                if len(positions) < 2:
                    continue
                for left_position, left_offset in enumerate(positions):
                    left = int(indices[left_offset])
                    left_actions = json.dumps(
                        self.store.actions(left, "U_realized"), sort_keys=True
                    )
                    for right_offset in positions[left_position + 1 :]:
                        right = int(indices[right_offset])
                        if np.max(np.abs(states[left_offset] - states[right_offset])) > 1e-5:
                            continue
                        right_actions = json.dumps(
                            self.store.actions(right, "U_realized"), sort_keys=True
                        )
                        if left_actions != right_actions:
                            return left, right
        return None

    def _event_pos_weight(self, indices: Sequence[int], device: Any) -> Any:
        import torch

        target = self.store.event_targets(indices)["event_presence"]
        positive = target.sum(0)
        negative = len(target) - positive
        weight = np.clip(negative / np.maximum(positive, 1.0), 1.0, 100.0)
        return torch.as_tensor(weight, dtype=torch.float32, device=device)

    def _branch_loss(self, model: Any, pair: Optional[Tuple[int, int]], device: Any, ablation: str) -> Any:
        import torch

        if pair is None:
            return next(model.parameters()).new_zeros(())
        raw = self._batch(pair, include_targets=True)
        batch = self._torch_batch(raw, device)
        output = model(batch, ablation=ablation, decode_teacher=False)
        predicted = torch.sqrt(torch.mean((output["voltage"][0] - output["voltage"][1]) ** 2) + 1e-12)
        teacher = torch.sqrt(torch.mean((batch["voltage_target"][0] - batch["voltage_target"][1]) ** 2) + 1e-12)
        event_distance = torch.mean(torch.abs(torch.sigmoid(output["event_logits"][0]) - torch.sigmoid(output["event_logits"][1])))
        target_events = torch.mean(torch.abs(batch["event_presence"][0] - batch["event_presence"][1]))
        relative_voltage = torch.abs(predicted - teacher) / teacher.detach().clamp_min(1e-3)
        return relative_voltage + torch.abs(event_distance - target_events)

    def _canary_metrics(self, model: Any, indices: np.ndarray, pair: Optional[Tuple[int, int]], device: Any, ablation: str) -> Dict[str, Any]:
        import torch

        model.eval()
        raw = self._batch(indices, include_targets=True)
        prediction = self._predict_one_step(model, indices, device, ablation)
        error = prediction["voltage"] - raw["voltage_target"]
        probability = prediction["event_probability"]
        f1 = self._f1(probability, raw["event_presence"])
        present = raw["event_presence"].sum(0) > 0
        teacher_peak = raw["voltage_target"].max(1)
        predicted_peak = prediction["voltage"].max(1)
        retention = math.nan
        if pair is not None:
            pair_raw = self._batch(pair, include_targets=True)
            pair_batch = self._torch_batch(pair_raw, device)
            with torch.no_grad():
                pair_output = model(pair_batch, ablation=ablation, decode_teacher=False)
            teacher_distance = float(np.sqrt(np.mean((pair_raw["voltage_target"][0] - pair_raw["voltage_target"][1]) ** 2)))
            predicted_values = pair_output["voltage"].cpu().numpy()
            predicted_distance = float(np.sqrt(np.mean((predicted_values[0] - predicted_values[1]) ** 2)))
            retention = predicted_distance / max(teacher_distance, 1e-8)
        return {
            "voltage_rmse_mv": float(np.sqrt(np.mean(error ** 2))),
            "event_f1_by_class": {kind: float(f1[index]) for index, kind in enumerate(EVENT_KINDS)},
            "minimum_present_event_f1": float(f1[present].min()) if present.any() else 0.0,
            "maximum_peak_error_mv": float(np.max(np.abs(predicted_peak - teacher_peak))),
            "branching_retention": retention,
        }

    def _canary_selection_score(self, metrics: Mapping[str, float]) -> float:
        """Metric-aligned checkpoint score; lower is better."""

        voltage = float(metrics["voltage_rmse_mv"]) / self.config.canary_voltage_rmse_mv
        peak = float(metrics["maximum_peak_error_mv"]) / self.config.canary_peak_error_mv
        f1 = float(metrics["minimum_present_event_f1"])
        branching = float(metrics["branching_retention"])
        f1_deficit = max(0.0, self.config.canary_event_f1 - f1) / self.config.canary_event_f1
        branch_deficit = (
            max(0.0, self.config.canary_branching_retention - branching)
            / self.config.canary_branching_retention
            if math.isfinite(branching) else 2.0
        )
        return voltage + peak + 2.0 * f1_deficit + 2.0 * branch_deficit

    @staticmethod
    def _canary_target_audit(raw: Mapping[str, Any]) -> Dict[str, Any]:
        voltage_delta = np.asarray(raw["voltage_target"]) - np.asarray(raw["voltage_t"])
        state = np.asarray(raw["selected_state_target"])
        state_mask = np.asarray(raw["selected_state_mask"], dtype=bool)
        selected = np.abs(state[state_mask]) if state_mask.any() else np.zeros(1)
        return {
            "voltage_delta_absolute_mv": {
                "p50": float(np.percentile(np.abs(voltage_delta), 50)),
                "p95": float(np.percentile(np.abs(voltage_delta), 95)),
                "p99": float(np.percentile(np.abs(voltage_delta), 99)),
                "maximum": float(np.max(np.abs(voltage_delta))),
            },
            "selected_state_delta_normalized_absolute": {
                "p50": float(np.percentile(selected, 50)),
                "p95": float(np.percentile(selected, 95)),
                "p99": float(np.percentile(selected, 99)),
                "maximum": float(np.max(selected)),
                "regression_mask_fraction": float(state_mask.mean()),
            },
            "contract": (
                "selected biological targets are normalized 1 ms deltas; "
                "sparse targets are regressed only when active"
            ),
        }

    def _train_canary_model(self, name: str, model: Any, indices: np.ndarray, pair: Optional[Tuple[int, int]], device: Any) -> Dict[str, Any]:
        import torch

        ablation = "H2" if name == "HayFlow-Hines-H2" else "H2"
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=self.config.canary_learning_rate, weight_decay=0.0
        )
        positive_weight = self._event_pos_weight(indices, device)
        stages = (
            (
                "voltage_peak",
                self.config.canary_voltage_epochs,
                {
                    "voltage": 4.0, "peak": 2.0, "event": 0.0,
                    "timing": 0.0, "state": 0.0,
                    "privileged": 0.0, "microtrace": 0.0,
                },
                0.0,
            ),
            (
                "events_branching",
                self.config.canary_event_epochs,
                {
                    "voltage": 2.0, "peak": 1.0, "event": 1.0,
                    "timing": self.config.lambda_timing, "state": 0.0,
                    "privileged": 0.0, "microtrace": 0.0,
                },
                self.config.lambda_branching,
            ),
            (
                "joint_with_biological_deltas",
                self.config.canary_joint_epochs,
                {
                    "voltage": 2.0, "peak": 1.0, "event": 1.0,
                    "timing": self.config.lambda_timing,
                    "state": min(self.config.lambda_state, 0.01),
                    "privileged": 0.0, "microtrace": 0.0,
                },
                self.config.lambda_branching,
            ),
        )
        total_epochs = sum(int(row[1]) for row in stages)
        progress = Progress(f"canary-v2 {name}", total_epochs)
        best_score = math.inf
        best_state = None
        best_epoch = None
        best_stage = None
        history = []
        rng = np.random.default_rng(9103)
        # Materialise the deliberately tiny canary once.  Re-reading 17,220
        # state variables and JSON event records for every epoch would turn an
        # overfit unit test into an I/O benchmark.
        cached_raw = self._batch(indices, include_targets=True, include_microtrace=False)
        target_audit = self._canary_target_audit(cached_raw)
        index_position = {int(value): position for position, value in enumerate(indices)}
        completed = 0
        stop = False
        for stage, stage_epochs, objective_weights, branch_weight in stages:
            for stage_epoch in range(int(stage_epochs)):
                order = indices.copy()
                rng.shuffle(order)
                model.train()
                losses: List[float] = []
                gradients: List[float] = []
                components: Dict[str, List[float]] = {}
                for start in range(0, len(order), self.config.batch_size):
                    selected = order[start : start + self.config.batch_size]
                    positions = np.asarray(
                        [index_position[int(value)] for value in selected],
                        dtype=np.int64,
                    )
                    raw = {}
                    for key, value in cached_raw.items():
                        if (
                            isinstance(value, np.ndarray) and value.ndim
                            and value.shape[0] == len(indices)
                            and key != "anchor_segment_ids"
                        ):
                            raw[key] = value[positions]
                        else:
                            raw[key] = value
                    batch = self._torch_batch(raw, device)
                    optimizer.zero_grad(set_to_none=True)
                    output = model(
                        batch, ablation=ablation,
                        decode_teacher=name.startswith("HayFlow"),
                    )
                    loss, terms = self._loss(
                        output, batch, event_pos_weight=positive_weight,
                        include_privileged=False,
                        objective_weights=objective_weights,
                    )
                    loss.backward()
                    gradient = torch.nn.utils.clip_grad_norm_(
                        model.parameters(), self.config.gradient_clip_norm
                    )
                    optimizer.step()
                    losses.append(float(loss.detach()))
                    gradients.append(float(gradient.detach()))
                    for key, value in terms.items():
                        components.setdefault(key, []).append(float(value))
                branch_value = 0.0
                if pair is not None and branch_weight > 0.0:
                    optimizer.zero_grad(set_to_none=True)
                    branch = branch_weight * self._branch_loss(
                        model, pair, device, ablation
                    )
                    branch.backward()
                    gradient = torch.nn.utils.clip_grad_norm_(
                        model.parameters(), self.config.gradient_clip_norm
                    )
                    optimizer.step()
                    branch_value = float(branch.detach())
                    gradients.append(float(gradient.detach()))
                completed += 1
                loss_value = float(np.mean(losses)) if losses else math.nan
                row: Dict[str, Any] = {
                    "epoch": completed - 1,
                    "stage": stage,
                    "stage_epoch": stage_epoch,
                    "loss": loss_value,
                    "branching_loss_weighted": branch_value,
                    "gradient_norm_pre_clip": float(np.mean(gradients)),
                }
                row.update(
                    {f"loss_{key}": float(np.mean(value)) for key, value in components.items()}
                )
                evaluate = (
                    completed == 1
                    or completed % self.config.canary_evaluation_interval == 0
                    or stage_epoch + 1 == int(stage_epochs)
                )
                if evaluate:
                    metrics = self._canary_metrics(model, indices, pair, device, ablation)
                    selection_score = self._canary_selection_score(metrics)
                    row.update(
                        selection_score=selection_score,
                        voltage_rmse_mv=metrics["voltage_rmse_mv"],
                        minimum_present_event_f1=metrics["minimum_present_event_f1"],
                        maximum_peak_error_mv=metrics["maximum_peak_error_mv"],
                        branching_retention=metrics["branching_retention"],
                    )
                    if selection_score < best_score:
                        best_score = selection_score
                        best_epoch = completed - 1
                        best_stage = stage
                        best_state = {
                            key: value.detach().cpu().clone()
                            for key, value in model.state_dict().items()
                        }
                    progress.update(
                        completed,
                        f"stage={stage} loss={loss_value:.4g} "
                        f"V={metrics['voltage_rmse_mv']:.3g} "
                        f"peak={metrics['maximum_peak_error_mv']:.3g} "
                        f"F1min={metrics['minimum_present_event_f1']:.3f} "
                        f"branch={metrics['branching_retention']:.3f}",
                    )
                    if stage == "joint_with_biological_deltas" and (
                        metrics["voltage_rmse_mv"] < self.config.canary_voltage_rmse_mv
                        and metrics["minimum_present_event_f1"] > self.config.canary_event_f1
                        and metrics["maximum_peak_error_mv"] < self.config.canary_peak_error_mv
                        and metrics["branching_retention"] > self.config.canary_branching_retention
                    ):
                        stop = True
                history.append(row)
                if stop:
                    break
            if stop:
                break
        if best_state is not None:
            model.load_state_dict(best_state)
        metrics = self._canary_metrics(model, indices, pair, device, ablation)
        metrics["passed"] = bool(
            metrics["voltage_rmse_mv"] < self.config.canary_voltage_rmse_mv
            and metrics["minimum_present_event_f1"] > self.config.canary_event_f1
            and metrics["maximum_peak_error_mv"] < self.config.canary_peak_error_mv
            and metrics["branching_retention"] > self.config.canary_branching_retention
        )
        metrics["epochs_completed"] = len(history)
        metrics["best_checkpoint_epoch"] = best_epoch
        metrics["best_checkpoint_stage"] = best_stage
        metrics["best_checkpoint_selection_score"] = best_score
        metrics["target_audit"] = target_audit
        metrics["history"] = history
        return metrics

    def run_canary(self) -> Dict[str, Any]:
        require_torch()
        import torch

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        indices, pair = self._canary_indices()
        if not len(indices):
            raise RuntimeError("the balanced overfit canary is empty")
        models = self._models(device)
        reports = {
            "HayFlow-Hines-H2": self._train_canary_model(
                "HayFlow-Hines-H2", models["H2"], indices, pair, device
            ),
            "ConvGRU": self._train_canary_model(
                "ConvGRU", models["ConvGRU"], indices, pair, device
            ),
        }
        hines = reports["HayFlow-Hines-H2"]["passed"]
        recurrent = reports["ConvGRU"]["passed"]
        if hines:
            scenario = "A_HAYFLOW_HINES_SHOWS_LIFE"
        elif recurrent:
            scenario = "B_CONVENTIONAL_RECURRENCE_ONLY"
        else:
            scenario = "C_NEITHER_MODEL_OVERFITS_CANARY"
        report = {
            "schema_version": "05b-canary-v2",
            "valid": True,
            "scenario": scenario,
            "proceed_to_full_training": bool(hines),
            "transition_count": int(len(indices)),
            "logical_indices_sha256": hashlib.sha256(indices.tobytes()).hexdigest(),
            "event_support": dict(zip(EVENT_KINDS, self.store.event_targets(indices)["event_presence"].sum(0).astype(int).tolist())),
            "branch_pair": list(pair) if pair else None,
            "curriculum": [
                {"stage": "voltage_peak", "epochs": self.config.canary_voltage_epochs},
                {"stage": "events_branching", "epochs": self.config.canary_event_epochs},
                {
                    "stage": "joint_with_biological_deltas",
                    "epochs": self.config.canary_joint_epochs,
                },
            ],
            "criteria": {
                "voltage_rmse_mv_below": self.config.canary_voltage_rmse_mv,
                "minimum_present_event_f1_above": self.config.canary_event_f1,
                "maximum_peak_error_mv_below": self.config.canary_peak_error_mv,
                "branching_retention_above": self.config.canary_branching_retention,
            },
            "models": reports,
        }
        _write_json(self.output_dir / "canary_overfit_report.json", report)
        torch.save(
            {name: model.state_dict() for name, model in models.items() if name in {"H2", "ConvGRU"}},
            self.checkpoint_dir / "canary_models.pt",
        )
        return report

    def _iter_batches(self, values: Sequence[Any], size: int) -> Iterable[Sequence[Any]]:
        for start in range(0, len(values), int(size)):
            yield values[start : start + int(size)]

    def _balanced_epoch_indices(self, seed: int) -> np.ndarray:
        return self._stratified_indices(
            "train", seed=seed,
            per_episode=self.config.phase1_transitions_per_episode,
        )

    def _stratified_rollout_windows(self, split: str, horizon: int, *, seed: int, per_regime: int) -> List[np.ndarray]:
        rng = np.random.default_rng(seed)
        groups: Dict[str, List[np.ndarray]] = {}
        accepted = (
            set(self.store.split_indices) - {"train", "validation", "test"}
            if split == "test" else {split}
        )
        for trajectory, indices in self.store.trajectory_indices.items():
            if str(self.store.metadata["split"][indices[0]]) not in accepted or len(indices) < horizon:
                continue
            episode = self.store.episode_by_trajectory.get(trajectory, {})
            regime = classify_regime(episode, self.store.metadata["category"][indices[0]])
            starts = np.arange(len(indices) - horizon + 1)
            rng.shuffle(starts)
            for start in starts[: max(1, per_regime)]:
                window = indices[int(start) : int(start) + horizon]
                steps = self.store.metadata["step_index"][window]
                if np.array_equal(steps, np.arange(steps[0], steps[0] + horizon)):
                    groups.setdefault(regime, []).append(window)
        selected = []
        for regime in sorted(groups):
            values = groups[regime]
            rng.shuffle(values)
            selected.extend(values[:per_regime])
        return selected

    def _train_sequence_batch(
        self,
        model: Any,
        windows: Sequence[np.ndarray],
        optimizer: Any,
        device: Any,
        positive_weight: Any,
        ablation: str,
        *,
        include_privileged: bool,
    ) -> float:
        import torch

        recurrent = None
        total = None
        horizon = len(windows[0])
        for step in range(horizon):
            indices = [int(window[step]) for window in windows]
            raw = self._batch(indices, include_targets=True, include_microtrace=include_privileged and step == 0)
            batch = self._torch_batch(raw, device)
            output = model(
                batch, recurrent=recurrent, ablation=ablation,
                decode_teacher=include_privileged and step == 0,
            )
            recurrent = {
                key: output[key] for key in ("voltage", "local", "global", "calcium", "synapse")
            }
            loss, _ = self._loss(
                output, batch, event_pos_weight=positive_weight,
                include_privileged=include_privileged and step == 0,
                rollout_factor=(step + 1) / horizon,
            )
            total = loss if total is None else total + loss
        optimizer.zero_grad(set_to_none=True)
        total = total / horizon
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.gradient_clip_norm)
        optimizer.step()
        return float(total.detach())

    def _checkpoint_contract(self, name: str, seed: int) -> Dict[str, Any]:
        contract = {
            "schema_version": "05-checkpoint-v1",
            "dataset_fingerprint": self.bundle.fingerprint,
            "normalizer_fingerprint": self.normalizer.fingerprint(),
            "code_commit": self.code_commit,
            "model": name,
            "seed": int(seed),
            "config": asdict(self.config),
        }
        contract["fingerprint"] = _fingerprint(contract)
        return contract

    def train_full_model(self, name: str, model: Any, seed: int, device: Any) -> Any:
        import torch

        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        ablation = name if name in {"H0", "H1", "H2"} else "H2"
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=self.config.learning_rate, weight_decay=self.config.weight_decay
        )
        train_indices = self._balanced_epoch_indices(seed)
        positive_weight = self._event_pos_weight(train_indices, device)
        train_branch_pair = self.train_branch_pair
        contract = self._checkpoint_contract(name, seed)
        run_dir = self.checkpoint_dir / f"{name}-seed{seed}" / contract["fingerprint"][:16]
        run_dir.mkdir(parents=True, exist_ok=True)
        _write_json(run_dir / "fingerprint.json", contract)
        history = []
        phases = [
            ("one_step", self.config.phase1_epochs, (1,)),
            ("rollout_2_4_8", self.config.phase2_epochs, (2, 4, 8)),
            ("rollout_16_32", self.config.phase3_epochs, (16, 32)),
        ]
        total_epochs = sum(row[1] for row in phases)
        progress = Progress(f"{name}-seed{seed}", total_epochs)
        completed = 0
        for phase, epochs, horizons in phases:
            for epoch in range(epochs):
                model.train()
                losses = []
                if horizons == (1,):
                    indices = self._balanced_epoch_indices(seed + 1000 * completed)
                    np.random.default_rng(seed + completed).shuffle(indices)
                    for batch_indices in self._iter_batches(indices, self.config.batch_size):
                        windows = [np.asarray([int(index)]) for index in batch_indices]
                        losses.append(self._train_sequence_batch(
                            model, windows, optimizer, device, positive_weight, ablation,
                            include_privileged=name != "ConvGRU",
                        ))
                else:
                    windows = []
                    per = max(1, self.config.rollout_windows_per_epoch // len(horizons))
                    for horizon in horizons:
                        windows.extend(self._stratified_rollout_windows(
                            "train", horizon, seed=seed + completed + horizon, per_regime=per
                        ))
                    np.random.default_rng(seed + completed).shuffle(windows)
                    windows = windows[: self.config.rollout_windows_per_epoch]
                    for group in self._iter_batches(windows, self.config.sequence_batch_size):
                        # Same-length windows are required for a batched recurrent pass.
                        by_length: Dict[int, List[np.ndarray]] = {}
                        for window in group:
                            by_length.setdefault(len(window), []).append(window)
                        for same_length in by_length.values():
                            losses.append(self._train_sequence_batch(
                                model, same_length, optimizer, device, positive_weight,
                                ablation, include_privileged=False,
                            ))
                if train_branch_pair is not None:
                    optimizer.zero_grad(set_to_none=True)
                    branch_loss = self._branch_loss(
                        model, train_branch_pair, device, ablation
                    )
                    weighted_branch = self.config.lambda_branching * branch_loss
                    weighted_branch.backward()
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), self.config.gradient_clip_norm
                    )
                    optimizer.step()
                    losses.append(float(weighted_branch.detach()))
                completed += 1
                mean_loss = float(np.mean(losses)) if losses else math.nan
                history.append({"phase": phase, "epoch": epoch, "loss": mean_loss})
                progress.update(completed, f"phase={phase} loss={mean_loss:.4g}")
            torch.save(
                {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "contract": contract, "history": history},
                run_dir / f"{phase}.pt",
            )
            if phase == "one_step":
                torch.save(
                    {"model": model.state_dict(), "contract": contract, "history": history, "selection": "event-head checkpoint after balanced one-step phase"},
                    run_dir / "event.pt",
                )
        self.registry[f"{name}-seed{seed}"] = {
            "fingerprint": contract["fingerprint"],
            "parameter_count": model_parameter_count(model),
            "checkpoints": {
                phase: str((run_dir / f"{phase}.pt").relative_to(self.output_dir))
                for phase, _, _ in phases
            },
            "history": history,
        }
        self.registry[f"{name}-seed{seed}"]["checkpoints"]["event"] = str(
            (run_dir / "event.pt").relative_to(self.output_dir)
        )
        return model

    def _predict_one_step(
        self,
        model: Any,
        indices: np.ndarray,
        device: Any,
        ablation: str,
        *,
        progress_label: Optional[str] = None,
    ) -> Dict[str, np.ndarray]:
        import torch

        parts = {"voltage": [], "event_probability": [], "event_timing": [], "event_region": []}
        progress = Progress(progress_label, len(indices)) if progress_label else None
        next_report = max(1, len(indices) // 20)
        model.eval()
        with torch.no_grad():
            for batch_indices in self._iter_batches(indices, self.config.batch_size):
                raw = self._batch(batch_indices, include_targets=False)
                batch = self._torch_batch(raw, device)
                output = model(batch, ablation=ablation, decode_teacher=False)
                parts["voltage"].append(output["voltage"].cpu().numpy())
                parts["event_probability"].append(torch.sigmoid(output["event_logits"]).cpu().numpy())
                parts["event_timing"].append(output["event_timing"].cpu().numpy())
                parts["event_region"].append(output["event_region_logits"].argmax(-1).cpu().numpy())
                completed = sum(len(values) for values in parts["voltage"])
                if progress and (completed == len(indices) or completed % next_report < len(batch_indices)):
                    progress.update(completed)
        return {key: np.concatenate(values) for key, values in parts.items()}

    def _calibrate_thresholds(self, probability: np.ndarray, target: np.ndarray) -> np.ndarray:
        result = np.full(len(EVENT_KINDS), 0.5, dtype=float)
        for column in range(len(EVENT_KINDS)):
            best = (-1.0, 0.5)
            for threshold in np.linspace(0.05, 0.95, 19):
                f1 = self._f1(probability[:, [column]], target[:, [column]], threshold)[0]
                if f1 > best[0]:
                    best = (float(f1), float(threshold))
            result[column] = best[1]
        return result

    def evaluate_one_step(self, model: Any, name: str, seed: int, device: Any) -> np.ndarray:
        ablation = name if name in {"H0", "H1", "H2"} else "H2"
        validation = self._stratified_indices("validation", seed=811, per_episode=4, limit=self.config.evaluation_transitions)
        validation_prediction = self._predict_one_step(
            model, validation, device, ablation,
            progress_label=f"evaluate {name}-seed{seed} validation one-step",
        )
        thresholds = self._calibrate_thresholds(validation_prediction["event_probability"], self.store.event_targets(validation)["event_presence"])
        for split in ("validation", "test"):
            indices = self._stratified_indices(split, seed=812, per_episode=None, limit=self.config.evaluation_transitions)
            prediction = self._predict_one_step(
                model, indices, device, ablation,
                progress_label=f"evaluate {name}-seed{seed} {split} one-step",
            )
            target = self.store.read_state(indices, "t_plus_1")
            voltage_rows = voltage_fidelity_rows(
                prediction["voltage"], target[:, : self.layout.segment_count],
                model=f"{name}-seed{seed}", split=split, horizon_ms=1,
                segment_regions=[str(row["region"]) for row in self.layout.segments],
                regimes=[self._regime(int(index)) for index in indices],
            )
            for row in voltage_rows:
                row.update(seed=seed, architecture=name, input_view="U_realized")
                self.rows["one_step"].append(row)
                self.rows["drift"].append(dict(row))
                self.rows["attenuation"].append(dict(row))
            below = prediction["voltage"] < self.train_voltage_min[None, :]
            above = prediction["voltage"] > self.train_voltage_max[None, :]
            self.rows["ood"].append({
                "model": f"{name}-seed{seed}", "seed": seed,
                "architecture": name, "split": split, "horizon_ms": 1,
                "voltage_out_of_training_range_fraction": float(np.mean(below | above)),
                "voltage_below_physical_floor_fraction": float(np.mean(prediction["voltage"] < -150.0)),
                "voltage_above_physical_ceiling_fraction": float(np.mean(prediction["voltage"] > 100.0)),
            })
            targets = self.store.event_targets(indices)
            event_rows = pooled_event_metrics(
                prediction["event_probability"], targets["event_presence"],
                [self._episode_id(int(index)) for index in indices],
                model=f"{name}-seed{seed}", split=split, thresholds=thresholds,
                timing_prediction=prediction["event_timing"], timing_target=targets["event_timing"],
                timing_mask=targets["event_timing_mask"], region_prediction=prediction["event_region"],
                region_target=targets["event_region"], region_mask=targets["event_region_mask"],
            )
            for row in event_rows:
                row.update(seed=seed, architecture=name, input_view="U_realized", subset="all", evaluation="one_step", horizon_ms=1)
            self.rows["events"].extend(event_rows)
        return thresholds

    def _rollout(self, model: Any, window: np.ndarray, device: Any, ablation: str) -> Tuple[np.ndarray, np.ndarray]:
        values = self._rollout_full(model, window, device, ablation)
        return values["voltage"], values["event_probability"]

    def _rollout_full(self, model: Any, window: np.ndarray, device: Any, ablation: str) -> Dict[str, np.ndarray]:
        import torch

        recurrent = None
        voltage = []
        event = []
        calcium = []
        synapse = []
        model.eval()
        with torch.no_grad():
            for index in window:
                raw = self._batch([int(index)], include_targets=False)
                batch = self._torch_batch(raw, device)
                output = model(batch, recurrent=recurrent, ablation=ablation, decode_teacher=False)
                recurrent = {key: output[key] for key in ("voltage", "local", "global", "calcium", "synapse")}
                voltage.append(output["voltage"][0].cpu().numpy())
                event.append(torch.sigmoid(output["event_logits"])[0].cpu().numpy())
                calcium.append(output["calcium"][0].cpu().numpy())
                synapse.append(output["synapse"][0].cpu().numpy())
        return {
            "voltage": np.asarray(voltage),
            "event_probability": np.asarray(event),
            "calcium": np.asarray(calcium),
            "synapse": np.asarray(synapse),
        }

    def evaluate_rollouts(self, model: Any, name: str, seed: int, device: Any, thresholds: np.ndarray) -> None:
        ablation = name if name in {"H0", "H1", "H2"} else "H2"
        for split in ("validation", "test"):
            for horizon in self.config.rollout_horizons_ms:
                windows = self._stratified_rollout_windows(
                    split, horizon, seed=9000 + seed + horizon,
                    per_regime=self.config.evaluation_windows_per_regime,
                )
                if not windows:
                    continue
                prediction = []
                target = []
                probabilities = []
                event_targets = []
                episodes = []
                regimes = []
                progress = Progress(
                    f"evaluate {name}-seed{seed} {split} rollout-{horizon}ms",
                    len(windows),
                )
                for window_index, window in enumerate(windows):
                    trace, event = self._rollout(model, window, device, ablation)
                    prediction.append(trace[-1])
                    target.append(self.store.read_state([int(window[-1])], "t_plus_1")[0, : self.layout.segment_count])
                    probabilities.append(event)
                    event_targets.append(self.store.event_targets(window)["event_presence"])
                    episodes.extend([self._episode_id(int(window[0]))] * len(window))
                    regimes.append(self._regime(int(window[0])))
                    progress.update(window_index + 1)
                rows = voltage_fidelity_rows(
                    np.asarray(prediction), np.asarray(target),
                    model=f"{name}-seed{seed}", split=split, horizon_ms=horizon,
                    segment_regions=[str(row["region"]) for row in self.layout.segments], regimes=regimes,
                )
                for row in rows:
                    row.update(seed=seed, architecture=name, input_view="U_realized")
                    self.rows["rollout"].append(row)
                    self.rows["drift"].append(dict(row))
                    self.rows["attenuation"].append(dict(row))
                final_prediction = np.asarray(prediction)
                below = final_prediction < self.train_voltage_min[None, :]
                above = final_prediction > self.train_voltage_max[None, :]
                self.rows["ood"].append({
                    "model": f"{name}-seed{seed}", "seed": seed,
                    "architecture": name, "split": split, "horizon_ms": horizon,
                    "voltage_out_of_training_range_fraction": float(np.mean(below | above)),
                    "voltage_below_physical_floor_fraction": float(np.mean(final_prediction < -150.0)),
                    "voltage_above_physical_ceiling_fraction": float(np.mean(final_prediction > 100.0)),
                })
                events = pooled_event_metrics(
                    np.concatenate(probabilities), np.concatenate(event_targets), episodes,
                    model=f"{name}-seed{seed}", split=split, thresholds=thresholds,
                )
                for row in events:
                    row.update(seed=seed, architecture=name, input_view="U_realized", subset="all", evaluation="rollout", horizon_ms=horizon)
                self.rows["events"].extend(events)

    def evaluate_branching(self, model: Any, name: str, seed: int, device: Any) -> None:
        ablation = name if name in {"H0", "H1", "H2"} else "H2"
        trajectories: Dict[str, Dict[str, Any]] = {}
        for split, kind in (("branching_near_test", "near"), ("branching_far_test", "far"), ("release_identifiability_test", "release_identifiability")):
            for indices in self.store.episode_indices(split=split):
                horizon = min(32, len(indices))
                window = indices[:horizon]
                trajectory = str(self.store.metadata["trajectory_id"][window[0]])
                episode = self.store.episode_by_trajectory.get(trajectory, {})
                pair_id = str(episode.get("branch_pair_id", episode.get("release_pair_id", episode.get("snapshot_id", "unknown"))))
                prediction, event_probability = self._rollout(model, window, device, ablation)
                teacher = self.store.read_state(window, "t_plus_1")[:, : self.layout.segment_count]
                trajectories[trajectory] = {
                    "pair_id": pair_id, "kind": kind, "prediction": prediction,
                    "teacher": teacher, "event_probability": event_probability,
                    "event_target": self.store.event_targets(window)["event_presence"],
                }
        grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
        for row in trajectories.values():
            grouped.setdefault((row["kind"], row["pair_id"]), []).append(row)
        for (kind, pair_id), values in grouped.items():
            if len(values) < 2:
                continue
            left, right = values[:2]
            divergent_target = bool(
                np.any(left["event_target"] != right["event_target"])
            )
            divergent_prediction = bool(
                np.any((left["event_probability"] >= 0.5) != (right["event_probability"] >= 0.5))
            )
            rows, _ = branching_metrics(
                [{
                    "pair_id": pair_id,
                    "branching_kind": kind,
                    "horizon_ms": int(len(left["teacher"])),
                    "teacher_a": left["teacher"],
                    "teacher_b": right["teacher"],
                    "prediction_a": left["prediction"],
                    "prediction_b": right["prediction"],
                    "divergent_event_correct": divergent_target == divergent_prediction,
                }],
                teacher_distance_floor=0.05,
            )
            for row in rows:
                row.update(model=f"{name}-seed{seed}", seed=seed, architecture=name, input_view="U_realized")
            self.rows["branching"].extend(rows)

    def evaluate_recovery(self, model: Any, name: str, seed: int, device: Any) -> None:
        ablation = name if name in {"H0", "H1", "H2"} else "H2"
        for indices in self.store.episode_indices(split="recovery_test"):
            horizon = min(32, len(indices))
            window = indices[:horizon]
            prediction = self._rollout_full(model, window, device, ablation)
            teacher_raw = self.store.read_state(window, "t_plus_1")
            teacher_normalized = self.normalizer.normalize_state(teacher_raw).astype(np.float32)
            teacher_calcium, teacher_synapse = explicit_teacher_views(
                teacher_normalized,
                self.arrays,
                segment_count=self.layout.segment_count,
                calcium_dim=self.config.model.calcium_state_dim,
                synapse_dim=self.config.model.synapse_state_dim,
            )
            voltage_error = np.sqrt(np.mean(
                (prediction["voltage"] - teacher_raw[:, : self.layout.segment_count]) ** 2,
                axis=1,
            ))
            recovered = np.flatnonzero(voltage_error <= 1.0)
            target_events = self.store.event_targets(window)["event_presence"]
            predicted_events = prediction["event_probability"] >= 0.5
            self.rows["recovery"].append({
                "model": f"{name}-seed{seed}", "seed": seed,
                "architecture": name, "input_view": "U_realized",
                "episode_id": self._episode_id(int(window[0])), "horizon_ms": horizon,
                "voltage_rmse_mv": float(np.sqrt(np.mean(
                    (prediction["voltage"] - teacher_raw[:, : self.layout.segment_count]) ** 2
                ))),
                "calcium_rmse": float(np.sqrt(np.mean(
                    (prediction["calcium"] - teacher_calcium) ** 2
                ))),
                "synapse_state_rmse": float(np.sqrt(np.mean(
                    (prediction["synapse"] - teacher_synapse) ** 2
                ))),
                "recovery_time_error_proxy_ms": int(recovered[0]) + 1 if len(recovered) else math.nan,
                "post_event_excitability_teacher_event_count": int(target_events[horizon // 2 :].sum()),
                "post_event_excitability_predicted_event_count": int(predicted_events[horizon // 2 :].sum()),
                "post_event_excitability_count_error": int(abs(
                    predicted_events[horizon // 2 :].sum() - target_events[horizon // 2 :].sum()
                )),
                "similar_voltage_different_history_test": "recovery episode family",
            })

    def _write_tables(self) -> None:
        names = {
            "one_step": "one_step_metrics.parquet",
            "rollout": "rollout_metrics.parquet",
            "events": "event_metrics.parquet",
            "branching": "branching_metrics.parquet",
            "recovery": "recovery_metrics.parquet",
            "drift": "regional_drift.parquet",
            "attenuation": "peak_attenuation.parquet",
            "comparison": "model_comparison.parquet",
            "ood": "out_of_domain_metrics.parquet",
        }
        for key, filename in names.items():
            write_parquet(self.output_dir / filename, self.rows[key])

    def finalize(self, canary: Mapping[str, Any]) -> Dict[str, Any]:
        h2_test = [
            row for row in self.rows["one_step"]
            if row.get("architecture") == "H2" and row.get("split") == "test" and row.get("region") == "all"
        ]
        h2_events = [
            row for row in self.rows["events"]
            if row.get("architecture") == "H2" and row.get("split") == "test"
            and row.get("evaluation") == "one_step" and row.get("subset") == "all"
        ]
        h2_branch = [
            row for row in self.rows["branching"]
            if row.get("architecture") == "H2" and row.get("eligible")
        ]
        event_f1 = float(np.mean([row["f1"] for row in h2_events])) if h2_events else math.nan
        branching_retention = float(np.median([row["divergence_retention"] for row in h2_branch])) if h2_branch else math.nan
        h2_rmse = float(np.mean([row["rmse_mv"] for row in h2_test])) if h2_test else math.nan
        conv_events = [row for row in self.rows["events"] if row.get("architecture") == "ConvGRU" and row.get("split") == "test" and row.get("evaluation") == "one_step" and row.get("subset") == "all"]
        conv_f1 = float(np.mean([row["f1"] for row in conv_events])) if conv_events else math.nan
        for architecture in ("H0", "H1", "H2", "ConvGRU"):
            voltage_rows = [
                row for row in self.rows["one_step"]
                if row.get("architecture") == architecture
                and row.get("split") == "test" and row.get("region") == "all"
            ]
            event_rows = [
                row for row in self.rows["events"]
                if row.get("architecture") == architecture
                and row.get("split") == "test"
                and row.get("evaluation") == "one_step"
                and row.get("subset") == "all"
            ]
            rollout_rows = [
                row for row in self.rows["rollout"]
                if row.get("architecture") == architecture
                and row.get("split") == "test" and row.get("region") == "all"
            ]
            if voltage_rows or event_rows or rollout_rows:
                self.rows["comparison"].append({
                    "architecture": architecture,
                    "one_step_rmse_mv": float(np.mean([row["rmse_mv"] for row in voltage_rows])) if voltage_rows else math.nan,
                    "macro_event_f1": float(np.mean([row["f1"] for row in event_rows])) if event_rows else math.nan,
                    "mean_rollout_rmse_mv": float(np.mean([row["rmse_mv"] for row in rollout_rows])) if rollout_rows else math.nan,
                    "source": "notebook_05",
                })
        if self.b3_report:
            self.rows["comparison"].append({
                "architecture": "B3_reference",
                "one_step_rmse_mv": math.nan,
                "macro_event_f1": float(self.b3_report.get("event_fidelity", {}).get("macro_f1_overall", math.nan)),
                "mean_rollout_rmse_mv": float(self.b3_report.get("flowmap_learnability", {}).get("b3_realized_rollout_rmse_mv", math.nan)),
                "source": "notebook_04_final_report",
            })
        self._write_tables()
        if not canary.get("proceed_to_full_training"):
            scenario = canary.get("scenario")
        elif math.isfinite(event_f1) and event_f1 > 0.3 and math.isfinite(branching_retention) and branching_retention > 0.3:
            scenario = "A_HAYFLOW_HINES_SHOWS_LIFE"
        elif math.isfinite(conv_f1) and conv_f1 > event_f1:
            scenario = "B_CONVENTIONAL_RECURRENCE_OUTPERFORMS_HAYFLOW"
        else:
            scenario = "D_CANARY_OVERFITS_BUT_GENERALISATION_IS_INSUFFICIENT"
        report = {
            "schema_version": "05-final-report-v1",
            "valid": not self.training_contract_blockers,
            "decision_grade": self.config.profile == "diagnostic_full",
            "scenario": scenario,
            "dataset": self.store.report(),
            "canary": {
                "scenario": canary.get("scenario"),
                "proceed_to_full_training": canary.get("proceed_to_full_training"),
            },
            "hayflow_hines": {
                "one_step_rmse_mv": h2_rmse,
                "macro_event_f1": event_f1,
                "median_branching_retention": branching_retention,
            },
            "convgru": {"macro_event_f1": conv_f1},
            "b3_reference": self.b3_report.get("flowmap_learnability", {}),
            "registry": self.registry,
            "training_contract": {
                "branch_pair": (
                    list(self.train_branch_pair)
                    if self.train_branch_pair is not None else None
                ),
                "blockers": self.training_contract_blockers,
                "test_pairs_used_for_training": False,
            },
            "methodology": {
                "primary_input": "U_realized",
                "physical_hines_solve": True,
                "teacher_encoder_used_during_rollout": False,
                "rollout_sampling": "episode-and-regime-stratified",
                "B3_forward_reused": False,
            },
        }
        _write_json(self.output_dir / "final_report.json", report)
        _write_json(self.output_dir / "checkpoint_registry.json", self.registry)
        return report

    def run_full(self, canary: Mapping[str, Any]) -> Dict[str, Any]:
        if not canary.get("proceed_to_full_training"):
            return self.finalize(canary)
        if self.training_contract_blockers:
            blocked_canary = dict(canary)
            blocked_canary["proceed_to_full_training"] = False
            blocked_canary["scenario"] = "C_TRAINING_CONTRACT_BLOCKED"
            return self.finalize(blocked_canary)
        require_torch()
        import torch

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        for seed_index, seed in enumerate(self.config.seeds):
            models = self._models(device)
            # H0/H1 are mechanistic ablations; H2 and ConvGRU answer the main
            # architecture question.  Each has its own curriculum/checkpoints.
            names = ("H0", "H1", "H2", "ConvGRU") if seed_index == 0 else ("H2", "ConvGRU")
            for name in names:
                model = self.train_full_model(name, models[name], int(seed), device)
                thresholds = self.evaluate_one_step(model, name, int(seed), device)
                self.evaluate_rollouts(model, name, int(seed), device, thresholds)
                self.evaluate_branching(model, name, int(seed), device)
                self.evaluate_recovery(model, name, int(seed), device)
                del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        return self.finalize(canary)
