"""End-to-end experiment runner used by notebook 02.

The runner is intentionally diagnostic: it trains small baselines, records
every split separately, resumes from checkpoints, and applies conservative
GO/NO-GO rules.  It does not implement the final HayFlow architecture.
"""

from __future__ import annotations

import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from ..hayflow_data import (
    DYNAMIC_CATEGORIES,
    EVENT_KINDS,
    FlowmapBundle,
    FlowmapLayout,
    FlowmapTransitionStore,
    StateNormalizer,
    batch_iterator,
)
from ..hayflow_eval import (
    binary_event_metric_rows,
    decide_go_no_go,
    rollout_metric_row,
    state_metric_rows,
    write_parquet,
)
from .full_state_flowmap import (
    DualRidgeBaseline,
    FlatResidualMLP,
    FlowmapModelConfig,
    PersistenceBaseline,
    StructuredSharedResidual,
    parameter_count,
    require_torch,
    ridge_design_matrix,
    structured_arrays,
)


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


@dataclass(frozen=True)
class FlowmapExperimentConfig:
    profile: str = "diagnostic_full"
    initialization_seeds: Tuple[int, ...] = (17, 29, 43)
    maximum_epochs: int = 60
    early_stopping_patience: int = 8
    batch_size_b2: int = 8
    batch_size_b3: int = 2
    evaluation_batch_size_b2: int = 8
    evaluation_batch_size_b3: int = 2
    learning_rate: float = 3e-4
    weight_decay: float = 1e-5
    gradient_clip_norm: float = 1.0
    ridge_alpha: float = 10.0
    state_loss_weight: float = 1.0
    event_presence_loss_weight: float = 0.25
    event_timing_loss_weight: float = 0.05
    event_region_loss_weight: float = 0.02
    privileged_current_loss_weight: float = 0.05
    privileged_microtrace_loss_weight: float = 0.02
    rollout_horizons_ms: Tuple[int, ...] = (2, 4, 8, 16)
    maximum_rollout_windows_per_split: int = 32
    num_workers: int = 0
    deterministic_algorithms: bool = True

    def validate(self) -> None:
        if self.profile not in {"smoke", "diagnostic_full"}:
            raise ValueError("profile must be smoke or diagnostic_full")
        if not self.initialization_seeds:
            raise ValueError("at least one initialization seed is required")
        if min(
            self.maximum_epochs,
            self.early_stopping_patience,
            self.batch_size_b2,
            self.batch_size_b3,
            self.evaluation_batch_size_b2,
            self.evaluation_batch_size_b3,
        ) <= 0:
            raise ValueError("training counts must be positive")
        if any(horizon <= 0 for horizon in self.rollout_horizons_ms):
            raise ValueError("rollout horizons must be positive")

    def effective(self) -> "FlowmapExperimentConfig":
        self.validate()
        if self.profile == "diagnostic_full":
            return self
        return FlowmapExperimentConfig(
            **{
                **asdict(self),
                "initialization_seeds": (self.initialization_seeds[0],),
                "maximum_epochs": min(2, self.maximum_epochs),
                "early_stopping_patience": 1,
                "rollout_horizons_ms": (2,),
                "maximum_rollout_windows_per_split": 2,
            }
        )


class AuxiliaryNormalizer:
    """Robust train-only normalization for P1 auxiliary targets."""

    def __init__(self) -> None:
        self.center: Optional[np.ndarray] = None
        self.scale: Optional[np.ndarray] = None
        self.layout: Dict[str, slice] = {}

    def fit(
        self, values: np.ndarray, layout: Mapping[str, slice]
    ) -> "AuxiliaryNormalizer":
        values = np.asarray(values, dtype=np.float64)
        self.center = np.median(values, axis=0)
        q25, q75 = np.percentile(values, [25.0, 75.0], axis=0)
        self.scale = (q75 - q25) / 1.349
        standard = values.std(axis=0)
        self.scale = np.where(self.scale > 1e-10, self.scale, standard)
        self.scale = np.where(self.scale > 1e-10, self.scale, 1.0)
        self.layout = dict(layout)
        return self

    def transform(self, values: np.ndarray) -> np.ndarray:
        return ((values - self.center) / self.scale).astype(np.float32)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "fit_split": "train",
            "center": self.center.tolist(),
            "scale": self.scale.tolist(),
            "layout": {
                key: [value.start, value.stop] for key, value in self.layout.items()
            },
        }


class Progress:
    def __init__(self, label: str, total: int) -> None:
        self.label = label
        self.total = max(1, int(total))
        self.started = time.monotonic()

    def update(self, current: int, detail: str = "") -> None:
        current = max(0, min(int(current), self.total))
        elapsed = time.monotonic() - self.started
        rate = current / elapsed if elapsed > 0.0 else 0.0
        eta = (self.total - current) / rate if rate > 0.0 else math.inf
        eta_text = "?" if not math.isfinite(eta) else f"{eta / 60.0:.1f} min"
        print(
            f"[HayFlow 02][{self.label}] {current}/{self.total} "
            f"({100.0 * current / self.total:.1f}%) ETA {eta_text} {detail}",
            flush=True,
        )


def _seed_everything(seed: int, deterministic: bool) -> None:
    require_torch()
    import torch

    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)


def _torch_batch(batch: Mapping[str, Any], device: Any) -> Dict[str, Any]:
    import torch

    result: Dict[str, Any] = {}
    integer = {"indices", "u2_segment_ids", "event_region"}
    boolean = {"u2_mask", "event_timing_mask", "event_region_mask"}
    excluded = {
        "raw_state_t",
        "raw_state_t_plus_1",
        "auxiliary_target_raw",
        "auxiliary_layout",
    }
    for key, value in batch.items():
        if key in excluded:
            continue
        if not isinstance(value, np.ndarray):
            result[key] = value
        elif key in integer:
            result[key] = torch.as_tensor(value, dtype=torch.long, device=device)
        elif key in boolean:
            result[key] = torch.as_tensor(value, dtype=torch.bool, device=device)
        else:
            result[key] = torch.as_tensor(value, dtype=torch.float32, device=device)
    return result


def _regimes(store: FlowmapTransitionStore, indices: Sequence[int]) -> List[str]:
    result = []
    for index in indices:
        category = str(store.metadata["category"][int(index)])
        protocol = str(store.metadata["protocol"][int(index)])
        if bool(store.metadata["negative_control"][int(index)]):
            result.append("event_boundary_negative")
        elif "plateau" in protocol:
            result.append("nmda_plateau")
        elif "calcium" in protocol or "bap" in protocol:
            result.append("calcium_bac")
        elif category == "somatic_events":
            result.append("somatic_spike")
        elif category == "rest_subthreshold":
            result.append("rest_subthreshold")
        elif category == "branching":
            result.append("branching")
        else:
            result.append(category)
    return result


class FullStateFlowmapExperiment:
    """Orchestrate B0--B3, ablations, rollouts, figures, and final report."""

    def __init__(
        self,
        bundle: FlowmapBundle,
        output_dir: Path,
        config: FlowmapExperimentConfig,
    ) -> None:
        self.bundle = bundle
        self.output_dir = Path(output_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.figure_dir = self.output_dir / "figures"
        self.prediction_dir = self.output_dir / "prediction_examples"
        for path in (self.checkpoint_dir, self.figure_dir, self.prediction_dir):
            path.mkdir(parents=True, exist_ok=True)
        self.config = config.effective()
        self.layout = FlowmapLayout(bundle)
        self.store = FlowmapTransitionStore(bundle, self.layout)
        self.normalizer: Optional[StateNormalizer] = None
        self.aux_normalizer: Optional[AuxiliaryNormalizer] = None
        self.auxiliary_layout: Dict[str, slice] = {}
        self.one_step_rows: List[Dict[str, Any]] = []
        self.event_rows: List[Dict[str, Any]] = []
        self.rollout_rows: List[Dict[str, Any]] = []
        self.ablation_rows: List[Dict[str, Any]] = []
        self.training_history: Dict[str, List[Dict[str, Any]]] = {}
        self.model_registry: Dict[str, Dict[str, Any]] = {}
        self.branching_report: Dict[str, Any] = {
            "available": False,
            "reason": "reference B3 model has not been evaluated",
        }
        self._write_initial_manifest()

    def _write_initial_manifest(self) -> None:
        split_counts = {
            key: int(len(value)) for key, value in self.store.split_indices.items()
        }
        payload = {
            "schema_version": "0.1.0",
            "experiment": "full_state_flowmap_baseline",
            "diagnostic_purpose_only": True,
            "code_commit": _git_commit(Path(__file__).resolve().parents[2]),
            "runtime_versions": _runtime_versions(),
            "dataset_schema_version": self.bundle.schema_version,
            "dataset_manifest_sha256": _file_sha256(
                self.bundle.root / "dataset_manifest.json"
            ),
            "teacher_commit": self.bundle.manifest["teacher_commit"],
            "transition_count": self.store.count,
            "split_counts": split_counts,
            "state_width": self.layout.state_width,
            "privileged_width": self.layout.privileged_width,
            "rng_and_release_contract": self.store.release_contract,
            "synapse_state_update_contract": {
                "exact_updater_used": False,
                "policy": "predict the stored teacher synapse-state delta",
                "reason": (
                    "schema 1.0.1 does not provide a separately validated exact "
                    "discrete updater for the teacher-specific synapse states"
                ),
            },
            "anti_leakage": {
                "normalization_fit_split": "train",
                "teacher_forcing_input": "S_t only",
                "future_microtraces_as_input": False,
                "future_event_labels_as_input": False,
                "rng_regression_target": False,
                "whole_trajectory_splits": True,
            },
            "limitations": [
                "Only 1224 diagnostic transitions are available.",
                "Release outcomes are unavailable; scheduled events are used.",
                "This notebook tests signal and short-rollout feasibility only.",
            ],
            "config": asdict(self.config),
        }
        write_json(self.output_dir / "experiment_manifest.json", payload)

    def prepare(self) -> Dict[str, Any]:
        self.normalizer = self.store.fit_state_normalizer()
        train = self.store.split_indices["train"]
        auxiliary, layout = self.store.auxiliary_targets(train)
        self.auxiliary_layout = layout
        self.aux_normalizer = AuxiliaryNormalizer().fit(auxiliary, layout)
        train_state_t = self.store.read_state(train, "t")
        train_state_t1 = self.store.read_state(train, "t_plus_1")
        normalized_delta = self.normalizer.normalize_delta(
            train_state_t, train_state_t1
        )
        delta_diagnostics = {}
        for category in DYNAMIC_CATEGORIES:
            state_slice = self.layout.category_slices[category]
            absolute = np.abs(normalized_delta[:, state_slice]).reshape(-1)
            delta_diagnostics[category] = {
                "absolute_p50": float(np.percentile(absolute, 50.0)),
                "absolute_p95": float(np.percentile(absolute, 95.0)),
                "absolute_p99": float(np.percentile(absolute, 99.0)),
                "absolute_maximum": float(np.max(absolute)),
            }
        write_json(
            self.output_dir / "normalization_schema.json",
            {
                "state": self.normalizer.to_dict(),
                "auxiliary": self.aux_normalizer.to_dict(),
                "train_normalized_delta_diagnostics": delta_diagnostics,
            },
        )
        model_configs = [row.to_dict() for row in self._neural_configs()]
        write_json(
            self.output_dir / "model_configs.json",
            {
                "models": model_configs,
                "training": asdict(self.config),
                "parameter_to_transition_ratio_must_be_reported": True,
            },
        )
        return {
            "train_transitions": len(train),
            "state_width": self.layout.state_width,
            "auxiliary_width": int(auxiliary.shape[1]),
            "release_contract": self.store.release_contract,
            "train_normalized_delta_diagnostics": delta_diagnostics,
        }

    def preflight(self) -> Dict[str, Any]:
        """Run shape/finite/causality checks before starting long training."""

        import torch

        require_torch()
        if self.normalizer is None:
            self.prepare()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        sample_indices = self.store.split_indices["train"][:2]
        checks = []
        for config in self._neural_configs():
            model = self._build_model(config, device)
            batch = self._load_training_batch(sample_indices, config, device)
            model.eval()
            with torch.no_grad():
                output = model(batch)
            valid = (
                tuple(output["delta"].shape)
                == (len(sample_indices), self.layout.state_width)
                and tuple(output["event_logits"].shape)
                == (len(sample_indices), len(EVENT_KINDS))
                and bool(torch.isfinite(output["delta"]).all())
            )
            if config.privileged_loss:
                valid = valid and (
                    output["privileged_current"].shape[-1]
                    == self.layout.privileged_width
                    and output["aux_dense"].shape[-1]
                    == config.auxiliary_dense_dim
                )
            checks.append(
                {
                    "model": config.to_dict(),
                    "parameter_count": parameter_count(model),
                    "valid": bool(valid),
                }
            )
            del model, output, batch
            _clear_cuda_cache()
        report = {
            "valid": all(row["valid"] for row in checks),
            "device": str(device),
            "models": checks,
            "dataset_schema": self.bundle.schema_version,
            "future_targets_excluded_from_model_input": True,
            "rng_excluded_from_regression": True,
            "release_contract": self.store.release_contract,
        }
        write_json(self.output_dir / "preflight_report.json", report)
        if not report["valid"]:
            raise RuntimeError(f"flow-map preflight failed: {report}")
        return report

    def _neural_configs(self) -> List[FlowmapModelConfig]:
        dense_width = 0
        if self.auxiliary_layout:
            current = self.auxiliary_layout["currents_conductances_t_plus_1"]
            dense_width = max(value.stop for value in self.auxiliary_layout.values()) - current.stop
        configs = [
            FlowmapModelConfig("B2_flat_mlp", "full", "U2"),
            FlowmapModelConfig("B3_structured", "voltage_only", "none"),
            FlowmapModelConfig("B3_structured", "voltage_only", "U1"),
            FlowmapModelConfig("B3_structured", "voltage_only", "U2"),
            FlowmapModelConfig("B3_structured", "full", "U1"),
            FlowmapModelConfig("B3_structured", "full", "U2"),
            FlowmapModelConfig(
                "B3_structured",
                "full",
                "U2",
                privileged_loss=True,
                auxiliary_dense_dim=dense_width,
            ),
        ]
        if self.config.profile == "smoke":
            return [configs[0], configs[-2], configs[-1]]
        return configs

    @staticmethod
    def model_id(config: FlowmapModelConfig, seed: int) -> str:
        privilege = "P1" if config.privileged_loss else "P0"
        return (
            f"{config.model_kind}-{config.state_mode}-{config.input_encoding}-"
            f"{privilege}-seed{int(seed)}"
        )

    def run(self) -> Dict[str, Any]:
        if self.normalizer is None:
            self.prepare()
        self.run_b0()
        self.run_b1()
        require_torch()
        configs = self._neural_configs()
        total = sum(
            1 if row.model_kind == "B2_flat_mlp" else len(self.config.initialization_seeds)
            for row in configs
        )
        progress = Progress("modelli neurali", total)
        completed = 0
        for model_config in configs:
            seeds = (
                self.config.initialization_seeds[:1]
                if model_config.model_kind == "B2_flat_mlp"
                else self.config.initialization_seeds
            )
            for seed in seeds:
                model_id = self.model_id(model_config, seed)
                model, history = self.train_neural(model_config, seed)
                self.training_history[model_id] = history
                self.evaluate_neural(model_id, model)
                self.evaluate_rollouts_neural(model_id, model)
                completed += 1
                progress.update(completed, model_id)
                del model
                _clear_cuda_cache()
        return self.finalize()

    def run_b0(self) -> None:
        for split, indices in self.store.split_indices.items():
            true = self.store.read_state(indices, "t_plus_1")
            pred = PersistenceBaseline.predict_raw(
                self.store.read_state(indices, "t")
            )
            self.one_step_rows.extend(
                state_metric_rows(
                    pred,
                    true,
                    layout=self.layout,
                    normalizer=self.normalizer,
                    model_name=PersistenceBaseline.name,
                    split=split,
                    regimes=_regimes(self.store, indices),
                )
            )
            event = self.store.event_targets(indices)
            self.event_rows.extend(
                binary_event_metric_rows(
                    np.zeros_like(event["event_presence"]),
                    event["event_presence"],
                    timing_prediction=None,
                    timing_target=None,
                    timing_mask=None,
                    model_name=PersistenceBaseline.name,
                    split=split,
                )
            )
        self.evaluate_rollouts_classical(PersistenceBaseline.name, None)

    def run_b1(self) -> None:
        train = self.store.split_indices["train"]
        train_batch = self.store.load_batch(train, self.normalizer)
        event_target = train_batch["event_presence"]
        combined_target = np.concatenate(
            [train_batch["delta_target"], event_target], axis=1
        )
        for encoding in ("U1", "U2"):
            model_name = f"B1_affine_delta-full-{encoding}"
            features = ridge_design_matrix(
                train_batch,
                voltage_width=642,
                state_mode="full",
                input_encoding=encoding,
            )
            model = DualRidgeBaseline(self.config.ridge_alpha).fit(
                features, combined_target
            )
            path = self.checkpoint_dir / f"{model_name}.npz"
            model.save(path)
            self.model_registry[model_name] = {
                "kind": "B1",
                "checkpoint": str(path.relative_to(self.output_dir)),
                "parameter_count_proxy": int(features.shape[0] ** 2),
                "parameter_to_train_transition_ratio": float(features.shape[0]),
            }
            for split, indices in self.store.split_indices.items():
                batch = self.store.load_batch(indices, self.normalizer)
                prediction = model.predict(
                    ridge_design_matrix(
                        batch,
                        voltage_width=642,
                        state_mode="full",
                        input_encoding=encoding,
                    )
                )
                delta = prediction[:, : self.layout.state_width]
                raw = self.normalizer.reconstruct(batch["raw_state_t"], delta)
                self.one_step_rows.extend(
                    state_metric_rows(
                        raw,
                        batch["raw_state_t_plus_1"],
                        layout=self.layout,
                        normalizer=self.normalizer,
                        model_name=model_name,
                        split=split,
                        regimes=_regimes(self.store, indices),
                    )
                )
                probability = np.clip(
                    prediction[:, self.layout.state_width :], 0.0, 1.0
                )
                self.event_rows.extend(
                    binary_event_metric_rows(
                        probability,
                        batch["event_presence"],
                        timing_prediction=None,
                        timing_target=None,
                        timing_mask=None,
                        model_name=model_name,
                        split=split,
                    )
                )
            self.evaluate_rollouts_classical(model_name, (model, encoding))

    def _build_model(self, config: FlowmapModelConfig, device: Any) -> Any:
        import torch

        metadata = self.layout.to_model_metadata()
        if config.model_kind == "B2_flat_mlp":
            model = FlatResidualMLP(
                config,
                state_width=self.layout.state_width,
                voltage_width=642,
                u1_width=len(metadata["u1_feature_names"]),
                u2_width=len(metadata["u2_event_feature_names"]),
                segment_count=self.layout.segment_count,
                event_count=len(EVENT_KINDS),
                region_count=len(self.layout.region_names),
            )
        else:
            model = StructuredSharedResidual(
                config, metadata, structured_arrays(self.layout)
            )
        return model.to(device)

    def _event_pos_weight(self) -> np.ndarray:
        targets = self.store.event_targets(self.store.split_indices["train"])[
            "event_presence"
        ]
        positives = targets.sum(axis=0)
        negatives = len(targets) - positives
        return np.clip(negatives / np.maximum(positives, 1.0), 1.0, 50.0)

    def _loss(
        self,
        output: Mapping[str, Any],
        batch: Mapping[str, Any],
        pos_weight: Any,
        config: FlowmapModelConfig,
    ) -> Tuple[Any, Dict[str, float]]:
        import torch

        block_losses = []
        details: Dict[str, float] = {}
        for category in DYNAMIC_CATEGORIES:
            state_slice = self.layout.category_slices[category]
            value = torch.nn.functional.smooth_l1_loss(
                output["delta"][:, state_slice],
                batch["delta_target"][:, state_slice],
            )
            block_losses.append(value)
            details[f"state_{category}"] = float(value.detach())
        state_loss = torch.stack(block_losses).mean()
        presence = torch.nn.functional.binary_cross_entropy_with_logits(
            output["event_logits"], batch["event_presence"], pos_weight=pos_weight
        )
        timing_mask = batch["event_timing_mask"]
        timing = (
            torch.nn.functional.smooth_l1_loss(
                output["event_timing"][timing_mask],
                batch["event_timing"][timing_mask],
            )
            if timing_mask.any()
            else state_loss.new_zeros(())
        )
        region_mask = batch["event_region_mask"]
        region = (
            torch.nn.functional.cross_entropy(
                output["event_region_logits"][region_mask],
                batch["event_region"][region_mask],
            )
            if region_mask.any()
            else state_loss.new_zeros(())
        )
        total = (
            self.config.state_loss_weight * state_loss
            + self.config.event_presence_loss_weight * presence
            + self.config.event_timing_loss_weight * timing
            + self.config.event_region_loss_weight * region
        )
        if config.privileged_loss:
            current_slice = self.auxiliary_layout[
                "currents_conductances_t_plus_1"
            ]
            current_target = batch["auxiliary_target"][:, current_slice]
            dense_target = batch["auxiliary_target"][:, current_slice.stop :]
            current_loss = torch.nn.functional.smooth_l1_loss(
                output["privileged_current"], current_target
            )
            dense_loss = torch.nn.functional.smooth_l1_loss(
                output["aux_dense"], dense_target
            )
            total = (
                total
                + self.config.privileged_current_loss_weight * current_loss
                + self.config.privileged_microtrace_loss_weight * dense_loss
            )
            details["privileged_current"] = float(current_loss.detach())
            details["privileged_microtrace"] = float(dense_loss.detach())
        details.update(
            {
                "state": float(state_loss.detach()),
                "event_presence": float(presence.detach()),
                "event_timing": float(timing.detach()),
                "event_region": float(region.detach()),
                "total": float(total.detach()),
            }
        )
        return total, details

    def _load_training_batch(
        self, indices: Sequence[int], config: FlowmapModelConfig, device: Any
    ) -> Dict[str, Any]:
        batch = self.store.load_batch(
            indices, self.normalizer, include_auxiliary=config.privileged_loss
        )
        if config.privileged_loss:
            batch["auxiliary_target"] = self.aux_normalizer.transform(
                batch["auxiliary_target_raw"]
            )
        return _torch_batch(batch, device)

    def _epoch(
        self,
        model: Any,
        config: FlowmapModelConfig,
        indices: Sequence[int],
        device: Any,
        pos_weight: Any,
        *,
        optimizer: Optional[Any],
        seed: int,
    ) -> Dict[str, float]:
        import torch

        training = optimizer is not None
        model.train(training)
        totals: Dict[str, float] = {}
        count = 0
        size = (
            self.config.batch_size_b2
            if config.model_kind == "B2_flat_mlp"
            else self.config.batch_size_b3
        )
        iterator = batch_iterator(
            indices,
            batch_size=size,
            seed=seed,
            shuffle=training,
        )
        for batch_indices in iterator:
            batch = self._load_training_batch(batch_indices, config, device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            with torch.set_grad_enabled(training):
                output = model(batch)
                loss, details = self._loss(
                    output, batch, pos_weight, config
                )
                if training:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), self.config.gradient_clip_norm
                    )
                    optimizer.step()
            for key, value in details.items():
                totals[key] = totals.get(key, 0.0) + value * len(batch_indices)
            count += len(batch_indices)
        return {key: value / max(1, count) for key, value in totals.items()}

    def train_neural(
        self, config: FlowmapModelConfig, seed: int
    ) -> Tuple[Any, List[Dict[str, Any]]]:
        import torch

        _seed_everything(seed, self.config.deterministic_algorithms)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = self._build_model(config, device)
        model_id = self.model_id(config, seed)
        run_dir = self.checkpoint_dir / model_id
        run_dir.mkdir(parents=True, exist_ok=True)
        last_path = run_dir / "last.pt"
        best_path = run_dir / "best.pt"
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        start_epoch = 0
        best_validation = math.inf
        patience = 0
        history: List[Dict[str, Any]] = []
        if last_path.is_file():
            saved = torch.load(last_path, map_location=device)
            if saved["model_config"] != config.to_dict():
                raise RuntimeError(f"checkpoint config mismatch for {model_id}")
            model.load_state_dict(saved["model_state"])
            optimizer.load_state_dict(saved["optimizer_state"])
            start_epoch = int(saved["epoch"]) + 1
            best_validation = float(saved["best_validation"])
            patience = int(saved["patience"])
            history = list(saved.get("history", []))
        pos_weight = torch.as_tensor(
            self._event_pos_weight(), dtype=torch.float32, device=device
        )
        progress = Progress(model_id, self.config.maximum_epochs)
        epoch_range: Iterable[int] = range(start_epoch, self.config.maximum_epochs)
        if patience >= self.config.early_stopping_patience:
            print(
                f"[HayFlow 02][{model_id}] early stopping already reached; "
                "resuming from best.pt for evaluation",
                flush=True,
            )
            epoch_range = ()
        for epoch in epoch_range:
            train_metrics = self._epoch(
                model,
                config,
                self.store.split_indices["train"],
                device,
                pos_weight,
                optimizer=optimizer,
                seed=seed + epoch,
            )
            validation_metrics = self._epoch(
                model,
                config,
                self.store.split_indices["validation"],
                device,
                pos_weight,
                optimizer=None,
                seed=seed,
            )
            row = {
                "epoch": epoch,
                "train": train_metrics,
                "validation": validation_metrics,
            }
            history.append(row)
            current = float(validation_metrics["total"])
            improved = current < best_validation - 1e-8
            if improved:
                best_validation = current
                patience = 0
                torch.save(
                    {
                        "model_state": model.state_dict(),
                        "model_config": config.to_dict(),
                        "seed": seed,
                        "epoch": epoch,
                        "best_validation": best_validation,
                    },
                    best_path,
                )
            else:
                patience += 1
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "model_config": config.to_dict(),
                    "seed": seed,
                    "epoch": epoch,
                    "best_validation": best_validation,
                    "patience": patience,
                    "history": history,
                },
                last_path,
            )
            progress.update(
                epoch + 1,
                (
                    f"train={train_metrics['total']:.4g} val={current:.4g} "
                    f"val[V={validation_metrics['state_voltage']:.3g}, "
                    f"STATE={validation_metrics['state_mechanism_states']:.3g}, "
                    f"ions={validation_metrics['state_calcium_ions']:.3g}, "
                    f"syn={validation_metrics['state_synapse_states']:.3g}]"
                ),
            )
            if patience >= self.config.early_stopping_patience:
                break
        saved = torch.load(best_path, map_location=device)
        model.load_state_dict(saved["model_state"])
        count = parameter_count(model)
        train_count = len(self.store.split_indices["train"])
        self.model_registry[model_id] = {
            "kind": config.model_kind,
            "config": config.to_dict(),
            "seed": int(seed),
            "parameter_count": count,
            "train_transition_count": train_count,
            "parameter_to_train_transition_ratio": float(count / train_count),
            "best_validation_loss": float(saved["best_validation"]),
            "best_checkpoint": str(best_path.relative_to(self.output_dir)),
            "last_checkpoint": str(last_path.relative_to(self.output_dir)),
            "epochs_completed": len(history),
        }
        return model, history

    def _predict_neural_batch(
        self,
        model: Any,
        indices: Sequence[int],
        *,
        raw_state_override: Optional[np.ndarray] = None,
    ) -> Dict[str, np.ndarray]:
        import torch

        device = next(model.parameters()).device
        batch = self.store.load_batch(indices, self.normalizer)
        if raw_state_override is not None:
            batch["raw_state_t"] = np.asarray(raw_state_override)
            batch["state_t"] = self.normalizer.normalize_state(
                raw_state_override
            ).astype(np.float32)
        torch_values = _torch_batch(batch, device)
        model.eval()
        with torch.no_grad():
            output = model(torch_values)
        delta = output["delta"].detach().cpu().numpy()
        return {
            "raw_state": self.normalizer.reconstruct(batch["raw_state_t"], delta),
            "delta": delta,
            "event_probability": torch.sigmoid(output["event_logits"])
            .cpu()
            .numpy(),
            "event_timing": output["event_timing"].cpu().numpy(),
            "event_region": output["event_region_logits"]
            .argmax(dim=-1)
            .cpu()
            .numpy(),
        }

    def _predict_neural(
        self,
        model: Any,
        indices: Sequence[int],
        *,
        raw_state_override: Optional[np.ndarray] = None,
    ) -> Dict[str, np.ndarray]:
        """Run inference in bounded GPU batches and preserve transition order."""

        indices = np.asarray(indices, dtype=np.int64)
        if not len(indices):
            raise ValueError("neural prediction requires at least one transition")
        if raw_state_override is not None:
            raw_state_override = np.asarray(raw_state_override)
            if len(raw_state_override) != len(indices):
                raise ValueError("state override and transition indices must align")
        config = model.config
        batch_size = (
            self.config.evaluation_batch_size_b2
            if config.model_kind == "B2_flat_mlp"
            else self.config.evaluation_batch_size_b3
        )
        pieces: Dict[str, List[np.ndarray]] = {}
        for start in range(0, len(indices), batch_size):
            stop = min(start + batch_size, len(indices))
            override = (
                raw_state_override[start:stop]
                if raw_state_override is not None
                else None
            )
            result = self._predict_neural_batch(
                model,
                indices[start:stop],
                raw_state_override=override,
            )
            for key, value in result.items():
                pieces.setdefault(key, []).append(value)
        return {key: np.concatenate(values, axis=0) for key, values in pieces.items()}

    def evaluate_neural(self, model_id: str, model: Any) -> None:
        for split, indices in self.store.split_indices.items():
            prediction = self._predict_neural(model, indices)
            true = self.store.read_state(indices, "t_plus_1")
            self.one_step_rows.extend(
                state_metric_rows(
                    prediction["raw_state"],
                    true,
                    layout=self.layout,
                    normalizer=self.normalizer,
                    model_name=model_id,
                    split=split,
                    regimes=_regimes(self.store, indices),
                )
            )
            target = self.store.event_targets(indices)
            self.event_rows.extend(
                binary_event_metric_rows(
                    prediction["event_probability"],
                    target["event_presence"],
                    timing_prediction=prediction["event_timing"],
                    timing_target=target["event_timing"],
                    timing_mask=target["event_timing_mask"],
                    region_prediction=prediction["event_region"],
                    region_target=target["event_region"],
                    region_mask=target["event_region_mask"],
                    model_name=model_id,
                    split=split,
                )
            )
            if self._is_reference_model(model_id):
                regimes = np.asarray(_regimes(self.store, indices))
                selected = []
                for regime in sorted(set(regimes.tolist())):
                    selected.append(int(np.flatnonzero(regimes == regime)[0]))
                np.savez_compressed(
                    self.prediction_dir / f"one_step_{split}.npz",
                    transition_indices=np.asarray(indices)[selected],
                    regimes=regimes[selected],
                    predicted_voltage_mv=prediction["raw_state"][selected, :642],
                    target_voltage_mv=true[selected, :642],
                    event_probability=prediction["event_probability"][selected],
                    event_target=target["event_presence"][selected],
                )

    def _is_reference_model(self, model_id: str) -> bool:
        return (
            "B3_structured-full-U2-P1" in model_id
            and f"seed{self.config.initialization_seeds[0]}" in model_id
        )

    def _rollout_splits(self) -> Tuple[str, ...]:
        return tuple(
            split
            for split in (
                "validation",
                "deterministic_test",
                "event_boundary_test",
                "branching_test",
            )
            if split in self.store.split_indices
        )

    def evaluate_rollouts_neural(self, model_id: str, model: Any) -> None:
        for split in self._rollout_splits():
            for horizon in self.config.rollout_horizons_ms:
                windows = self.store.rollout_windows(split, horizon)[
                    : self.config.maximum_rollout_windows_per_split
                ]
                if not windows:
                    continue
                predicted = []
                target = []
                event_probabilities = []
                event_targets = []
                window_regimes = []
                saved_regimes = set()
                for window in windows:
                    state = self.store.read_state([int(window[0])], "t")
                    predicted_steps = []
                    target_steps = []
                    for index in window:
                        result = self._predict_neural(
                            model, [int(index)], raw_state_override=state
                        )
                        state = result["raw_state"]
                        predicted_steps.append(state[0, :642].copy())
                        target_steps.append(
                            self.store.read_state(
                                [int(index)], "t_plus_1"
                            )[0, :642]
                        )
                        event_probabilities.append(result["event_probability"][0])
                        event_targets.append(
                            self.store.event_targets([int(index)])["event_presence"][0]
                        )
                    predicted.append(state[0])
                    target.append(
                        self.store.read_state([int(window[-1])], "t_plus_1")[0]
                    )
                    window_regimes.append(
                        _regimes(self.store, [int(window[0])])[0]
                    )
                    if self._is_reference_model(model_id):
                        regime = _regimes(self.store, [int(window[0])])[0]
                        if regime not in saved_regimes:
                            safe = regime.replace("/", "_")
                            np.savez_compressed(
                                self.prediction_dir
                                / f"rollout_{split}_{safe}_{horizon}ms.npz",
                                transition_indices=np.asarray(window),
                                predicted_voltage_mv=np.asarray(predicted_steps),
                                target_voltage_mv=np.asarray(target_steps),
                                probe_segment_ids=np.asarray(
                                    [0, 640, 361, 387, 437, 460],
                                    dtype=np.int64,
                                ),
                                regime=regime,
                                trajectory_id=str(
                                    self.store.metadata["trajectory_id"][
                                        int(window[0])
                                    ]
                                ),
                            )
                            saved_regimes.add(regime)
                global_row = rollout_metric_row(
                        np.asarray(predicted),
                        np.asarray(target),
                        model_name=model_id,
                        split=split,
                        horizon_ms=horizon,
                        voltage_width=642,
                        layout=self.layout,
                    )
                global_row["regime"] = "all"
                self.rollout_rows.append(global_row)
                window_regimes_array = np.asarray(window_regimes)
                for regime in sorted(set(window_regimes)):
                    selected = window_regimes_array == regime
                    regime_row = rollout_metric_row(
                        np.asarray(predicted)[selected],
                        np.asarray(target)[selected],
                        model_name=model_id,
                        split=split,
                        horizon_ms=horizon,
                        voltage_width=642,
                        layout=self.layout,
                    )
                    regime_row["regime"] = regime
                    self.rollout_rows.append(regime_row)
                rollout_events = binary_event_metric_rows(
                    np.asarray(event_probabilities),
                    np.asarray(event_targets),
                    timing_prediction=None,
                    timing_target=None,
                    timing_mask=None,
                    model_name=model_id,
                    split=f"{split}_rollout_{horizon}ms",
                )
                self.event_rows.extend(rollout_events)
        if self._is_reference_model(model_id):
            self._evaluate_branching(model_id, model)

    def _evaluate_branching(self, model_id: str, model: Any) -> None:
        trajectories = []
        for trajectory_id, indices in self.store.trajectory_indices.items():
            if not len(indices) or self.store.metadata["split"][indices[0]] != "branching_test":
                continue
            horizon = min(max(self.config.rollout_horizons_ms), len(indices))
            if horizon < 1:
                continue
            window = indices[:horizon]
            state = self.store.read_state([int(window[0])], "t")
            initial = state.copy()
            for index in window:
                state = self._predict_neural(
                    model, [int(index)], raw_state_override=state
                )["raw_state"]
            trajectories.append(
                {
                    "trajectory_id": trajectory_id,
                    "horizon_ms": int(horizon),
                    "initial": initial[0],
                    "predicted": state[0],
                    "target": self.store.read_state(
                        [int(window[-1])], "t_plus_1"
                    )[0],
                }
            )
        pairs = []
        for left_index, left in enumerate(trajectories):
            for right in trajectories[left_index + 1 :]:
                initial_error = float(
                    np.max(np.abs(left["initial"] - right["initial"]))
                )
                if initial_error > 1e-5 or left["horizon_ms"] != right["horizon_ms"]:
                    continue
                true_divergence = float(
                    np.max(np.abs(left["target"][:642] - right["target"][:642]))
                )
                predicted_divergence = float(
                    np.max(
                        np.abs(
                            left["predicted"][:642] - right["predicted"][:642]
                        )
                    )
                )
                pairs.append(
                    {
                        "left_trajectory": left["trajectory_id"],
                        "right_trajectory": right["trajectory_id"],
                        "horizon_ms": left["horizon_ms"],
                        "initial_max_state_error": initial_error,
                        "teacher_max_voltage_divergence_mv": true_divergence,
                        "predicted_max_voltage_divergence_mv": predicted_divergence,
                        "divergence_ratio": float(
                            predicted_divergence / max(true_divergence, 1e-12)
                        ),
                        "collapsed": bool(
                            true_divergence > 1e-3
                            and predicted_divergence < 0.05 * true_divergence
                        ),
                    }
                )
        informative = [row for row in pairs if row["teacher_max_voltage_divergence_mv"] > 1e-3]
        self.branching_report = {
            "available": bool(pairs),
            "model": model_id,
            "trajectory_count": len(trajectories),
            "same_initial_state_pair_count": len(pairs),
            "informative_pair_count": len(informative),
            "collapsed_pair_count": sum(row["collapsed"] for row in informative),
            "distinguishes_different_futures": bool(informative)
            and not any(row["collapsed"] for row in informative),
            "pairs": pairs,
        }
        write_json(self.output_dir / "branching_report.json", self.branching_report)

    def evaluate_rollouts_classical(
        self,
        model_name: str,
        model_spec: Optional[Tuple[DualRidgeBaseline, str]],
    ) -> None:
        for split in self._rollout_splits():
            for horizon in self.config.rollout_horizons_ms:
                windows = self.store.rollout_windows(split, horizon)[
                    : self.config.maximum_rollout_windows_per_split
                ]
                if not windows:
                    continue
                predicted = []
                target = []
                window_regimes = []
                for window in windows:
                    state = self.store.read_state([int(window[0])], "t")
                    for index in window:
                        if model_spec is not None:
                            model, encoding = model_spec
                            batch = self.store.load_batch([int(index)], self.normalizer)
                            batch["raw_state_t"] = state
                            batch["state_t"] = self.normalizer.normalize_state(state).astype(
                                np.float32
                            )
                            prediction = model.predict(
                                ridge_design_matrix(
                                    batch,
                                    voltage_width=642,
                                    state_mode="full",
                                    input_encoding=encoding,
                                )
                            )[:, : self.layout.state_width]
                            state = self.normalizer.reconstruct(state, prediction)
                    predicted.append(state[0])
                    target.append(
                        self.store.read_state([int(window[-1])], "t_plus_1")[0]
                    )
                    window_regimes.append(
                        _regimes(self.store, [int(window[0])])[0]
                    )
                global_row = rollout_metric_row(
                        np.asarray(predicted),
                        np.asarray(target),
                        model_name=model_name,
                        split=split,
                        horizon_ms=horizon,
                        voltage_width=642,
                        layout=self.layout,
                    )
                global_row["regime"] = "all"
                self.rollout_rows.append(global_row)
                window_regimes_array = np.asarray(window_regimes)
                for regime in sorted(set(window_regimes)):
                    selected = window_regimes_array == regime
                    regime_row = rollout_metric_row(
                        np.asarray(predicted)[selected],
                        np.asarray(target)[selected],
                        model_name=model_name,
                        split=split,
                        horizon_ms=horizon,
                        voltage_width=642,
                        layout=self.layout,
                    )
                    regime_row["regime"] = regime
                    self.rollout_rows.append(regime_row)

    def _metric_value(
        self,
        model_contains: str,
        splits: Sequence[str],
        scope: str,
        name: str,
        key: str,
    ) -> float:
        values = [
            float(row[key])
            for row in self.one_step_rows
            if model_contains in str(row["model"])
            and row["split"] in splits
            and row["scope"] == scope
            and row["name"] == name
            and key in row
        ]
        return float(np.mean(values)) if values else math.inf

    def _event_f1_value(
        self, model_contains: str, splits: Sequence[str]
    ) -> float:
        values = [
            float(row["f1"])
            for row in self.event_rows
            if model_contains in str(row["model"])
            and row["split"] in splits
            and int(row.get("support", 0)) > 0
        ]
        return float(np.mean(values)) if values else 0.0

    def _rollout_value(
        self,
        model_contains: str,
        splits: Sequence[str],
        horizon_ms: int,
        key: str = "voltage_rmse_mv",
    ) -> float:
        values = [
            float(row[key])
            for row in self.rollout_rows
            if model_contains in str(row["model"])
            and row["split"] in splits
            and int(row["horizon_ms"]) == int(horizon_ms)
            and row.get("regime", "all") == "all"
            and key in row
        ]
        return float(np.mean(values)) if values else math.inf

    def _best_model(self) -> str:
        candidates: Dict[str, List[float]] = {}
        for row in self.one_step_rows:
            if row["split"] == "validation" and row["scope"] == "category" and row["name"] == "voltage":
                candidates.setdefault(str(row["model"]), []).append(float(row["rmse"]))
        return min(candidates, key=lambda key: np.mean(candidates[key]))

    def _ablation_summary(self) -> List[Dict[str, Any]]:
        test_splits = ("deterministic_test", "event_boundary_test", "branching_test")
        groups = {
            "A_voltage_only": "B3_structured-voltage_only-none-P0",
            "B_voltage_events_U1": "B3_structured-voltage_only-U1-P0",
            "B_voltage_events_U2": "B3_structured-voltage_only-U2-P0",
            "C_full_events_U1": "B3_structured-full-U1-P0",
            "C_full_events_U2": "B3_structured-full-U2-P0",
            "D_full_events_U2_privileged": "B3_structured-full-U2-P1",
        }
        rows = []
        for ablation, fragment in groups.items():
            rows.append(
                {
                    "ablation": ablation,
                    "model_fragment": fragment,
                    "validation_voltage_rmse": self._metric_value(
                        fragment, ("validation",), "category", "voltage", "rmse"
                    ),
                    "test_voltage_rmse": self._metric_value(
                        fragment, test_splits, "category", "voltage", "rmse"
                    ),
                }
            )
        return rows

    def _decision_inputs(self) -> Dict[str, Any]:
        test_splits = ("deterministic_test", "event_boundary_test", "branching_test")
        b3 = "B3_structured-full-U2-P1"
        p0 = "B3_structured-full-U2-P0"
        voltage = "B3_structured-voltage_only-U2-P0"
        b3_rmse = self._metric_value(b3, test_splits, "category", "voltage", "rmse")
        p0_rmse = self._metric_value(p0, test_splits, "category", "voltage", "rmse")
        voltage_rmse = self._metric_value(voltage, test_splits, "category", "voltage", "rmse")
        persistence = self._metric_value(
            "B0_persistence", test_splits, "category", "voltage", "rmse"
        )
        affine = min(
            self._metric_value(
                "B1_affine_delta-full-U1", test_splits, "category", "voltage", "rmse"
            ),
            self._metric_value(
                "B1_affine_delta-full-U2", test_splits, "category", "voltage", "rmse"
            ),
        )
        p1_rollout = self._rollout_value(b3, test_splits, 16)
        p0_rollout = self._rollout_value(p0, test_splits, 16)
        p1_event_f1 = self._event_f1_value(b3, test_splits)
        p0_event_f1 = self._event_f1_value(p0, test_splits)
        privileged_rollout_gain = (
            (p0_rollout - p1_rollout) / max(p0_rollout, 1e-12)
            if math.isfinite(p0_rollout) and math.isfinite(p1_rollout)
            else 0.0
        )
        rollout_16 = [
            row
            for row in self.rollout_rows
            if b3 in str(row["model"])
            and row["horizon_ms"] == 16
            and row["split"] in test_splits
            and row.get("regime", "all") == "all"
        ]
        return {
            "b3_test_voltage_rmse": b3_rmse,
            "persistence_test_voltage_rmse": persistence,
            "affine_test_voltage_rmse": affine,
            "b3_rollout_16ms_bounded": bool(rollout_16)
            and all(row["numerically_finite"] for row in rollout_16)
            and all(row["maximum_absolute_state_value"] < 1e9 for row in rollout_16),
            "b3_macro_event_f1": p1_event_f1,
            "full_state_gain_fraction": float(
                (voltage_rmse - p0_rmse) / max(voltage_rmse, 1e-12)
            ),
            "privileged_gain_fraction": float(
                (p0_rmse - b3_rmse) / max(p0_rmse, 1e-12)
            ),
            "privileged_rollout_gain_fraction": float(privileged_rollout_gain),
            "privileged_event_f1_gain": float(p1_event_f1 - p0_event_f1),
            "branching_futures_distinguished": bool(
                self.branching_report.get("distinguishes_different_futures", False)
            ),
        }

    def _report_summaries(self) -> Dict[str, Any]:
        test_splits = (
            "deterministic_test",
            "event_boundary_test",
            "branching_test",
        )
        u1 = "B3_structured-full-U1-P0"
        u2 = "B3_structured-full-U2-P0"
        p1 = "B3_structured-full-U2-P1"

        def comparison(left: str, right: str) -> Dict[str, Any]:
            return {
                "left_model": left,
                "right_model": right,
                "test_voltage_rmse_mv": {
                    "left": self._metric_value(
                        left, test_splits, "category", "voltage", "rmse"
                    ),
                    "right": self._metric_value(
                        right, test_splits, "category", "voltage", "rmse"
                    ),
                },
                "test_supported_event_macro_f1": {
                    "left": self._event_f1_value(left, test_splits),
                    "right": self._event_f1_value(right, test_splits),
                },
                "test_rollout_16ms_voltage_rmse_mv": {
                    "left": self._rollout_value(left, test_splits, 16),
                    "right": self._rollout_value(right, test_splits, 16),
                },
            }

        regimes: Dict[str, List[float]] = {}
        for row in self.one_step_rows:
            if (
                p1 in str(row["model"])
                and row["split"] in test_splits
                and row["scope"] == "voltage_regime"
            ):
                regimes.setdefault(str(row["name"]), []).append(float(row["rmse"]))

        rollout = []
        for horizon in self.config.rollout_horizons_ms:
            matching = [
                row
                for row in self.rollout_rows
                if p1 in str(row["model"])
                and row["split"] in test_splits
                and int(row["horizon_ms"]) == int(horizon)
                and row.get("regime", "all") == "all"
            ]
            if matching:
                rollout.append(
                    {
                        "horizon_ms": int(horizon),
                        "voltage_rmse_mv": float(
                            np.mean([row["voltage_rmse_mv"] for row in matching])
                        ),
                        "rest_drift_mv": float(
                            np.mean([row["voltage_rest_drift_mv"] for row in matching])
                        ),
                        "all_finite": all(row["numerically_finite"] for row in matching),
                        "maximum_outside_domain_fraction": float(
                            max(row.get("outside_domain_fraction", 0.0) for row in matching)
                        ),
                    }
                )

        gaps = {}
        for model_id, history in self.training_history.items():
            if history:
                best = min(history, key=lambda row: row["validation"]["total"])
                gaps[model_id] = {
                    "best_epoch": int(best["epoch"]),
                    "train_objective": float(best["train"]["total"]),
                    "validation_objective": float(best["validation"]["total"]),
                    "validation_minus_train": float(
                        best["validation"]["total"] - best["train"]["total"]
                    ),
                }
        return {
            "u1_vs_u2": comparison(u1, u2),
            "p0_vs_p1_privileged": comparison(u2, p1),
            "reference_b3_test_voltage_rmse_by_regime": {
                key: float(np.mean(values)) for key, values in sorted(regimes.items())
            },
            "reference_b3_rollout_stability": rollout,
            "train_validation_gap_at_best_epoch": gaps,
        }

    def _make_figures(self) -> None:
        import matplotlib.pyplot as plt

        if self.rollout_rows:
            selected = [
                row
                for row in self.rollout_rows
                if "B0_persistence" in row["model"]
                or "B1_affine_delta-full-U2" in row["model"]
                or "B3_structured-full-U2-P1" in row["model"]
            ]
            selected = [row for row in selected if row.get("regime", "all") == "all"]
            fig, axis = plt.subplots(figsize=(8, 5))
            for model in sorted({row["model"] for row in selected}):
                rows = sorted(
                    (row for row in selected if row["model"] == model),
                    key=lambda row: row["horizon_ms"],
                )
                by_horizon: Dict[int, List[float]] = {}
                for row in rows:
                    by_horizon.setdefault(int(row["horizon_ms"]), []).append(
                        float(row["voltage_rmse_mv"])
                    )
                x = sorted(by_horizon)
                y = [np.mean(by_horizon[value]) for value in x]
                axis.plot(x, y, marker="o", label=model)
            axis.set_xlabel("rollout horizon (ms)")
            axis.set_ylabel("voltage RMSE (mV)")
            axis.set_title("Short autoregressive rollout error")
            axis.grid(alpha=0.3)
            axis.legend(fontsize=6)
            fig.tight_layout()
            fig.savefig(self.figure_dir / "rollout_voltage_rmse.png", dpi=180)
            plt.close(fig)

        fig, axis = plt.subplots(figsize=(8, 5))
        for model_id, history in self.training_history.items():
            if not history:
                continue
            axis.plot(
                [row["epoch"] for row in history],
                [row["validation"]["total"] for row in history],
                label=model_id,
            )
        axis.set_xlabel("epoch")
        axis.set_ylabel("validation objective")
        axis.set_yscale("log")
        axis.grid(alpha=0.3)
        axis.legend(fontsize=5)
        fig.tight_layout()
        fig.savefig(self.figure_dir / "validation_learning_curves.png", dpi=180)
        plt.close(fig)

        for path in sorted(self.prediction_dir.glob("rollout_*_16ms.npz")):
            with np.load(path) as data:
                predicted = data["predicted_voltage_mv"]
                target = data["target_voltage_mv"]
                probes = data["probe_segment_ids"]
                regime = str(data["regime"])
            fig, axes = plt.subplots(len(probes), 1, figsize=(9, 10), sharex=True)
            time_ms = np.arange(1, len(predicted) + 1)
            for axis, segment_id in zip(axes, probes):
                axis.plot(
                    time_ms,
                    target[:, int(segment_id)],
                    color="black",
                    linewidth=1.5,
                    label="teacher",
                )
                axis.plot(
                    time_ms,
                    predicted[:, int(segment_id)],
                    color="tab:blue",
                    linewidth=1.2,
                    label="B3 rollout",
                )
                axis.set_ylabel(f"seg {int(segment_id)}\n(mV)")
                axis.grid(alpha=0.25)
            axes[0].set_title(f"16 ms rollout — {regime}")
            axes[0].legend()
            axes[-1].set_xlabel("autoregressive horizon (ms)")
            fig.tight_layout()
            fig.savefig(
                self.figure_dir / f"{path.stem}.png", dpi=180
            )
            plt.close(fig)

    def finalize(self) -> Dict[str, Any]:
        self.ablation_rows = self._ablation_summary()
        write_parquet(self.output_dir / "one_step_metrics.parquet", self.one_step_rows)
        write_parquet(self.output_dir / "rollout_metrics.parquet", self.rollout_rows)
        write_parquet(self.output_dir / "event_metrics.parquet", self.event_rows)
        write_parquet(self.output_dir / "ablation_metrics.parquet", self.ablation_rows)
        write_json(self.output_dir / "model_registry.json", self.model_registry)
        write_json(self.output_dir / "training_history.json", self.training_history)
        self._make_figures()
        decision_inputs = self._decision_inputs()
        decision = decide_go_no_go(decision_inputs)
        summaries = self._report_summaries()
        b3_parameters = [
            row["parameter_to_train_transition_ratio"]
            for name, row in self.model_registry.items()
            if "B3_structured" in name
        ]
        report = {
            "schema_version": "0.1.0",
            "valid": True,
            "best_baseline_by_validation_voltage_rmse": self._best_model(),
            "decision": decision,
            "comparison_inputs": decision_inputs,
            "comparison_summary": summaries,
            "branching_test": self.branching_report,
            "ablation_summary": self.ablation_rows,
            "parameter_to_train_transition_ratio": {
                "minimum_b3": min(b3_parameters, default=math.nan),
                "maximum_b3": max(b3_parameters, default=math.nan),
                "interpretation": (
                    "Ratios are deliberately reported because 1224 transitions "
                    "cannot establish high-dimensional generalization."
                ),
            },
            "rng_and_release_contract": self.store.release_contract,
            "synapse_state_update_contract": {
                "exact_updater_used": False,
                "comparison_available": False,
                "reason": (
                    "No exact updater has been validated independently of NEURON; "
                    "the synapse-state block is therefore learned in B1-B3."
                ),
            },
            "blockers": decision["blockers"],
            "limitations": [
                "The dataset is diagnostic and intentionally event enriched.",
                "Release outcomes are unavailable to the surrogate.",
                "A good training loss is not evidence that HayFlow is solved.",
            ],
            "next_step_policy": {
                "GO": "proceed to a structured HayFlow prototype",
                "CONDITIONAL_GO": "add rollout-aware structure before scaling",
                "NO_GO": "increase diagnostic data or revise intra-ms inputs",
            }[decision["decision"]],
        }
        write_json(self.output_dir / "final_report.json", report)
        return report


def _file_sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit(repository: Path) -> Optional[str]:
    import subprocess

    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _runtime_versions() -> Dict[str, Optional[str]]:
    import importlib.metadata
    import platform

    versions: Dict[str, Optional[str]] = {"python": platform.python_version()}
    for distribution in ("numpy", "h5py", "torch", "pandas", "pyarrow"):
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[distribution] = None
    return versions


def _clear_cuda_cache() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
