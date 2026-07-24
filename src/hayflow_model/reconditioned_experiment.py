"""Methodologically reconditioned full-state flow-map experiment (02b)."""

from __future__ import annotations

import hashlib
import json
import math
import random
import shutil
import subprocess
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
    ReconditionedAuxiliaryNormalizer,
    ReconditionedStateNormalizer,
    ReconditioningConfig,
    distribution_summary,
)
from ..hayflow_eval import (
    binary_event_metric_rows,
    rollout_metric_row,
    state_metric_rows,
    write_parquet,
)
from .full_state_flowmap import (
    DualRidgeBaseline,
    FlowmapModelConfig,
    PersistenceBaseline,
    parameter_count,
    require_torch,
    ridge_design_matrix,
    structured_arrays,
)
from .reconditioned_full_state import ReconditionedStructuredResidual


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_hash(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _code_commit() -> str:
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


def _seed(seed: int, deterministic: bool) -> None:
    require_torch()
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)


def _clear_cuda() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        return


def _regime(store: FlowmapTransitionStore, index: int) -> str:
    category = str(store.metadata["category"][index])
    protocol = str(store.metadata["protocol"][index])
    if bool(store.metadata["negative_control"][index]):
        return "event_boundary_negative"
    if "plateau" in protocol:
        return "nmda_plateau"
    if "calcium" in protocol or "bap" in protocol:
        return "calcium_bac"
    if category == "somatic_events":
        return "somatic_spike"
    if category == "rest_subthreshold":
        return "rest_subthreshold"
    if category == "branching":
        return "branching"
    return category


@dataclass(frozen=True)
class ReconditionedExperimentConfig:
    profile: str = "diagnostic_full"
    initialization_seeds: Tuple[int, ...] = (17, 29, 43)
    maximum_epochs: int = 60
    early_stopping_patience: int = 10
    batch_size: int = 2
    evaluation_batch_size: int = 2
    learning_rate: float = 2e-4
    weight_decay: float = 1e-5
    gradient_clip_norm: float = 1.0
    ridge_alpha: float = 10.0
    activity_epsilon: float = 1e-9
    sparse_update_fraction: float = 0.10
    minimum_scale: float = 1e-8
    oversample_dendritic_factor: int = 6
    lambda_voltage_subthreshold: float = 0.5
    lambda_voltage_spike: float = 1.0
    lambda_voltage_dendritic: float = 1.0
    lambda_voltage_boundary: float = 0.75
    lambda_mechanism_state: float = 0.20
    lambda_calcium: float = 0.30
    lambda_synapse_state: float = 0.05
    lambda_sparse_activity: float = 0.10
    lambda_sparse_value: float = 0.20
    lambda_event: float = 0.30
    lambda_event_timing: float = 0.03
    lambda_event_region: float = 0.01
    privileged_fixed_weight: float = 0.005
    privileged_gradient_fraction: float = 0.25
    rollout_horizons_ms: Tuple[int, ...] = (2, 4, 8, 16)
    selection_rollout_windows: int = 8
    maximum_rollout_windows_per_split: int = 32
    deterministic_algorithms: bool = True

    def effective(self) -> "ReconditionedExperimentConfig":
        if self.profile not in {"smoke", "diagnostic_full"}:
            raise ValueError("profile must be smoke or diagnostic_full")
        if self.profile == "diagnostic_full":
            return self
        values = asdict(self)
        values.update(
            initialization_seeds=(self.initialization_seeds[0],),
            maximum_epochs=2,
            early_stopping_patience=1,
            rollout_horizons_ms=(2,),
            maximum_rollout_windows_per_split=2,
            selection_rollout_windows=1,
        )
        return ReconditionedExperimentConfig(**values)


@dataclass(frozen=True)
class ReconditionedRunSpec:
    input_encoding: str
    synapse_mode: str
    privileged_mode: str
    gate_transform: str = "logit"

    def validate(self) -> None:
        if self.input_encoding not in {"U1", "U2"}:
            raise ValueError("input encoding must be U1 or U2")
        if self.synapse_mode not in {"exclude", "hurdle"}:
            raise ValueError("synapse mode must be exclude or hurdle")
        if self.privileged_mode not in {"P0", "P1a", "P1b"}:
            raise ValueError("privileged mode must be P0, P1a or P1b")
        if self.gate_transform not in {"identity", "logit"}:
            raise ValueError("gate transform must be identity or logit")

    @property
    def privileged(self) -> bool:
        return self.privileged_mode != "P0"

    def identifier(self, seed: int) -> str:
        self.validate()
        return (
            f"B3_reconditioned-full-{self.input_encoding}-"
            f"{self.synapse_mode}-{self.privileged_mode}-"
            f"gates_{self.gate_transform}-seed{seed}"
        )


class Progress:
    def __init__(self, label: str, total: int) -> None:
        self.label = label
        self.total = max(1, total)
        self.started = time.monotonic()

    def update(self, current: int, detail: str = "") -> None:
        elapsed = time.monotonic() - self.started
        rate = current / elapsed if current and elapsed else 0.0
        eta = (self.total - current) / rate if rate else math.inf
        eta_text = "?" if not math.isfinite(eta) else f"{eta / 60:.1f} min"
        print(
            f"[HayFlow 02b][{self.label}] {current}/{self.total} "
            f"({100 * current / self.total:.1f}%) ETA {eta_text} {detail}",
            flush=True,
        )


class ReconditionedFlowmapExperiment:
    """Run the frozen B0/B1/B3 comparison under a valid loss contract."""

    def __init__(
        self,
        bundle: FlowmapBundle,
        original_result_root: Path,
        output_dir: Path,
        config: ReconditionedExperimentConfig,
    ) -> None:
        self.bundle = bundle
        self.original_root = Path(original_result_root).resolve()
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
        self.normalizers: Dict[str, ReconditionedStateNormalizer] = {}
        self.aux_normalizer: Optional[ReconditionedAuxiliaryNormalizer] = None
        self.aux_layout: Dict[str, slice] = {}
        self.original_report = self._load_original_result()
        self.code_commit = _code_commit()
        self.learnable_event_mask: Optional[np.ndarray] = None
        self.event_support: Dict[str, int] = {}
        self.one_step_rows: List[Dict[str, Any]] = []
        self.event_rows: List[Dict[str, Any]] = []
        self.rollout_rows: List[Dict[str, Any]] = []
        self.gradient_rows: List[Dict[str, Any]] = []
        self.gradient_partial_path = self.output_dir / "gradient_contribution_report.partial.json"
        if self.gradient_partial_path.is_file():
            partial = json.loads(
                self.gradient_partial_path.read_text(encoding="utf-8")
            )
            if partial.get("code_commit") == self.code_commit:
                self.gradient_rows = list(partial.get("rows", []))
        self.seed_rows: List[Dict[str, Any]] = []
        self.branching_rows: List[Dict[str, Any]] = []
        self.registry: Dict[str, Dict[str, Any]] = {}
        self.histories: Dict[str, List[Dict[str, Any]]] = {}
        self.checkpoint_contracts: Dict[str, Dict[str, Any]] = {}
        self.conditioning_diagnostics: Dict[str, Any] = {}

    def _load_original_result(self) -> Dict[str, Any]:
        required = (
            "final_report.json",
            "experiment_manifest.json",
            "one_step_metrics.parquet",
            "rollout_metrics.parquet",
            "event_metrics.parquet",
        )
        missing = [name for name in required if not (self.original_root / name).is_file()]
        if missing:
            raise RuntimeError(f"original notebook 02 result is incomplete: {missing}")
        manifest = json.loads(
            (self.original_root / "experiment_manifest.json").read_text(encoding="utf-8")
        )
        if manifest.get("dataset_manifest_sha256") != _sha256(
            self.bundle.root / "dataset_manifest.json"
        ):
            raise RuntimeError("original 02 result used a different dataset manifest")
        report = json.loads(
            (self.original_root / "final_report.json").read_text(encoding="utf-8")
        )
        if not report.get("valid"):
            raise RuntimeError("original notebook 02 report is not valid")
        return report

    def _run_specs(self) -> List[Tuple[ReconditionedRunSpec, Tuple[int, ...]]]:
        seeds = self.config.initialization_seeds
        specs = [
            ReconditionedRunSpec("U1", "hurdle", "P0"),
            ReconditionedRunSpec("U2", "exclude", "P0"),
            ReconditionedRunSpec("U2", "hurdle", "P0"),
            ReconditionedRunSpec("U2", "hurdle", "P1a"),
            ReconditionedRunSpec("U2", "hurdle", "P1b"),
        ]
        result = [(spec, seeds) for spec in specs]
        if self.config.profile == "diagnostic_full":
            result.append(
                (ReconditionedRunSpec("U2", "hurdle", "P0", "identity"), (seeds[0],))
            )
        return result

    def prepare(self) -> Dict[str, Any]:
        train = self.store.split_indices["train"]
        state_t = self.store.read_state(train, "t")
        state_t1 = self.store.read_state(train, "t_plus_1")
        state_audit_rows: List[Dict[str, Any]] = []
        for gate_transform in ("logit", "identity"):
            normalizer = ReconditionedStateNormalizer(
                self.layout,
                ReconditioningConfig(
                    activity_epsilon=self.config.activity_epsilon,
                    sparse_update_fraction=self.config.sparse_update_fraction,
                    minimum_scale=self.config.minimum_scale,
                    gate_transform=gate_transform,
                ),
            ).fit(state_t, state_t1)
            self.normalizers[gate_transform] = normalizer
            if gate_transform == "logit":
                state_audit_rows = normalizer.audit_rows
                normalized_delta, activity = normalizer.delta_and_activity(
                    state_t, state_t1
                )
                family_rows = []
                for category in DYNAMIC_CATEGORIES:
                    state_slice = self.layout.category_slices[category]
                    selected = np.abs(normalized_delta[:, state_slice])
                    selected_activity = activity[:, state_slice]
                    active_values = selected[selected_activity]
                    family_rows.append(
                        {
                            "family": category,
                            "active_value_count": int(len(active_values)),
                            "active_normalized_absolute_p99": (
                                float(np.percentile(active_values, 99.0))
                                if len(active_values)
                                else 0.0
                            ),
                            "active_normalized_absolute_maximum": (
                                float(np.max(active_values))
                                if len(active_values)
                                else 0.0
                            ),
                            "minimum_scale": float(
                                np.min(normalizer.delta_scale[state_slice])
                            ),
                        }
                    )
                self.conditioning_diagnostics = {
                    "fit_split": "train",
                    "families": family_rows,
                    "all_scales_respect_floor": bool(
                        np.all(
                            normalizer.delta_scale
                            >= self.config.minimum_scale
                        )
                    ),
                    "maximum_active_normalized_absolute_value": float(
                        max(
                            row["active_normalized_absolute_maximum"]
                            for row in family_rows
                        )
                    ),
                }

        auxiliary, aux_layout = self.store.auxiliary_targets(train)
        self.aux_layout = aux_layout
        self.aux_normalizer = ReconditionedAuxiliaryNormalizer(
            self.config.minimum_scale
        ).fit(
            auxiliary,
            aux_layout,
            privileged_records=self.layout.privileged_records,
        )
        train_events = self.store.event_targets(train)["event_presence"]
        support = train_events.sum(axis=0).astype(int)
        self.learnable_event_mask = support > 0
        self.event_support = {
            kind: int(support[index]) for index, kind in enumerate(EVENT_KINDS)
        }

        write_parquet(
            self.output_dir / "distribution_audit_state_variables.parquet",
            state_audit_rows,
        )
        write_parquet(
            self.output_dir / "distribution_audit_privileged_variables.parquet",
            self.aux_normalizer.audit_rows,
        )
        audit_summary = {
            "schema_version": "02b-distribution-audit-v1",
            "fit_split": "train",
            "zero_definition": (
                f"absolute transformed delta <= {self.config.activity_epsilon:g}"
            ),
            "train_transition_count": len(train),
            "state": distribution_summary(state_audit_rows),
            "privileged": distribution_summary(self.aux_normalizer.audit_rows),
            "event_support": self.event_support,
            "conditioning_diagnostics": self.conditioning_diagnostics,
            "not_learnable_event_classes": [
                kind for kind, count in self.event_support.items() if count == 0
            ],
        }
        _write_json(self.output_dir / "distribution_audit.json", audit_summary)
        normalization_payload = {
            "schema_version": "02b-normalization-v1",
            "state": {
                key: normalizer.to_dict()
                for key, normalizer in self.normalizers.items()
            },
            "privileged": self.aux_normalizer.to_dict(),
        }
        _write_json(self.output_dir / "normalization_schema.json", normalization_payload)
        loss_schema = {
            "schema_version": "02b-loss-v1",
            "config": asdict(self.config),
            "synapse_variants": {
                "S0": "excluded from main loss and early stopping; metric only",
                "S1": (
                    "dense synaptic deltas use a small separate weight; sparse "
                    "deltas use activity BCE plus active-only normalized regression"
                ),
            },
            "privileged_variants": {
                "P0": "disabled",
                "P1a": "fixed normalized auxiliary weight",
                "P1b": "shared-representation gradient contribution capped",
            },
            "early_stopping": (
                "operational validation score: common/event voltage, learnable-event "
                "macro-F1, rest drift and physical-domain penalty; synapse and "
                "privileged losses excluded"
            ),
            "event_support": self.event_support,
        }
        _write_json(self.output_dir / "loss_schema.json", loss_schema)
        manifest = {
            "schema_version": "02b-experiment-v1",
            "experiment": "reconditioned_full_state_flowmap_baseline",
            "dataset_schema_version": self.bundle.schema_version,
            "dataset_manifest_sha256": _sha256(
                self.bundle.root / "dataset_manifest.json"
            ),
            "original_02_final_report_sha256": _sha256(
                self.original_root / "final_report.json"
            ),
            "code_commit": self.code_commit,
            "split_indices_sha256": self._split_hash(),
            "architecture_frozen": True,
            "new_trajectories": False,
            "release_outcome_available": self.store.release_contract[
                "release_outcome_available"
            ],
            "config": asdict(self.config),
        }
        _write_json(self.output_dir / "experiment_manifest.json", manifest)
        return {
            "train_transition_count": len(train),
            "event_support": self.event_support,
            "normalizer_fingerprints": {
                key: value.fingerprint() for key, value in self.normalizers.items()
            },
            "privileged_fingerprint": self.aux_normalizer.fingerprint(),
            "distribution_summary": audit_summary,
        }

    def _split_hash(self) -> str:
        digest = hashlib.sha256()
        for split, indices in sorted(self.store.split_indices.items()):
            digest.update(split.encode())
            digest.update(np.ascontiguousarray(indices).tobytes())
        return digest.hexdigest()

    def _model_config(self, spec: ReconditionedRunSpec) -> FlowmapModelConfig:
        dense_width = max(value.stop for value in self.aux_layout.values()) - self.aux_layout[
            "currents_conductances_t_plus_1"
        ].stop
        return FlowmapModelConfig(
            "B3_structured",
            "full",
            spec.input_encoding,
            privileged_loss=spec.privileged,
            auxiliary_dense_dim=dense_width if spec.privileged else 0,
        )

    def preflight(self) -> Dict[str, Any]:
        require_torch()
        import torch

        if not self.normalizers:
            self.prepare()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checks = []
        sample = self.store.split_indices["train"][:2]
        fingerprint_fields = {
            "dataset_manifest_sha256",
            "dataset_schema",
            "split_indices_sha256",
            "state_normalizer_fingerprint",
            "auxiliary_normalizer_fingerprint",
            "loss_config",
            "run_spec",
            "model_config",
            "code_commit",
            "seed",
            "fingerprint",
        }
        for spec, seeds in self._run_specs():
            model = self._build_model(spec, device)
            batch = self._load_batch(sample, spec, device)
            with torch.no_grad():
                output = model(batch)
            contract = self._checkpoint_contract(spec, seeds[0])
            fingerprint_valid = (
                fingerprint_fields <= set(contract)
                and contract["fingerprint"]
                == _stable_hash(
                    {
                        key: value
                        for key, value in contract.items()
                        if key != "fingerprint"
                    }
                )
            )
            valid = (
                tuple(output["delta"].shape) == (len(sample), self.layout.state_width)
                and tuple(output["activity_logits"].shape)
                == (len(sample), self.layout.state_width)
                and bool(torch.isfinite(output["delta"]).all())
                and fingerprint_valid
            )
            checks.append(
                {
                    "spec": asdict(spec),
                    "parameter_count": parameter_count(model),
                    "checkpoint_fingerprint_valid": bool(fingerprint_valid),
                    "valid": bool(valid),
                }
            )
            del model, batch, output
            _clear_cuda()
        release_outcome_leakage = bool(
            self.store.release_contract["release_outcome_available"]
        )
        report = {
            "valid": all(row["valid"] for row in checks)
            and bool(self.store.rollout_windows("validation", 8))
            and not release_outcome_leakage,
            "device": str(device),
            "checks": checks,
            "validation_rollout_8ms_window_count": len(
                self.store.rollout_windows("validation", 8)
            ),
            "future_targets_excluded_from_inputs": True,
            "release_outcome_leakage": release_outcome_leakage,
            "checkpoint_fingerprint_fields_complete": all(
                row["checkpoint_fingerprint_valid"] for row in checks
            ),
        }
        _write_json(self.output_dir / "preflight_report.json", report)
        if not report["valid"]:
            raise RuntimeError(f"02b preflight failed: {report}")
        return report

    def _build_model(self, spec: ReconditionedRunSpec, device: Any) -> Any:
        model = ReconditionedStructuredResidual(
            self._model_config(spec),
            self.layout.to_model_metadata(),
            structured_arrays(self.layout),
        )
        return model.to(device)

    def _batch_numpy(
        self,
        indices: Sequence[int],
        spec: ReconditionedRunSpec,
        *,
        include_auxiliary: bool = False,
        raw_state_override: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        indices = np.asarray(indices, dtype=np.int64)
        normalizer = self.normalizers[spec.gate_transform]
        raw_t = (
            np.asarray(raw_state_override, dtype=np.float64)
            if raw_state_override is not None
            else self.store.read_state(indices, "t")
        )
        raw_t1 = self.store.read_state(indices, "t_plus_1")
        delta, activity = normalizer.delta_and_activity(raw_t, raw_t1)
        batch: Dict[str, Any] = {
            "indices": indices,
            "raw_state_t": raw_t,
            "raw_state_t_plus_1": raw_t1,
            "state_t": normalizer.normalize_state(raw_t).astype(np.float32),
            "delta_target": delta,
            "activity_target": activity,
        }
        batch.update(self.store.encode_inputs(indices))
        batch.update(self.store.event_targets(indices))
        regimes = [_regime(self.store, int(index)) for index in indices]
        batch["regimes"] = regimes
        batch["negative_control"] = np.asarray(
            [bool(self.store.metadata["negative_control"][int(index)]) for index in indices]
        )
        if include_auxiliary:
            raw_aux, _ = self.store.auxiliary_targets(indices)
            normalized, applicable = self.aux_normalizer.transform(raw_aux)
            batch["auxiliary_target"] = normalized
            batch["auxiliary_mask"] = applicable
        return batch

    def _load_batch(
        self,
        indices: Sequence[int],
        spec: ReconditionedRunSpec,
        device: Any,
        *,
        raw_state_override: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        import torch

        raw = self._batch_numpy(
            indices,
            spec,
            include_auxiliary=spec.privileged,
            raw_state_override=raw_state_override,
        )
        result: Dict[str, Any] = {}
        integer = {"indices", "u2_segment_ids", "event_region"}
        boolean = {
            "u2_mask",
            "event_timing_mask",
            "event_region_mask",
            "activity_target",
            "negative_control",
            "auxiliary_mask",
        }
        excluded = {"raw_state_t", "raw_state_t_plus_1", "regimes"}
        for key, value in raw.items():
            if key in excluded:
                result[key] = value
            elif isinstance(value, np.ndarray):
                dtype = (
                    torch.long
                    if key in integer
                    else torch.bool
                    if key in boolean
                    else torch.float32
                )
                result[key] = torch.as_tensor(value, dtype=dtype, device=device)
            else:
                result[key] = value
        return result

    def _event_weights(self) -> np.ndarray:
        target = self.store.event_targets(self.store.split_indices["train"])[
            "event_presence"
        ]
        positives = target.sum(axis=0)
        negatives = len(target) - positives
        weights = np.clip(negatives / np.maximum(positives, 1.0), 1.0, 100.0)
        weights[~self.learnable_event_mask] = 0.0
        return weights.astype(np.float32)

    @staticmethod
    def _masked_huber(prediction: Any, target: Any, mask: Any) -> Any:
        import torch

        if not bool(mask.any()):
            return prediction.new_zeros(())
        return torch.nn.functional.smooth_l1_loss(prediction[mask], target[mask])

    def _family_hurdle_loss(
        self,
        output: Mapping[str, Any],
        batch: Mapping[str, Any],
        columns: np.ndarray,
        normalizer: ReconditionedStateNormalizer,
        *,
        include_sparse: bool,
    ) -> Tuple[Any, Any, Any]:
        import torch

        device = output["delta"].device
        column_tensor = torch.as_tensor(columns, dtype=torch.long, device=device)
        sparse = torch.as_tensor(
            normalizer.sparse_mask[columns], dtype=torch.bool, device=device
        )
        dense = ~sparse
        prediction = output["delta"][:, column_tensor]
        target = batch["delta_target"][:, column_tensor]
        activity_target = batch["activity_target"][:, column_tensor]
        dense_loss = self._masked_huber(
            prediction, target, dense.unsqueeze(0).expand_as(prediction)
        )
        zero = prediction.new_zeros(())
        if not include_sparse or not bool(sparse.any()):
            return dense_loss, zero, zero
        logits = output["activity_logits"][:, column_tensor][:, sparse]
        active = activity_target[:, sparse]
        positive_weight = torch.as_tensor(
            normalizer.activity_positive_weight[columns][sparse.cpu().numpy()],
            dtype=torch.float32,
            device=device,
        )
        activity_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            logits,
            active.float(),
            pos_weight=positive_weight,
        )
        value_loss = self._masked_huber(
            prediction[:, sparse], target[:, sparse], active
        )
        return dense_loss, activity_loss, value_loss

    def _component_losses(
        self,
        output: Mapping[str, Any],
        batch: Mapping[str, Any],
        spec: ReconditionedRunSpec,
        event_pos_weight: Any,
    ) -> Dict[str, Any]:
        import torch

        normalizer = self.normalizers[spec.gate_transform]
        voltage_slice = self.layout.category_slices["voltage"]
        voltage_prediction = output["delta"][:, voltage_slice]
        voltage_target = batch["delta_target"][:, voltage_slice]
        presence = batch["event_presence"].bool()
        somatic_rows = presence[:, 0] | presence[:, 1]
        dendritic_rows = presence[:, 2:].any(dim=1)
        boundary_rows = batch["negative_control"]
        subthreshold_rows = ~(somatic_rows | dendritic_rows | boundary_rows)

        def row_voltage(mask: Any, include_mse: bool) -> Any:
            if not bool(mask.any()):
                return voltage_prediction.new_zeros(())
            pred = voltage_prediction[mask]
            true = voltage_target[mask]
            huber = torch.nn.functional.smooth_l1_loss(pred, true)
            return huber + (0.05 * torch.mean((pred - true) ** 2) if include_mse else 0.0)

        losses: Dict[str, Any] = {
            "voltage_subthreshold": row_voltage(subthreshold_rows, False),
            "voltage_spike": row_voltage(somatic_rows, True),
            "voltage_dendritic": row_voltage(dendritic_rows, True),
            "voltage_boundary": row_voltage(boundary_rows, True),
        }
        mechanism = np.arange(
            self.layout.category_slices["mechanism_states"].start,
            self.layout.category_slices["mechanism_states"].stop,
        )
        calcium = np.arange(
            self.layout.category_slices["calcium_ions"].start,
            self.layout.category_slices["calcium_ions"].stop,
        )
        synapse = np.arange(
            self.layout.category_slices["synapse_states"].start,
            self.layout.category_slices["synapse_states"].stop,
        )
        for name, columns, include_sparse in (
            ("mechanism", mechanism, True),
            ("calcium", calcium, True),
            ("synapse", synapse, spec.synapse_mode == "hurdle"),
        ):
            dense, activity, value = self._family_hurdle_loss(
                output,
                batch,
                columns,
                normalizer,
                include_sparse=include_sparse,
            )
            if name == "synapse" and spec.synapse_mode == "exclude":
                dense = dense.new_zeros(())
            losses[f"{name}_dense"] = dense
            losses[f"{name}_activity"] = activity
            losses[f"{name}_sparse_value"] = value

        learnable = torch.as_tensor(
            self.learnable_event_mask,
            dtype=torch.bool,
            device=output["event_logits"].device,
        )
        losses["event_presence"] = torch.nn.functional.binary_cross_entropy_with_logits(
            output["event_logits"][:, learnable],
            batch["event_presence"][:, learnable],
            pos_weight=event_pos_weight[learnable],
        )
        timing_mask = batch["event_timing_mask"].clone()
        timing_mask[:, ~learnable, :] = False
        losses["event_timing"] = self._masked_huber(
            output["event_timing"], batch["event_timing"], timing_mask
        )
        region_mask = batch["event_region_mask"].clone()
        region_mask[:, ~learnable] = False
        losses["event_region"] = (
            torch.nn.functional.cross_entropy(
                output["event_region_logits"][region_mask],
                batch["event_region"][region_mask],
            )
            if bool(region_mask.any())
            else output["delta"].new_zeros(())
        )
        if spec.privileged:
            current_slice = self.aux_layout["currents_conductances_t_plus_1"]
            current_mask = batch["auxiliary_mask"][:, current_slice]
            dense_mask = batch["auxiliary_mask"][:, current_slice.stop :]
            losses["privileged_current"] = self._masked_huber(
                output["privileged_current"],
                batch["auxiliary_target"][:, current_slice],
                current_mask,
            )
            losses["privileged_dense"] = self._masked_huber(
                output["aux_dense"],
                batch["auxiliary_target"][:, current_slice.stop :],
                dense_mask,
            )
        return losses

    def _main_loss(self, losses: Mapping[str, Any]) -> Any:
        sparse_activity = (
            losses["mechanism_activity"]
            + losses["calcium_activity"]
            + losses["synapse_activity"]
        )
        sparse_value = (
            losses["mechanism_sparse_value"]
            + losses["calcium_sparse_value"]
            + losses["synapse_sparse_value"]
        )
        return (
            self.config.lambda_voltage_subthreshold * losses["voltage_subthreshold"]
            + self.config.lambda_voltage_spike * losses["voltage_spike"]
            + self.config.lambda_voltage_dendritic * losses["voltage_dendritic"]
            + self.config.lambda_voltage_boundary * losses["voltage_boundary"]
            + self.config.lambda_mechanism_state * losses["mechanism_dense"]
            + self.config.lambda_calcium * losses["calcium_dense"]
            + self.config.lambda_synapse_state * losses["synapse_dense"]
            + self.config.lambda_sparse_activity * sparse_activity
            + self.config.lambda_sparse_value * sparse_value
            + self.config.lambda_event * losses["event_presence"]
            + self.config.lambda_event_timing * losses["event_timing"]
            + self.config.lambda_event_region * losses["event_region"]
        )

    @staticmethod
    def _representation_gradient_norm(loss: Any, output: Mapping[str, Any]) -> float:
        import torch

        if not loss.requires_grad:
            return 0.0
        gradients = torch.autograd.grad(
            loss,
            (
                output["token_hidden"],
                output["segment_hidden"],
                output["global_hidden"],
            ),
            retain_graph=True,
            allow_unused=True,
        )
        squared = loss.new_zeros(())
        for gradient in gradients:
            if gradient is not None:
                squared = squared + torch.sum(gradient.detach() ** 2)
        return float(torch.sqrt(squared).cpu())

    def _total_loss(
        self,
        losses: Mapping[str, Any],
        output: Mapping[str, Any],
        spec: ReconditionedRunSpec,
    ) -> Tuple[Any, float, float, float]:
        main = self._main_loss(losses)
        if not spec.privileged:
            return main, 0.0, 0.0, 0.0
        privileged = losses["privileged_current"] + losses["privileged_dense"]
        if spec.privileged_mode == "P1a":
            return (
                main + self.config.privileged_fixed_weight * privileged,
                self.config.privileged_fixed_weight,
                0.0,
                0.0,
            )
        main_norm = self._representation_gradient_norm(main, output)
        privileged_norm = self._representation_gradient_norm(privileged, output)
        capped = self.config.privileged_gradient_fraction * main_norm / max(
            privileged_norm, 1e-12
        )
        weight = min(self.config.privileged_fixed_weight, capped)
        return main + weight * privileged, float(weight), main_norm, privileged_norm

    def _component_weight(
        self,
        name: str,
        spec: ReconditionedRunSpec,
        privileged_weight: float,
    ) -> float:
        fixed = {
            "voltage_subthreshold": self.config.lambda_voltage_subthreshold,
            "voltage_spike": self.config.lambda_voltage_spike,
            "voltage_dendritic": self.config.lambda_voltage_dendritic,
            "voltage_boundary": self.config.lambda_voltage_boundary,
            "mechanism_dense": self.config.lambda_mechanism_state,
            "calcium_dense": self.config.lambda_calcium,
            "synapse_dense": self.config.lambda_synapse_state,
            "mechanism_activity": self.config.lambda_sparse_activity,
            "calcium_activity": self.config.lambda_sparse_activity,
            "synapse_activity": self.config.lambda_sparse_activity,
            "mechanism_sparse_value": self.config.lambda_sparse_value,
            "calcium_sparse_value": self.config.lambda_sparse_value,
            "synapse_sparse_value": self.config.lambda_sparse_value,
            "event_presence": self.config.lambda_event,
            "event_timing": self.config.lambda_event_timing,
            "event_region": self.config.lambda_event_region,
        }
        if name in {"privileged_current", "privileged_dense"}:
            return float(privileged_weight)
        if name.startswith("synapse_") and spec.synapse_mode == "exclude":
            return 0.0
        return float(fixed[name])

    def _balanced_indices(self, seed: int) -> np.ndarray:
        indices = self.store.split_indices["train"]
        target = self.store.event_targets(indices)["event_presence"]
        dendritic = np.flatnonzero(target[:, 2:].any(axis=1))
        pieces = [indices]
        if len(dendritic):
            positive = indices[dendritic]
            for _ in range(max(0, self.config.oversample_dendritic_factor - 1)):
                pieces.append(positive)
        result = np.concatenate(pieces)
        np.random.default_rng(seed).shuffle(result)
        return result

    def _checkpoint_contract(
        self,
        spec: ReconditionedRunSpec,
        seed: int,
    ) -> Dict[str, Any]:
        model_config = self._model_config(spec).to_dict()
        payload = {
            "dataset_manifest_sha256": _sha256(
                self.bundle.root / "dataset_manifest.json"
            ),
            "dataset_schema": self.bundle.schema_version,
            "split_indices_sha256": self._split_hash(),
            "state_normalizer_fingerprint": self.normalizers[
                spec.gate_transform
            ].fingerprint(),
            "auxiliary_normalizer_fingerprint": (
                self.aux_normalizer.fingerprint() if spec.privileged else None
            ),
            "loss_config": asdict(self.config),
            "run_spec": asdict(spec),
            "model_config": model_config,
            "code_commit": self.code_commit,
            "seed": int(seed),
        }
        payload["fingerprint"] = _stable_hash(payload)
        return payload

    def _checkpoint_path(
        self,
        model_id: str,
        contract: Mapping[str, Any],
    ) -> Path:
        return self.checkpoint_dir / model_id / str(contract["fingerprint"])[:16]

    def _save_checkpoint(
        self,
        path: Path,
        model: Any,
        optimizer: Any,
        epoch: int,
        contract: Mapping[str, Any],
        history: Sequence[Mapping[str, Any]],
        selection: Mapping[str, Any],
        *,
        include_optimizer: bool = False,
    ) -> None:
        import torch

        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model_state": model.state_dict(),
            "epoch": int(epoch),
            "checkpoint_contract": dict(contract),
            "history": list(history),
            "selection": dict(selection),
        }
        if include_optimizer:
            payload["optimizer_state"] = optimizer.state_dict()
        torch.save(payload, path)

    def _load_checkpoint(
        self,
        path: Path,
        model: Any,
        contract: Mapping[str, Any],
        optimizer: Optional[Any] = None,
    ) -> Mapping[str, Any]:
        import torch

        saved = torch.load(path, map_location=next(model.parameters()).device)
        observed = saved.get("checkpoint_contract", {})
        if observed.get("fingerprint") != contract["fingerprint"]:
            raise RuntimeError(
                f"checkpoint fingerprint mismatch: {path}; stale checkpoints are refused"
            )
        model.load_state_dict(saved["model_state"])
        if optimizer is not None and "optimizer_state" in saved:
            optimizer.load_state_dict(saved["optimizer_state"])
        return saved

    def _iterate(self, indices: Sequence[int]) -> Iterable[np.ndarray]:
        values = np.asarray(indices, dtype=np.int64)
        for start in range(0, len(values), self.config.batch_size):
            yield values[start : start + self.config.batch_size]

    def _learnable_macro_f1(
        self,
        probabilities: np.ndarray,
        targets: np.ndarray,
        thresholds: Optional[np.ndarray] = None,
    ) -> float:
        thresholds = (
            np.full(len(EVENT_KINDS), 0.5) if thresholds is None else thresholds
        )
        predicted = probabilities >= thresholds[None, :]
        values = []
        for column, learnable in enumerate(self.learnable_event_mask):
            if not learnable:
                continue
            tp = np.sum(predicted[:, column] & (targets[:, column] > 0.5))
            fp = np.sum(predicted[:, column] & (targets[:, column] <= 0.5))
            fn = np.sum(~predicted[:, column] & (targets[:, column] > 0.5))
            precision = tp / max(1, tp + fp)
            recall = tp / max(1, tp + fn)
            values.append(2 * precision * recall / max(1e-12, precision + recall))
        return float(np.mean(values)) if values else 0.0

    def _epoch(
        self,
        model: Any,
        spec: ReconditionedRunSpec,
        indices: Sequence[int],
        device: Any,
        event_pos_weight: Any,
        *,
        optimizer: Optional[Any],
        epoch: int,
        model_id: str,
    ) -> Dict[str, Any]:
        import torch

        training = optimizer is not None
        model.train(training)
        totals: Dict[str, float] = {}
        count = 0
        voltage_prediction: List[np.ndarray] = []
        voltage_target: List[np.ndarray] = []
        event_probabilities: List[np.ndarray] = []
        event_targets: List[np.ndarray] = []
        regimes: List[str] = []
        outside = 0
        outside_denominator = 0
        for batch_number, batch_indices in enumerate(self._iterate(indices)):
            batch = self._load_batch(batch_indices, spec, device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            with torch.set_grad_enabled(training):
                output = model(batch)
                losses = self._component_losses(
                    output, batch, spec, event_pos_weight
                )
                total, priv_weight, main_grad, priv_grad = self._total_loss(
                    losses, output, spec
                )
                if training:
                    if batch_number == 0:
                        gradient_components = list(losses)
                        existing = {
                            (
                                str(row.get("model_id")),
                                int(row.get("epoch", -1)),
                                str(row.get("component")),
                            )
                            for row in self.gradient_rows
                        }
                        for name in gradient_components:
                            key = (model_id, int(epoch), name)
                            if key in existing:
                                continue
                            raw_norm = self._representation_gradient_norm(
                                losses[name], output
                            )
                            component_weight = self._component_weight(
                                name, spec, priv_weight
                            )
                            self.gradient_rows.append(
                                {
                                    "model": spec.identifier(0).rsplit("-seed", 1)[0],
                                    "model_id": model_id,
                                    "epoch": int(epoch),
                                    "component": name,
                                    "loss_weight": component_weight,
                                    "shared_representation_gradient_norm": raw_norm,
                                    "weighted_shared_representation_gradient_norm": (
                                        component_weight * raw_norm
                                    ),
                                }
                            )
                    total.backward()
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), self.config.gradient_clip_norm
                    )
                    optimizer.step()
            details = {key: float(value.detach()) for key, value in losses.items()}
            details.update(
                total=float(total.detach()),
                privileged_effective_weight=float(priv_weight),
                main_shared_gradient_norm=float(main_grad),
                privileged_shared_gradient_norm=float(priv_grad),
            )
            for key, value in details.items():
                totals[key] = totals.get(key, 0.0) + value * len(batch_indices)
            count += len(batch_indices)

            if not training:
                probability = torch.sigmoid(output["activity_logits"]).cpu().numpy()
                raw_prediction = self.normalizers[spec.gate_transform].reconstruct(
                    batch["raw_state_t"],
                    output["delta"].detach().cpu().numpy(),
                    activity_probability=probability,
                    synapse_mode=spec.synapse_mode,
                )
                voltage_prediction.append(raw_prediction[:, :642])
                voltage_target.append(batch["raw_state_t_plus_1"][:, :642])
                event_probabilities.append(
                    torch.sigmoid(output["event_logits"]).cpu().numpy()
                )
                event_targets.append(batch["event_presence"].cpu().numpy())
                regimes.extend(batch["regimes"])
                normalizer = self.normalizers[spec.gate_transform]
                bounded = np.flatnonzero(normalizer.transform_codes == normalizer.LOGIT)
                positive = np.flatnonzero(normalizer.transform_codes == normalizer.LOG1P)
                if len(bounded):
                    values = raw_prediction[:, bounded]
                    outside += int(((values < 0.0) | (values > 1.0)).sum())
                    outside_denominator += values.size
                if len(positive):
                    values = raw_prediction[:, positive]
                    outside += int((values < 0.0).sum())
                    outside_denominator += values.size
        result: Dict[str, Any] = {
            key: value / max(1, count) for key, value in totals.items()
        }
        if not training:
            pred = np.concatenate(voltage_prediction)
            true = np.concatenate(voltage_target)
            probabilities = np.concatenate(event_probabilities)
            targets = np.concatenate(event_targets)
            error = pred - true
            event_rows = np.asarray(
                [
                    regime in {"somatic_spike", "nmda_plateau", "calcium_bac"}
                    for regime in regimes
                ]
            )
            sub_rows = np.asarray([regime == "rest_subthreshold" for regime in regimes])
            negative_rows = np.asarray(
                [regime == "event_boundary_negative" for regime in regimes]
            )

            def rmse(mask: np.ndarray) -> float:
                selected = error[mask] if mask.any() else error
                return float(np.sqrt(np.mean(selected ** 2)))

            result["operational"] = {
                "voltage_rmse_mv": rmse(np.ones(len(error), dtype=bool)),
                "event_voltage_rmse_mv": rmse(event_rows),
                "boundary_voltage_rmse_mv": rmse(negative_rows),
                "subthreshold_voltage_mae_mv": float(
                    np.mean(np.abs(error[sub_rows] if sub_rows.any() else error))
                ),
                "rest_drift_mv": float(
                    np.mean(error[sub_rows] if sub_rows.any() else error)
                ),
                "learnable_event_macro_f1": self._learnable_macro_f1(
                    probabilities, targets
                ),
                "outside_domain_fraction": float(
                    outside / max(1, outside_denominator)
                ),
            }
            op = result["operational"]
            result["selection_score"] = float(
                op["voltage_rmse_mv"]
                + op["event_voltage_rmse_mv"]
                + 0.5 * op["boundary_voltage_rmse_mv"]
                + abs(op["rest_drift_mv"])
                + 2.0 * (1.0 - op["learnable_event_macro_f1"])
                + 100.0 * op["outside_domain_fraction"]
            )
        return result

    def train_model(
        self,
        spec: ReconditionedRunSpec,
        seed: int,
    ) -> Tuple[Any, Dict[str, Path], List[Dict[str, Any]]]:
        require_torch()
        import torch

        _seed(seed, self.config.deterministic_algorithms)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = self._build_model(spec, device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        model_id = spec.identifier(seed)
        contract = self._checkpoint_contract(spec, seed)
        run_dir = self._checkpoint_path(model_id, contract)
        paths = {
            name: run_dir / f"{name}.pt"
            for name in (
                "last",
                "best_selection",
                "best_one_step",
                "best_event",
                "best_rollout",
            )
        }
        run_dir.mkdir(parents=True, exist_ok=True)
        _write_json(run_dir / "fingerprint.json", contract)
        self.checkpoint_contracts[model_id] = contract
        start = 0
        patience = 0
        history: List[Dict[str, Any]] = []
        best_selection = math.inf
        best_one_step = math.inf
        best_event = -math.inf
        best_rollout = math.inf
        if paths["last"].is_file():
            saved = self._load_checkpoint(
                paths["last"], model, contract, optimizer
            )
            start = int(saved["epoch"]) + 1
            history = list(saved.get("history", []))
            patience = int(saved.get("selection", {}).get("patience", 0))
            if history:
                best_selection = min(row["validation"]["selection_score"] for row in history)
                best_one_step = min(
                    row["validation"]["operational"]["voltage_rmse_mv"]
                    for row in history
                )
                best_event = max(
                    row["validation"]["operational"]["learnable_event_macro_f1"]
                    for row in history
                )
                previous_rollouts = [
                    row["validation"]["operational"].get(
                        "rollout_8ms_selection_score", math.inf
                    )
                    for row in history
                ]
                best_rollout = min(previous_rollouts, default=math.inf)
        event_weight = torch.as_tensor(
            self._event_weights(), dtype=torch.float32, device=device
        )
        progress = Progress(model_id, self.config.maximum_epochs)
        epoch_range: Iterable[int] = range(start, self.config.maximum_epochs)
        if patience >= self.config.early_stopping_patience:
            print(
                f"[HayFlow 02b][{model_id}] early stopping already reached; "
                "reusing fingerprint-compatible checkpoints",
                flush=True,
            )
            epoch_range = ()
        for epoch in epoch_range:
            training_indices = self._balanced_indices(seed + epoch)
            train_metrics = self._epoch(
                model,
                spec,
                training_indices,
                device,
                event_weight,
                optimizer=optimizer,
                epoch=epoch,
                model_id=model_id,
            )
            validation_metrics = self._epoch(
                model,
                spec,
                self.store.split_indices["validation"],
                device,
                event_weight,
                optimizer=None,
                epoch=epoch,
                model_id=model_id,
            )
            rollout_validation = self._quick_rollout(
                model, spec, "validation", 8
            )
            rollout_selection = float(
                rollout_validation["voltage_rmse_mv"]
                + abs(rollout_validation["voltage_rest_drift_mv"])
            )
            validation_metrics["operational"].update(
                {
                    "rollout_8ms_voltage_rmse_mv": float(
                        rollout_validation["voltage_rmse_mv"]
                    ),
                    "rollout_8ms_rest_drift_mv": float(
                        rollout_validation["voltage_rest_drift_mv"]
                    ),
                    "rollout_8ms_selection_score": rollout_selection,
                }
            )
            validation_metrics["selection_score"] += 0.25 * rollout_selection
            row = {
                "epoch": int(epoch),
                "train": train_metrics,
                "validation": validation_metrics,
            }
            history.append(row)
            selection = validation_metrics["selection_score"]
            improved = selection < best_selection - 1e-6
            if improved:
                best_selection = selection
                patience = 0
                self._save_checkpoint(
                    paths["best_selection"], model, optimizer, epoch, contract, history,
                    {"selection_score": selection, "patience": patience},
                )
            else:
                patience += 1
            voltage_score = validation_metrics["operational"]["voltage_rmse_mv"]
            if voltage_score < best_one_step:
                best_one_step = voltage_score
                self._save_checkpoint(
                    paths["best_one_step"], model, optimizer, epoch, contract, history,
                    {"voltage_rmse_mv": voltage_score, "patience": patience},
                )
            event_score = validation_metrics["operational"]["learnable_event_macro_f1"]
            if event_score > best_event:
                best_event = event_score
                self._save_checkpoint(
                    paths["best_event"], model, optimizer, epoch, contract, history,
                    {"learnable_event_macro_f1": event_score, "patience": patience},
                )
            if rollout_selection < best_rollout:
                best_rollout = rollout_selection
                self._save_checkpoint(
                    paths["best_rollout"], model, optimizer, epoch, contract, history,
                    {
                        "rollout_8ms_selection_score": rollout_selection,
                        "patience": patience,
                    },
                )
            self._save_checkpoint(
                paths["last"], model, optimizer, epoch, contract, history,
                {"selection_score": selection, "patience": patience},
                include_optimizer=True,
            )
            _write_json(
                self.gradient_partial_path,
                {
                    "schema_version": "02b-gradient-partial-v1",
                    "code_commit": self.code_commit,
                    "rows": self.gradient_rows,
                },
            )
            progress.update(
                epoch + 1,
                (
                    f"score={selection:.3g} V={voltage_score:.3g} "
                    f"eventV={validation_metrics['operational']['event_voltage_rmse_mv']:.3g} "
                    f"F1={event_score:.3f} drift="
                    f"{validation_metrics['operational']['rest_drift_mv']:.3g}"
                ),
            )
            if patience >= self.config.early_stopping_patience:
                break
        if not paths["best_rollout"].is_file():
            self._select_rollout_checkpoint(model, spec, contract, paths)
        self._load_checkpoint(paths["best_selection"], model, contract)
        self.histories[model_id] = history
        self.registry[model_id] = {
            "spec": asdict(spec),
            "seed": seed,
            "parameter_count": parameter_count(model),
            "train_transition_count": len(self.store.split_indices["train"]),
            "parameter_to_transition_ratio": parameter_count(model)
            / len(self.store.split_indices["train"]),
            "fingerprint": contract["fingerprint"],
            "checkpoints": {
                key: str(path.relative_to(self.output_dir))
                for key, path in paths.items()
                if path.is_file()
            },
            "epochs_completed": len(history),
        }
        return model, paths, history

    def _select_rollout_checkpoint(
        self,
        model: Any,
        spec: ReconditionedRunSpec,
        contract: Mapping[str, Any],
        paths: Mapping[str, Path],
    ) -> None:
        import torch

        candidates = [
            path for name, path in paths.items()
            if name in {"best_selection", "best_one_step", "best_event"} and path.is_file()
        ]
        best_score = math.inf
        best_payload = None
        for path in candidates:
            saved = self._load_checkpoint(path, model, contract)
            row = self._quick_rollout(model, spec, "validation", 8)
            score = row["voltage_rmse_mv"] + abs(row["voltage_rest_drift_mv"])
            if score < best_score:
                best_score = score
                best_payload = saved
        if best_payload is not None:
            torch.save(best_payload, paths["best_rollout"])

    def _predict(
        self,
        model: Any,
        spec: ReconditionedRunSpec,
        indices: Sequence[int],
        *,
        raw_state_override: Optional[np.ndarray] = None,
    ) -> Dict[str, np.ndarray]:
        import torch

        device = next(model.parameters()).device
        indices = np.asarray(indices, dtype=np.int64)
        pieces: Dict[str, List[np.ndarray]] = {}
        for start in range(0, len(indices), self.config.evaluation_batch_size):
            stop = min(start + self.config.evaluation_batch_size, len(indices))
            override = (
                raw_state_override[start:stop]
                if raw_state_override is not None
                else None
            )
            batch = self._load_batch(
                indices[start:stop], spec, device, raw_state_override=override
            )
            model.eval()
            with torch.no_grad():
                output = model(batch)
            activity = torch.sigmoid(output["activity_logits"]).cpu().numpy()
            raw = self.normalizers[spec.gate_transform].reconstruct(
                batch["raw_state_t"],
                output["delta"].cpu().numpy(),
                activity_probability=activity,
                synapse_mode=spec.synapse_mode,
            )
            current = {
                "raw_state": raw,
                "event_probability": torch.sigmoid(output["event_logits"]).cpu().numpy(),
                "event_timing": output["event_timing"].cpu().numpy(),
                "event_region": output["event_region_logits"].argmax(-1).cpu().numpy(),
            }
            for key, value in current.items():
                pieces.setdefault(key, []).append(value)
        return {key: np.concatenate(value) for key, value in pieces.items()}

    def _calibrate_thresholds(self, model: Any, spec: ReconditionedRunSpec) -> np.ndarray:
        indices = self.store.split_indices["validation"]
        probability = self._predict(model, spec, indices)["event_probability"]
        target = self.store.event_targets(indices)["event_presence"]
        thresholds = np.full(len(EVENT_KINDS), 0.5, dtype=np.float64)
        for column, learnable in enumerate(self.learnable_event_mask):
            if not learnable:
                thresholds[column] = np.nan
                continue
            positive_count = int(np.sum(target[:, column] > 0.5))
            if positive_count == 0 or positive_count == len(target):
                # Validation cannot identify a classification threshold without
                # both classes. Keep the declared neutral threshold.
                continue
            best_f1 = -1.0
            for candidate in np.linspace(0.05, 0.95, 19):
                pred = probability[:, column] >= candidate
                true = target[:, column] > 0.5
                tp = np.sum(pred & true)
                fp = np.sum(pred & ~true)
                fn = np.sum(~pred & true)
                precision = tp / max(1, tp + fp)
                recall = tp / max(1, tp + fn)
                f1 = 2 * precision * recall / max(1e-12, precision + recall)
                if f1 > best_f1:
                    best_f1 = f1
                    thresholds[column] = candidate
        return thresholds

    def _calibrate_array_thresholds(
        self,
        probability: np.ndarray,
        target: np.ndarray,
    ) -> np.ndarray:
        thresholds = np.full(len(EVENT_KINDS), 0.5, dtype=np.float64)
        for column, learnable in enumerate(self.learnable_event_mask):
            if not learnable:
                thresholds[column] = 1.1
                continue
            positive_count = int(np.sum(target[:, column] > 0.5))
            if positive_count == 0 or positive_count == len(target):
                continue
            best_f1 = -1.0
            for candidate in np.linspace(0.05, 0.95, 19):
                predicted = probability[:, column] >= candidate
                true = target[:, column] > 0.5
                tp = np.sum(predicted & true)
                fp = np.sum(predicted & ~true)
                fn = np.sum(~predicted & true)
                precision = tp / max(1, tp + fp)
                recall = tp / max(1, tp + fn)
                f1 = 2 * precision * recall / max(1e-12, precision + recall)
                if f1 > best_f1:
                    best_f1 = f1
                    thresholds[column] = candidate
        return thresholds

    def evaluate_model(
        self,
        model_id: str,
        model: Any,
        spec: ReconditionedRunSpec,
    ) -> Dict[str, Any]:
        threshold = self._calibrate_thresholds(model, spec)
        safe_threshold = np.where(np.isfinite(threshold), threshold, 1.1)
        for split, indices in self.store.split_indices.items():
            prediction = self._predict(model, spec, indices)
            true = self.store.read_state(indices, "t_plus_1")
            self.one_step_rows.extend(
                state_metric_rows(
                    prediction["raw_state"],
                    true,
                    layout=self.layout,
                    normalizer=self.normalizers[spec.gate_transform],
                    model_name=model_id,
                    split=split,
                    regimes=[_regime(self.store, int(index)) for index in indices],
                )
            )
            target = self.store.event_targets(indices)
            rows = binary_event_metric_rows(
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
                threshold=safe_threshold,
            )
            for row, learnable in zip(rows, self.learnable_event_mask):
                row["learnable_from_train"] = bool(learnable)
                if not learnable:
                    row["interpretation"] = "not_learnable_from_current_split"
            self.event_rows.extend(rows)
        self._evaluate_rollouts(model_id, model, spec, safe_threshold)
        self._evaluate_branching(model_id, model, spec)
        return {
            kind: (float(threshold[index]) if np.isfinite(threshold[index]) else None)
            for index, kind in enumerate(EVENT_KINDS)
        }

    def _quick_rollout(
        self,
        model: Any,
        spec: ReconditionedRunSpec,
        split: str,
        horizon: int,
    ) -> Dict[str, Any]:
        windows = self.store.rollout_windows(split, horizon)[
            : self.config.selection_rollout_windows
        ]
        predicted = []
        target = []
        for window in windows:
            state = self.store.read_state([int(window[0])], "t")
            for index in window:
                state = self._predict(
                    model, spec, [int(index)], raw_state_override=state
                )["raw_state"]
            predicted.append(state[0])
            target.append(self.store.read_state([int(window[-1])], "t_plus_1")[0])
        if not predicted:
            return {
                "voltage_rmse_mv": math.inf,
                "voltage_rest_drift_mv": math.inf,
            }
        return rollout_metric_row(
            np.asarray(predicted),
            np.asarray(target),
            model_name="selection",
            split=split,
            horizon_ms=horizon,
            voltage_width=642,
            layout=self.layout,
        )

    def _evaluate_rollouts(
        self,
        model_id: str,
        model: Any,
        spec: ReconditionedRunSpec,
        thresholds: np.ndarray,
    ) -> None:
        for split in (
            "validation",
            "deterministic_test",
            "event_boundary_test",
            "branching_test",
        ):
            if split not in self.store.split_indices:
                continue
            for horizon in self.config.rollout_horizons_ms:
                windows = self.store.rollout_windows(split, horizon)[
                    : self.config.maximum_rollout_windows_per_split
                ]
                predicted = []
                target = []
                regimes = []
                peak_attenuation = []
                lost_events = []
                added_events = []
                for window in windows:
                    state = self.store.read_state([int(window[0])], "t")
                    predicted_trace = []
                    teacher_trace = []
                    predicted_event_steps = []
                    teacher_event_steps = []
                    for index in window:
                        step_prediction = self._predict(
                            model, spec, [int(index)], raw_state_override=state
                        )
                        state = step_prediction["raw_state"]
                        predicted_trace.append(state[0, :642])
                        teacher_trace.append(
                            self.store.read_state([int(index)], "t_plus_1")[0, :642]
                        )
                        predicted_event_steps.append(
                            step_prediction["event_probability"][0] >= thresholds
                        )
                        teacher_event_steps.append(
                            self.store.event_targets([int(index)])["event_presence"][0]
                            > 0.5
                        )
                    predicted.append(state[0])
                    target.append(self.store.read_state([int(window[-1])], "t_plus_1")[0])
                    regimes.append(_regime(self.store, int(window[0])))
                    peak_attenuation.append(
                        float(np.max(predicted_trace) - np.max(teacher_trace))
                    )
                    predicted_event_steps = np.asarray(predicted_event_steps)
                    teacher_event_steps = np.asarray(teacher_event_steps)
                    lost_events.append(
                        np.sum(teacher_event_steps & ~predicted_event_steps, axis=0)
                    )
                    added_events.append(
                        np.sum(~teacher_event_steps & predicted_event_steps, axis=0)
                    )
                if not predicted:
                    continue
                pred = np.asarray(predicted)
                true = np.asarray(target)
                for regime in ["all", *sorted(set(regimes))]:
                    selected = (
                        np.ones(len(regimes), dtype=bool)
                        if regime == "all"
                        else np.asarray(regimes) == regime
                    )
                    row = rollout_metric_row(
                        pred[selected],
                        true[selected],
                        model_name=model_id,
                        split=split,
                        horizon_ms=horizon,
                        voltage_width=642,
                        layout=self.layout,
                    )
                    row["regime"] = regime
                    row["mean_peak_attenuation_mv"] = float(
                        np.mean(np.asarray(peak_attenuation)[selected])
                    )
                    for event_column, event_kind in enumerate(EVENT_KINDS):
                        row[f"lost_{event_kind}_count"] = int(
                            np.sum(np.asarray(lost_events)[selected, event_column])
                        )
                        row[f"added_{event_kind}_count"] = int(
                            np.sum(np.asarray(added_events)[selected, event_column])
                        )
                    if regime in {"somatic_spike", "nmda_plateau", "calcium_bac"}:
                        row["recovery_endpoint_voltage_rmse_mv"] = row[
                            "voltage_rmse_mv"
                        ]
                    voltage_error = pred[selected, :642] - true[selected, :642]
                    for region_id, region_name in enumerate(self.layout.region_names):
                        region_mask = self.layout.segment_region_ids == region_id
                        if region_mask.any():
                            safe_name = "".join(
                                character if character.isalnum() else "_"
                                for character in region_name
                            )
                            row[f"drift_region_{safe_name}_mv"] = float(
                                np.mean(voltage_error[:, region_mask])
                            )
                    self.rollout_rows.append(row)

    def _evaluate_branching(
        self,
        model_id: str,
        model: Any,
        spec: ReconditionedRunSpec,
    ) -> None:
        trajectories = []
        for trajectory_id, indices in self.store.trajectory_indices.items():
            if not len(indices) or self.store.metadata["split"][indices[0]] != "branching_test":
                continue
            horizon = min(16, len(indices))
            window = indices[:horizon]
            state = self.store.read_state([int(window[0])], "t")
            initial = state.copy()
            for index in window:
                state = self._predict(
                    model, spec, [int(index)], raw_state_override=state
                )["raw_state"]
            trajectories.append(
                {
                    "id": trajectory_id,
                    "initial": initial[0],
                    "predicted": state[0],
                    "teacher": self.store.read_state([int(window[-1])], "t_plus_1")[0],
                    "horizon": horizon,
                }
            )
        for left_index, left in enumerate(trajectories):
            for right in trajectories[left_index + 1 :]:
                initial_error = float(np.max(np.abs(left["initial"] - right["initial"])))
                if initial_error > 1e-5 or left["horizon"] != right["horizon"]:
                    continue
                teacher_distance = float(
                    np.max(np.abs(left["teacher"][:642] - right["teacher"][:642]))
                )
                predicted_distance = float(
                    np.max(
                        np.abs(left["predicted"][:642] - right["predicted"][:642])
                    )
                )
                self.branching_rows.append(
                    {
                        "model": model_id,
                        "left_trajectory": left["id"],
                        "right_trajectory": right["id"],
                        "horizon_ms": left["horizon"],
                        "initial_max_state_error": initial_error,
                        "teacher_future_distance_mv": teacher_distance,
                        "predicted_future_distance_mv": predicted_distance,
                        "divergence_retention": predicted_distance
                        / max(teacher_distance, 1e-12),
                        "collapsed": bool(
                            teacher_distance > 1e-3
                            and predicted_distance < 0.05 * teacher_distance
                        ),
                    }
                )

    def _classical_batch(
        self,
        indices: Sequence[int],
        normalizer: ReconditionedStateNormalizer,
    ) -> Dict[str, Any]:
        raw_t = self.store.read_state(indices, "t")
        raw_t1 = self.store.read_state(indices, "t_plus_1")
        delta, _ = normalizer.delta_and_activity(raw_t, raw_t1)
        batch = {
            "raw_state_t": raw_t,
            "raw_state_t_plus_1": raw_t1,
            "state_t": normalizer.normalize_state(raw_t).astype(np.float32),
            "delta_target": delta,
        }
        batch.update(self.store.encode_inputs(indices))
        batch.update(self.store.event_targets(indices))
        return batch

    def run_classical(self) -> None:
        normalizer = self.normalizers["logit"]
        for split, indices in self.store.split_indices.items():
            true = self.store.read_state(indices, "t_plus_1")
            pred = self.store.read_state(indices, "t")
            self.one_step_rows.extend(
                state_metric_rows(
                    pred,
                    true,
                    layout=self.layout,
                    normalizer=normalizer,
                    model_name=PersistenceBaseline.name,
                    split=split,
                    regimes=[_regime(self.store, int(index)) for index in indices],
                )
            )
            event_target = self.store.event_targets(indices)
            rows = binary_event_metric_rows(
                np.zeros_like(event_target["event_presence"]),
                event_target["event_presence"],
                timing_prediction=None,
                timing_target=None,
                timing_mask=None,
                model_name=PersistenceBaseline.name,
                split=split,
                threshold=0.5,
            )
            for row, learnable in zip(rows, self.learnable_event_mask):
                row["learnable_from_train"] = bool(learnable)
            self.event_rows.extend(rows)
        self._evaluate_classical_rollouts(
            PersistenceBaseline.name, normalizer, None
        )
        self._evaluate_classical_branching(
            PersistenceBaseline.name, normalizer, None
        )
        train = self.store.split_indices["train"]
        train_batch = self._classical_batch(train, normalizer)
        target = np.concatenate(
            [train_batch["delta_target"], train_batch["event_presence"]], axis=1
        )
        for encoding in ("U1", "U2"):
            model_name = f"B1_reconditioned-full-{encoding}"
            features = ridge_design_matrix(
                train_batch,
                voltage_width=642,
                state_mode="full",
                input_encoding=encoding,
            )
            model = DualRidgeBaseline(self.config.ridge_alpha).fit(features, target)
            model.save(self.checkpoint_dir / f"{model_name}.npz")
            validation = self._classical_batch(
                self.store.split_indices["validation"], normalizer
            )
            validation_output = model.predict(
                ridge_design_matrix(
                    validation,
                    voltage_width=642,
                    state_mode="full",
                    input_encoding=encoding,
                )
            )
            thresholds = self._calibrate_array_thresholds(
                np.clip(validation_output[:, self.layout.state_width :], 0.0, 1.0),
                validation["event_presence"],
            )
            for split, indices in self.store.split_indices.items():
                batch = self._classical_batch(indices, normalizer)
                output = model.predict(
                    ridge_design_matrix(
                        batch,
                        voltage_width=642,
                        state_mode="full",
                        input_encoding=encoding,
                    )
                )
                raw = normalizer.reconstruct(
                    batch["raw_state_t"],
                    output[:, : self.layout.state_width],
                    apply_hurdle=False,
                )
                self.one_step_rows.extend(
                    state_metric_rows(
                        raw,
                        batch["raw_state_t_plus_1"],
                        layout=self.layout,
                        normalizer=normalizer,
                        model_name=model_name,
                        split=split,
                        regimes=[_regime(self.store, int(index)) for index in indices],
                    )
                )
                rows = binary_event_metric_rows(
                    np.clip(output[:, self.layout.state_width :], 0.0, 1.0),
                    batch["event_presence"],
                    timing_prediction=None,
                    timing_target=None,
                    timing_mask=None,
                    model_name=model_name,
                    split=split,
                    threshold=thresholds,
                )
                for row, learnable in zip(rows, self.learnable_event_mask):
                    row["learnable_from_train"] = bool(learnable)
                self.event_rows.extend(rows)
            self._evaluate_classical_rollouts(
                model_name, normalizer, (model, encoding)
            )
            self._evaluate_classical_branching(
                model_name, normalizer, (model, encoding)
            )

    def _evaluate_classical_rollouts(
        self,
        model_name: str,
        normalizer: ReconditionedStateNormalizer,
        model_spec: Optional[Tuple[DualRidgeBaseline, str]],
    ) -> None:
        for split in (
            "validation",
            "deterministic_test",
            "event_boundary_test",
            "branching_test",
        ):
            if split not in self.store.split_indices:
                continue
            for horizon in self.config.rollout_horizons_ms:
                windows = self.store.rollout_windows(split, horizon)[
                    : self.config.maximum_rollout_windows_per_split
                ]
                predicted = []
                target = []
                regimes = []
                for window in windows:
                    state = self.store.read_state([int(window[0])], "t")
                    for index in window:
                        if model_spec is not None:
                            model, encoding = model_spec
                            batch = self._classical_batch([int(index)], normalizer)
                            batch["raw_state_t"] = state
                            batch["state_t"] = normalizer.normalize_state(state).astype(
                                np.float32
                            )
                            delta = model.predict(
                                ridge_design_matrix(
                                    batch,
                                    voltage_width=642,
                                    state_mode="full",
                                    input_encoding=encoding,
                                )
                            )[:, : self.layout.state_width]
                            state = normalizer.reconstruct(
                                state, delta, apply_hurdle=False
                            )
                    predicted.append(state[0])
                    target.append(
                        self.store.read_state([int(window[-1])], "t_plus_1")[0]
                    )
                    regimes.append(_regime(self.store, int(window[0])))
                if not predicted:
                    continue
                pred = np.asarray(predicted)
                true = np.asarray(target)
                for regime in ["all", *sorted(set(regimes))]:
                    selected = (
                        np.ones(len(regimes), dtype=bool)
                        if regime == "all"
                        else np.asarray(regimes) == regime
                    )
                    row = rollout_metric_row(
                        pred[selected],
                        true[selected],
                        model_name=model_name,
                        split=split,
                        horizon_ms=horizon,
                        voltage_width=642,
                        layout=self.layout,
                    )
                    row["regime"] = regime
                    self.rollout_rows.append(row)

    def _evaluate_classical_branching(
        self,
        model_name: str,
        normalizer: ReconditionedStateNormalizer,
        model_spec: Optional[Tuple[DualRidgeBaseline, str]],
    ) -> None:
        trajectories = []
        for trajectory_id, indices in self.store.trajectory_indices.items():
            if (
                not len(indices)
                or self.store.metadata["split"][indices[0]] != "branching_test"
            ):
                continue
            horizon = min(16, len(indices))
            window = indices[:horizon]
            state = self.store.read_state([int(window[0])], "t")
            initial = state.copy()
            for index in window:
                if model_spec is not None:
                    model, encoding = model_spec
                    batch = self._classical_batch([int(index)], normalizer)
                    batch["raw_state_t"] = state
                    batch["state_t"] = normalizer.normalize_state(state).astype(
                        np.float32
                    )
                    delta = model.predict(
                        ridge_design_matrix(
                            batch,
                            voltage_width=642,
                            state_mode="full",
                            input_encoding=encoding,
                        )
                    )[:, : self.layout.state_width]
                    state = normalizer.reconstruct(
                        state, delta, apply_hurdle=False
                    )
            trajectories.append(
                {
                    "id": trajectory_id,
                    "initial": initial[0],
                    "predicted": state[0],
                    "teacher": self.store.read_state(
                        [int(window[-1])], "t_plus_1"
                    )[0],
                    "horizon": horizon,
                }
            )
        for left_index, left in enumerate(trajectories):
            for right in trajectories[left_index + 1 :]:
                initial_error = float(
                    np.max(np.abs(left["initial"] - right["initial"]))
                )
                if initial_error > 1e-5 or left["horizon"] != right["horizon"]:
                    continue
                teacher_distance = float(
                    np.max(np.abs(left["teacher"][:642] - right["teacher"][:642]))
                )
                predicted_distance = float(
                    np.max(
                        np.abs(
                            left["predicted"][:642]
                            - right["predicted"][:642]
                        )
                    )
                )
                self.branching_rows.append(
                    {
                        "model": model_name,
                        "left_trajectory": left["id"],
                        "right_trajectory": right["id"],
                        "horizon_ms": left["horizon"],
                        "initial_max_state_error": initial_error,
                        "teacher_future_distance_mv": teacher_distance,
                        "predicted_future_distance_mv": predicted_distance,
                        "divergence_retention": predicted_distance
                        / max(teacher_distance, 1e-12),
                        "collapsed": bool(
                            teacher_distance > 1e-3
                            and predicted_distance < 0.05 * teacher_distance
                        ),
                    }
                )

    def run(self) -> Dict[str, Any]:
        if not self.normalizers:
            self.prepare()
        self.run_classical()
        total = sum(len(seeds) for _, seeds in self._run_specs())
        progress = Progress("neural models", total)
        completed = 0
        for spec, seeds in self._run_specs():
            for seed in seeds:
                model_id = spec.identifier(seed)
                model, _, history = self.train_model(spec, seed)
                thresholds = self.evaluate_model(model_id, model, spec)
                self.registry[model_id]["validation_calibrated_event_thresholds"] = thresholds
                completed += 1
                progress.update(completed, model_id)
                del model
                _clear_cuda()
        return self.finalize()

    def _metric_mean(
        self,
        fragment: str,
        rows: Sequence[Mapping[str, Any]],
        key: str,
        **filters: Any,
    ) -> float:
        values = [
            float(row[key])
            for row in rows
            if fragment in str(row.get("model", ""))
            and all(row.get(name) == value for name, value in filters.items())
            and key in row
            and math.isfinite(float(row[key]))
        ]
        return float(np.mean(values)) if values else math.inf

    def _seed_summary(self) -> List[Dict[str, Any]]:
        rows = []
        test_splits = {"deterministic_test", "event_boundary_test", "branching_test"}
        evaluation_horizon = max(self.config.rollout_horizons_ms)
        for model_id, registry in self.registry.items():
            voltage = [
                float(row["rmse"])
                for row in self.one_step_rows
                if row["model"] == model_id
                and row["split"] in test_splits
                and row["scope"] == "category"
                and row["name"] == "voltage"
            ]
            rollout = [
                float(row["voltage_rmse_mv"])
                for row in self.rollout_rows
                if row["model"] == model_id
                and row["split"] in test_splits
                and row["regime"] == "all"
                and row["horizon_ms"] == evaluation_horizon
            ]
            event = [
                float(row["f1"])
                for row in self.event_rows
                if row["model"] == model_id
                and row["split"] in test_splits
                and row.get("learnable_from_train")
                and int(row.get("support", 0)) > 0
            ]
            branch = [
                float(row["divergence_retention"])
                for row in self.branching_rows
                if row["model"] == model_id
                and float(row["teacher_future_distance_mv"]) > 1e-3
            ]
            rows.append(
                {
                    "model": model_id,
                    "seed": registry["seed"],
                    **registry["spec"],
                    "rollout_evaluation_horizon_ms": evaluation_horizon,
                    "test_voltage_rmse_mv": float(np.mean(voltage)),
                    "test_rollout_16ms_rmse_mv": float(np.mean(rollout)),
                    "test_learnable_event_macro_f1": float(np.mean(event)) if event else 0.0,
                    "branching_divergence_retention_median": float(np.median(branch)) if branch else 0.0,
                }
            )
        return rows

    def _aggregate_seeds(self, seed_rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
        grouped: Dict[Tuple[str, str, str, str], List[Mapping[str, Any]]] = {}
        for row in seed_rows:
            key = (
                str(row["input_encoding"]),
                str(row["synapse_mode"]),
                str(row["privileged_mode"]),
                str(row["gate_transform"]),
            )
            grouped.setdefault(key, []).append(row)
        result = []
        metrics = (
            "test_voltage_rmse_mv",
            "test_rollout_16ms_rmse_mv",
            "test_learnable_event_macro_f1",
            "branching_divergence_retention_median",
        )
        for key, values in sorted(grouped.items()):
            row: Dict[str, Any] = {
                "input_encoding": key[0],
                "synapse_mode": key[1],
                "privileged_mode": key[2],
                "gate_transform": key[3],
                "seed_count": len(values),
            }
            for metric in metrics:
                numbers = np.asarray([value[metric] for value in values], dtype=float)
                row[f"{metric}_mean"] = float(np.mean(numbers))
                row[f"{metric}_std"] = float(np.std(numbers))
            result.append(row)
        return result

    def _decision(self, aggregate: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
        def aggregate_row(
            input_encoding: str,
            synapse_mode: str,
            privileged_mode: str,
            gate_transform: str = "logit",
        ) -> Mapping[str, Any]:
            return next(
                row
                for row in aggregate
                if row["input_encoding"] == input_encoding
                and row["synapse_mode"] == synapse_mode
                and row["privileged_mode"] == privileged_mode
                and row["gate_transform"] == gate_transform
            )

        main = aggregate_row("U2", "hurdle", "P0")
        u1 = aggregate_row("U1", "hurdle", "P0")
        p1a = aggregate_row("U2", "hurdle", "P1a")
        p1b = aggregate_row("U2", "hurdle", "P1b")
        test_splits = {"deterministic_test", "event_boundary_test", "branching_test"}
        evaluation_horizon = max(self.config.rollout_horizons_ms)
        b1_values = [
            float(row["voltage_rmse_mv"])
            for row in self.rollout_rows
            if row["model"] == "B1_reconditioned-full-U2"
            and row["split"] in test_splits
            and row["horizon_ms"] == evaluation_horizon
            and row["regime"] == "all"
        ]
        b1_rollout = float(np.mean(b1_values)) if b1_values else math.inf
        nonlinear_regimes = {"somatic_spike", "calcium_bac", "nmda_plateau"}
        b1_nonlinear_values = [
            float(row["voltage_rmse_mv"])
            for row in self.rollout_rows
            if row["model"] == "B1_reconditioned-full-U2"
            and row["split"] in test_splits
            and row["horizon_ms"] == evaluation_horizon
            and row["regime"] in nonlinear_regimes
        ]
        b1_nonlinear_rollout = (
            float(np.mean(b1_nonlinear_values))
            if b1_nonlinear_values
            else math.inf
        )
        b1_one_step_values = [
            float(row["rmse"])
            for row in self.one_step_rows
            if row["model"] == "B1_reconditioned-full-U2"
            and row["split"] in test_splits
            and row["scope"] == "category"
            and row["name"] == "voltage"
        ]
        b1_one_step = (
            float(np.mean(b1_one_step_values))
            if b1_one_step_values
            else math.inf
        )
        original = self.original_report["comparison_inputs"]
        original_learnable_event_f1 = self._original_learnable_event_f1()
        seed_stable = main["test_rollout_16ms_rmse_mv_std"] < 0.15 * max(
            main["test_rollout_16ms_rmse_mv_mean"], 1e-12
        )
        rollout_advantage = main["test_rollout_16ms_rmse_mv_mean"] < b1_rollout
        clear_rollout_advantage = (
            main["test_rollout_16ms_rmse_mv_mean"] <= 0.95 * b1_rollout
        )
        event_improved = (
            main["test_learnable_event_macro_f1_mean"]
            > original_learnable_event_f1
        )
        branching_retained = main[
            "branching_divergence_retention_median_mean"
        ] >= 0.10
        main_ids = {
            model_id
            for model_id, registry in self.registry.items()
            if registry["spec"]
            == {
                "input_encoding": "U2",
                "synapse_mode": "hurdle",
                "privileged_mode": "P0",
                "gate_transform": "logit",
            }
        }
        main_drift_values = [
            float(row["voltage_rest_drift_mv"])
            for row in self.rollout_rows
            if row["model"] in main_ids
            and row["split"] in test_splits
            and row["horizon_ms"] == evaluation_horizon
            and row["regime"] == "all"
        ]
        main_drift = (
            float(np.mean(main_drift_values)) if main_drift_values else math.inf
        )
        original_rollout = self.original_report.get("comparison_summary", {}).get(
            "reference_b3_rollout_stability", []
        )
        original_drift = next(
            (
                float(row["rest_drift_mv"])
                for row in original_rollout
                if int(row["horizon_ms"]) == evaluation_horizon
            ),
            math.inf,
        )
        main_nonlinear_values = [
            float(row["voltage_rmse_mv"])
            for row in self.rollout_rows
            if row["model"] in main_ids
            and row["split"] in test_splits
            and row["horizon_ms"] == evaluation_horizon
            and row["regime"] in nonlinear_regimes
        ]
        main_nonlinear_rollout = (
            float(np.mean(main_nonlinear_values))
            if main_nonlinear_values
            else math.inf
        )
        nonlinear_advantage = main_nonlinear_rollout < b1_nonlinear_rollout
        drift_reduced = abs(main_drift) < abs(original_drift)
        required_checkpoint_fields = {
            "dataset_manifest_sha256",
            "dataset_schema",
            "split_indices_sha256",
            "state_normalizer_fingerprint",
            "auxiliary_normalizer_fingerprint",
            "loss_config",
            "run_spec",
            "model_config",
            "code_commit",
            "seed",
            "fingerprint",
        }
        checkpoint_contracts_complete = bool(self.checkpoint_contracts) and all(
            required_checkpoint_fields <= set(contract)
            and contract["fingerprint"]
            == _stable_hash(
                {
                    key: value
                    for key, value in contract.items()
                    if key != "fingerprint"
                }
            )
            for contract in self.checkpoint_contracts.values()
        )
        conditioning_well_formed = bool(
            self.conditioning_diagnostics.get("all_scales_respect_floor")
        ) and float(
            self.conditioning_diagnostics.get(
                "maximum_active_normalized_absolute_value", math.inf
            )
        ) <= 100.0
        gradient_model_ids = {
            model_id
            for model_id, registry in self.registry.items()
            if registry["spec"]["gate_transform"] == "logit"
        }
        per_epoch: Dict[Tuple[str, int], List[float]] = {}
        for row in self.gradient_rows:
            if row.get("model_id") not in gradient_model_ids:
                continue
            value = float(
                row.get("weighted_shared_representation_gradient_norm", math.nan)
            )
            if math.isfinite(value) and value > 0.0:
                per_epoch.setdefault(
                    (str(row["model_id"]), int(row["epoch"])), []
                ).append(value)
        dominance = [max(values) / sum(values) for values in per_epoch.values() if values]
        median_gradient_dominance = (
            float(np.median(dominance)) if dominance else math.inf
        )
        loss_not_dominated = median_gradient_dominance <= 0.90
        common_seed_count = int(main["seed_count"])
        seed_contract_complete = common_seed_count >= 3
        finite_outputs = all(
            math.isfinite(float(main[key]))
            for key in main if key.endswith("_mean")
        )
        method_valid = all(
            (
                seed_stable,
                finite_outputs,
                conditioning_well_formed,
                loss_not_dominated,
                checkpoint_contracts_complete,
                seed_contract_complete,
            )
        )
        p1_candidates = [p1a, p1b]
        best_p1 = min(
            p1_candidates,
            key=lambda row: row["test_rollout_16ms_rmse_mv_mean"],
        )
        informative_branching = [
            row
            for row in self.branching_rows
            if row["model"] in main_ids
            and float(row["teacher_future_distance_mv"]) > 1e-3
        ]
        collapse_fraction = (
            float(np.mean([bool(row["collapsed"]) for row in informative_branching]))
            if informative_branching
            else 1.0
        )
        branching_not_systematically_collapsed = collapse_fraction <= 0.5
        if self.config.profile == "smoke":
            decision = "SMOKE_ONLY"
            method_valid = False
        elif not method_valid:
            decision = "NO_GO_METHODOLOGICAL"
        elif all(
            (
                clear_rollout_advantage,
                nonlinear_advantage,
                drift_reduced,
                event_improved,
                branching_retained,
                branching_not_systematically_collapsed,
            )
        ):
            decision = "GO"
        elif (
            main["test_rollout_16ms_rmse_mv_mean"] <= 1.02 * b1_rollout
            and (rollout_advantage or nonlinear_advantage)
        ):
            decision = "CONDITIONAL_GO"
        else:
            decision = "NO_GO_MODEL_PROVISIONAL"
        scientific_questions = {
            "b3_beats_b1_after_reconditioning": {
                "answer": bool(
                    main["test_voltage_rmse_mv_mean"] < b1_one_step
                    or main["test_rollout_16ms_rmse_mv_mean"] < b1_rollout
                ),
                "b3_one_step_voltage_rmse_mv": main["test_voltage_rmse_mv_mean"],
                "b1_one_step_voltage_rmse_mv": b1_one_step,
                "b3_rollout_rmse_mv": main["test_rollout_16ms_rmse_mv_mean"],
                "b1_rollout_rmse_mv": b1_rollout,
            },
            "b3_rollout_advantage_consistent_across_seeds": {
                "answer": bool(rollout_advantage and seed_stable),
                "mean_rmse_mv": main["test_rollout_16ms_rmse_mv_mean"],
                "std_rmse_mv": main["test_rollout_16ms_rmse_mv_std"],
            },
            "negative_drift_reduced_vs_original_02": {
                "answer": bool(drift_reduced),
                "reconditioned_drift_mv": main_drift,
                "original_02_drift_mv": original_drift,
            },
            "privileged_supervision_improves_rollout": {
                "answer": bool(
                    best_p1["test_rollout_16ms_rmse_mv_mean"]
                    < main["test_rollout_16ms_rmse_mv_mean"]
                ),
                "best_variant": best_p1["privileged_mode"],
                "p0_rmse_mv": main["test_rollout_16ms_rmse_mv_mean"],
                "best_p1_rmse_mv": best_p1["test_rollout_16ms_rmse_mv_mean"],
            },
            "u2_better_than_u1": {
                "answer": bool(
                    main["test_rollout_16ms_rmse_mv_mean"]
                    < u1["test_rollout_16ms_rmse_mv_mean"]
                ),
                "u2_rollout_rmse_mv": main["test_rollout_16ms_rmse_mv_mean"],
                "u1_rollout_rmse_mv": u1["test_rollout_16ms_rmse_mv_mean"],
            },
            "branching_still_systematically_damped": {
                "answer": bool(collapse_fraction > 0.5 or not branching_retained),
                "informative_pair_count": len(informative_branching),
                "collapse_fraction": collapse_fraction,
                "median_divergence_retention": main[
                    "branching_divergence_retention_median_mean"
                ],
            },
        }
        return {
            "decision": decision,
            "methodological_validity": method_valid,
            "criteria": {
                "stable_across_seeds": seed_stable,
                "finite_primary_outputs": finite_outputs,
                "zero_inflated_normalization_well_conditioned": conditioning_well_formed,
                "median_weighted_gradient_dominance_at_most_0_90": loss_not_dominated,
                "checkpoint_contracts_complete_and_self_consistent": checkpoint_contracts_complete,
                "common_seed_contract_complete": seed_contract_complete,
                "b3_rollout_better_than_b1": rollout_advantage,
                "b3_rollout_at_least_5_percent_better_than_b1": clear_rollout_advantage,
                "b3_nonlinear_rollout_better_than_b1": nonlinear_advantage,
                "negative_drift_reduced_vs_original_02": drift_reduced,
                "learnable_events_improved_vs_02": event_improved,
                "median_branching_retention_at_least_0_10": branching_retained,
                "branching_not_systematically_collapsed": branching_not_systematically_collapsed,
            },
            "identifiability_limitations": {
                "release_outcome_missing": not self.store.release_contract[
                    "release_outcome_available"
                ],
                "not_learnable_event_classes": [
                    kind for kind, count in self.event_support.items() if count == 0
                ],
                "small_event_support": self.event_support,
            },
            "scientific_questions": scientific_questions,
            "inputs": {
                "main_b3": main,
                "rollout_evaluation_horizon_ms": evaluation_horizon,
                "b1_u2_rollout_16ms_rmse_mv": b1_rollout,
                "b3_nonlinear_rollout_rmse_mv": main_nonlinear_rollout,
                "b1_nonlinear_rollout_rmse_mv": b1_nonlinear_rollout,
                "median_weighted_gradient_dominance": median_gradient_dominance,
                "conditioning_diagnostics": self.conditioning_diagnostics,
                "original_02": original,
                "original_02_learnable_event_macro_f1": original_learnable_event_f1,
            },
        }

    def _original_learnable_event_f1(self) -> float:
        import pandas as pd

        table = pd.read_parquet(self.original_root / "event_metrics.parquet")
        learnable = {
            kind for kind, count in self.event_support.items() if count > 0
        }
        selected = table[
            table["model"].astype(str).str.contains("B3_structured-full-U2-P1")
            & table["split"].isin(
                ["deterministic_test", "event_boundary_test", "branching_test"]
            )
            & table["event_kind"].isin(learnable)
            & (table["support"] > 0)
        ]
        return float(selected["f1"].mean()) if len(selected) else 0.0

    def finalize(self) -> Dict[str, Any]:
        self.seed_rows = self._seed_summary()
        aggregate = self._aggregate_seeds(self.seed_rows)
        write_parquet(self.output_dir / "one_step_metrics.parquet", self.one_step_rows)
        write_parquet(self.output_dir / "rollout_metrics.parquet", self.rollout_rows)
        write_parquet(self.output_dir / "event_metrics.parquet", self.event_rows)
        write_parquet(self.output_dir / "branching_metrics.parquet", self.branching_rows)
        write_parquet(self.output_dir / "gradient_contribution_report.parquet", self.gradient_rows)
        write_parquet(self.output_dir / "seed_level_metrics.parquet", self.seed_rows)
        write_parquet(self.output_dir / "seed_aggregate_metrics.parquet", aggregate)
        _write_json(self.output_dir / "model_registry.json", self.registry)
        _write_json(self.output_dir / "training_history.json", self.histories)
        _write_json(
            self.output_dir / "checkpoint_fingerprints.json",
            self.checkpoint_contracts,
        )
        comparison = {
            "original_02_final_report_sha256": _sha256(
                self.original_root / "final_report.json"
            ),
            "original_02_decision": self.original_report["decision"],
            "original_02_comparison_inputs": self.original_report["comparison_inputs"],
            "reconditioned_seed_aggregates": aggregate,
        }
        _write_json(self.output_dir / "comparison_with_original_02.json", comparison)
        decision = self._decision(aggregate)
        report = {
            "schema_version": "02b-final-v1",
            "valid": True,
            "methodological_validity": decision["methodological_validity"],
            "modeling_result": decision["decision"],
            "decision": decision,
            "identifiability_limitations": decision["identifiability_limitations"],
            "scientific_questions": decision["scientific_questions"],
            "seed_aggregate_metrics": aggregate,
            "comparison_with_original_02": comparison,
            "architecture_interpretation": (
                "Only the frozen B3 diagnostic backbone was tested. This result "
                "does not evaluate Hines, persistent latents, morphology reduction, "
                "Mamba or S4."
            ),
        }
        _write_json(self.output_dir / "final_report.json", report)
        self._make_figures(aggregate)
        return report

    def _make_figures(self, aggregate: Sequence[Mapping[str, Any]]) -> None:
        import matplotlib.pyplot as plt

        labels = [
            f"{row['input_encoding']}/{row['synapse_mode']}/{row['privileged_mode']}"
            for row in aggregate
            if row["gate_transform"] == "logit"
        ]
        rows = [row for row in aggregate if row["gate_transform"] == "logit"]
        x = np.arange(len(rows))
        fig, axes = plt.subplots(2, 1, figsize=(10, 9), sharex=True)
        axes[0].bar(
            x,
            [row["test_voltage_rmse_mv_mean"] for row in rows],
            yerr=[row["test_voltage_rmse_mv_std"] for row in rows],
        )
        axes[0].set_ylabel("one-step voltage RMSE (mV)")
        axes[1].bar(
            x,
            [row["test_rollout_16ms_rmse_mv_mean"] for row in rows],
            yerr=[row["test_rollout_16ms_rmse_mv_std"] for row in rows],
        )
        axes[1].set_ylabel(
            f"{max(self.config.rollout_horizons_ms)} ms rollout RMSE (mV)"
        )
        axes[1].set_xticks(x, labels, rotation=30, ha="right")
        for axis in axes:
            axis.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(self.figure_dir / "seed_aggregate_comparison.png", dpi=180)
        plt.close(fig)
