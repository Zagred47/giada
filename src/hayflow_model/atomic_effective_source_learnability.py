"""06b-p: atomic effective-source learnability and boundary-shift matrix.

The preceding 06b-o run established an exact cable-equation decomposition and
a useful passive Hines prior, but almost every learned-source arm selected its
zero-output checkpoint. This train-only component playground first separates
target scaling from objective alignment, then adaptively crosses three nested
causal synaptic input contracts with two physical targets. Disjoint calibration
halves prevent the adaptive stage from reusing its own selection evidence.
Frozen development evaluates teacher boundaries, recursive voltage boundaries,
regime gradients and directional sensitivity. Algebraic substep diagnostics are
nonselective because complete intermediate mechanism state was not stored.
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
    CausalMembraneSourceCell,
    HINES_SOURCE,
    EffectiveMembraneSourceConfig,
    EffectiveMembraneSourcePlayground,
)
from ..hayflow_data.hines_inputs import (
    HINES_SYNAPTIC_FEATURE_NAMES,
    encode_realized_synaptic_drive,
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

COMPACT_MOMENTS = "compact_moments"
EXACT_EVENTS = "exact_events"
BOUNDARY_COMPLETE = "boundary_complete"
NET_EFFECTIVE_SOURCE = "net_effective_source"
INTRINSIC_RESIDUAL = "intrinsic_residual"

INPUT_CONTRACTS = (COMPACT_MOMENTS, EXACT_EVENTS, BOUNDARY_COMPLETE)
PHYSICAL_TARGETS = (NET_EFFECTIVE_SOURCE, INTRINSIC_RESIDUAL)


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
    input_contracts: Tuple[str, ...] = INPUT_CONTRACTS
    physical_targets: Tuple[str, ...] = PHYSICAL_TARGETS
    fragment_training_steps: int = 300
    fragment_checkpoints: Tuple[int, ...] = (0, 50, 100, 200, 300)
    fragment_batch_transition_count: int = 32
    structured_scale_quantile: float = 0.99
    structured_scale_floor: float = 1e-6
    substep_audit_transition_count: int = 12
    substep_audit_dt_ms: Tuple[float, ...] = (1.0, 0.5, 0.25)
    voltage_sensitivity_perturbations_mv: Tuple[float, ...] = (0.25, 1.0)

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
        if tuple(self.input_contracts) != INPUT_CONTRACTS:
            raise ValueError("06b-p nested input-contract matrix changed")
        if tuple(self.physical_targets) != PHYSICAL_TARGETS:
            raise ValueError("06b-p physical-target matrix changed")
        if self.fragment_training_steps <= 0:
            raise ValueError("fragment_training_steps must be positive")
        if (
            self.fragment_checkpoints[0] != 0
            or self.fragment_checkpoints[-1] != self.fragment_training_steps
        ):
            raise ValueError("fragment checkpoints must span the training budget")
        if self.fragment_batch_transition_count <= 0:
            raise ValueError("fragment batch size must be positive")
        if not 0.5 < self.structured_scale_quantile < 1.0:
            raise ValueError("structured_scale_quantile must be in (0.5, 1)")
        if self.structured_scale_floor <= 0.0:
            raise ValueError("structured_scale_floor must be positive")
        if self.substep_audit_transition_count <= 0:
            raise ValueError("substep audit transition count must be positive")
        if tuple(self.substep_audit_dt_ms) != (1.0, 0.5, 0.25):
            raise ValueError("06b-p substep audit grid changed")

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
            "input_contracts",
            "physical_targets",
            "fragment_checkpoints",
            "substep_audit_dt_ms",
            "voltage_sensitivity_perturbations_mv",
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
        self.target_source_scales: Dict[Tuple[str, str], np.ndarray] = {}
        self.structured_feature_scale: Optional[np.ndarray] = None
        self.atomic_models: Dict[Tuple[str, int], Any] = {}
        self.fragment_models: Dict[Tuple[str, int], Any] = {}
        self.atomic_training_valid = False
        self.fragment_training_valid = False
        self.adaptive_choice: Dict[str, str] = {}

    def _atomic_specs(self) -> Tuple[Tuple[str, str], ...]:
        return tuple(itertools.product(self.config.scaling_modes, self.config.objectives))

    @staticmethod
    def _atomic_key(spec: Tuple[str, str]) -> str:
        return "|".join(spec)

    def _fragment_specs(self) -> Tuple[Tuple[str, str], ...]:
        return tuple(itertools.product(self.config.input_contracts, self.config.physical_targets))

    @staticmethod
    def _fragment_key(spec: Tuple[str, str]) -> str:
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
        for name in (
            "structured_exact_events",
            "structured_boundary_complete",
            "synaptic_conductance_us",
            "synaptic_source_na",
            "somatic_current_na",
        ):
            if name in values:
                payload[name] = values[name][window, step]
        return {
            name: atomic.torch.as_tensor(value, device=device)
            for name, value in payload.items()
        }

    def _calibration_rows(self, stage: str) -> np.ndarray:
        """Disjoint deterministic calibration halves for the adaptive stages."""

        rows = np.arange(self._flat_count("calibration"), dtype=np.int64)
        if stage == "atomic":
            return rows[::2]
        if stage == "fragment":
            return rows[1::2]
        raise ValueError(stage)

    def _materialize_structured_inputs(self) -> Dict[str, Any]:
        """Create nested causal synaptic views without changing tensor width.

        ``exact_events`` contains only events realized inside the current
        millisecond. ``boundary_complete`` additionally propagates authentic
        point-process A/B states already present in S_t.  Both are computed at
        V_t, never at the teacher endpoint.
        """

        reports: Dict[str, Any] = {}
        fit_full: Optional[np.ndarray] = None
        for role, values in self.window_data.items():
            flat = values["indices"].reshape(-1)
            voltage = values["voltage_t"].reshape(-1, self.layout.segment_count)
            raw_state = self.store.read_state(flat, "t").astype(np.float32)
            exact = encode_realized_synaptic_drive(
                self.store, flat, voltage, dt_ms=self.config.macro_step_ms
            )
            complete = encode_realized_synaptic_drive(
                self.store,
                flat,
                voltage,
                dt_ms=self.config.macro_step_ms,
                raw_state_t=raw_state,
            )
            shape = values["voltage_t"].shape[:2]
            values["structured_exact_events"] = exact["synaptic_features"].reshape(
                *shape, self.layout.segment_count, -1
            )
            values["structured_boundary_complete"] = complete[
                "synaptic_features"
            ].reshape(*shape, self.layout.segment_count, -1)
            for name in (
                "synaptic_conductance_us",
                "synaptic_source_na",
                "somatic_current_na",
            ):
                values[name] = complete[name].reshape(
                    *shape, self.layout.segment_count
                )
            if role == "fit":
                fit_full = complete["synaptic_features"]
            tail = complete["synaptic_features"] - exact["synaptic_features"]
            reports[role] = {
                "transition_count": int(len(flat)),
                "exact_event_nonzero_fraction": float(
                    np.mean(exact["synaptic_features"] != 0.0)
                ),
                "boundary_tail_nonzero_fraction": float(np.mean(tail != 0.0)),
                "boundary_tail_rms": float(
                    np.sqrt(np.mean(tail.astype(np.float64) ** 2))
                ),
                "maximum_boundary_tail": float(np.max(np.abs(tail))),
            }
        if fit_full is None:
            raise RuntimeError("fit structured input view was not materialized")
        scale = np.quantile(
            np.abs(fit_full.astype(np.float64)),
            self.config.structured_scale_quantile,
            axis=(0, 1),
        )
        self.structured_feature_scale = np.maximum(
            scale, self.config.structured_scale_floor
        ).astype(np.float64)
        return {
            "valid": bool(np.isfinite(self.structured_feature_scale).all()),
            "feature_count": len(HINES_SYNAPTIC_FEATURE_NAMES),
            "feature_names": list(HINES_SYNAPTIC_FEATURE_NAMES),
            "normalization_fit_role_only": True,
            "scale_quantile": self.config.structured_scale_quantile,
            "scale_minimum": float(self.structured_feature_scale.min()),
            "scale_median": float(np.median(self.structured_feature_scale)),
            "scale_maximum": float(self.structured_feature_scale.max()),
            "roles": reports,
        }

    def _fragment_feature_width(self) -> int:
        return self._feature_width() + len(HINES_SYNAPTIC_FEATURE_NAMES)

    def _new_fragment_model(self, seed: int, device: Any) -> Any:
        atomic.torch.manual_seed(seed + 680000)
        model = CausalMembraneSourceCell(
            self._fragment_feature_width(),
            len(self.layout.region_names),
            self.config.matrix_region_embedding_width,
            self.config.matrix_hidden_width,
            self.config.normalized_output_limit,
        ).to(device)
        count = sum(value.numel() for value in model.parameters())
        if count > self.config.maximum_source_model_parameter_count:
            raise RuntimeError(f"06b-p fragment model has {count} parameters")
        return model

    def _structured_tensor(self, mode: str, batch: Mapping[str, Any]) -> Any:
        if mode == COMPACT_MOMENTS:
            raw = atomic.torch.zeros_like(batch["structured_boundary_complete"])
        elif mode == EXACT_EVENTS:
            raw = batch["structured_exact_events"]
        elif mode == BOUNDARY_COMPLETE:
            raw = batch["structured_boundary_complete"]
        else:
            raise ValueError(mode)
        scale = atomic.torch.as_tensor(
            self.structured_feature_scale,
            dtype=raw.dtype,
            device=raw.device,
        )
        return raw / scale[None, None, :]

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

    def _physical_source_target(
        self, batch: Mapping[str, Any], target_kind: str
    ) -> Any:
        if target_kind == NET_EFFECTIVE_SOURCE:
            return self._normalized_source_target(
                batch["voltage"], batch["target_voltage"], "authentic"
            )
        if target_kind != INTRINSIC_RESIDUAL:
            raise ValueError(target_kind)
        diagonal, _, base_rhs, _ = self._physical_terms(
            batch["voltage"], "authentic"
        )
        conductance = batch["synaptic_conductance_us"]
        synaptic_rhs = batch["synaptic_source_na"] + batch["somatic_current_na"]
        augmented_scale = (diagonal + conductance) * self.config.voltage_scale_mv
        augmented_target = (
            self._matrix_apply(batch["target_voltage"], "authentic")
            + conductance * batch["target_voltage"]
        )
        return (augmented_target - base_rhs - synaptic_rhs) / augmented_scale

    def _apply_physical_target(
        self,
        normalized_source: Any,
        current_voltage: Any,
        target_kind: str,
        batch: Mapping[str, Any],
    ) -> Any:
        if target_kind == NET_EFFECTIVE_SOURCE:
            return self._apply_output(
                HINES_SOURCE, normalized_source, current_voltage, "authentic"
            )
        if target_kind != INTRINSIC_RESIDUAL:
            raise ValueError(target_kind)
        diagonal, coupling, base_rhs, _ = self._physical_terms(
            current_voltage, "authentic"
        )
        conductance = batch["synaptic_conductance_us"]
        synaptic_rhs = batch["synaptic_source_na"] + batch["somatic_current_na"]
        augmented_diagonal = diagonal + conductance
        source_scale = augmented_diagonal * self.config.voltage_scale_mv
        solver = self._topology_tensors(
            "authentic", current_voltage.dtype, current_voltage.device
        )["solver"]
        return solver(
            augmented_diagonal,
            coupling,
            base_rhs + synaptic_rhs + normalized_source * source_scale,
        )

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
        self.target_source_scales = {
            (NET_EFFECTIVE_SOURCE, name): values
            for name, values in self.source_scales.items()
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

    def _compute_intrinsic_source_scales(self, device: Any) -> Dict[str, Any]:
        rows = np.arange(self._flat_count("fit"), dtype=np.int64)
        batch = self._flat_tensors("fit", rows, device)
        with atomic.torch.no_grad():
            target = self._physical_source_target(batch, INTRINSIC_RESIDUAL)
        absolute = np.abs(target.detach().cpu().numpy().astype(np.float64))
        global_scale = max(
            self.config.source_scale_floor,
            float(np.quantile(absolute, self.config.source_scale_quantile)),
        )
        region_scale = np.full(self.layout.segment_count, global_scale, dtype=np.float64)
        for region_id in range(len(self.layout.region_names)):
            mask = self.layout.segment_region_ids == region_id
            if bool(mask.any()):
                region_scale[mask] = max(
                    self.config.source_scale_floor,
                    float(
                        np.quantile(
                            absolute[:, mask], self.config.source_scale_quantile
                        )
                    ),
                )
        target_scales = {
            RAW_SOURCE: np.ones(self.layout.segment_count, dtype=np.float64),
            GLOBAL_P99: np.full(
                self.layout.segment_count, global_scale, dtype=np.float64
            ),
            REGION_P99: region_scale,
        }
        for mode, values in target_scales.items():
            self.target_source_scales[(INTRINSIC_RESIDUAL, mode)] = values
        return {
            "valid": bool(np.isfinite(absolute).all()),
            "global_scale": global_scale,
            "raw_rms": float(np.sqrt(np.mean(absolute**2))),
            "raw_p99": float(np.quantile(absolute, 0.99)),
            "raw_maximum": float(absolute.max()),
        }

    def _target_scale_tensor(self, target: str, mode: str, reference: Any) -> Any:
        return atomic.torch.as_tensor(
            self.target_source_scales[(target, mode)],
            dtype=reference.dtype,
            device=reference.device,
        )[None, :]

    def _source_contract_audit(self, device: Any) -> Dict[str, Any]:
        rows = np.arange(self._flat_count("fit"), dtype=np.int64)
        batch = self._flat_tensors("fit", rows, device)
        with atomic.torch.no_grad():
            net = self._physical_source_target(batch, NET_EFFECTIVE_SOURCE)
            intrinsic = self._physical_source_target(batch, INTRINSIC_RESIDUAL)
            zeros = atomic.torch.zeros_like(net)
            passive = self._apply_physical_target(
                zeros, batch["voltage"], NET_EFFECTIVE_SOURCE, batch
            )
            mechanistic = self._apply_physical_target(
                zeros, batch["voltage"], INTRINSIC_RESIDUAL, batch
            )
        target = batch["target_voltage"].cpu().numpy()
        passive_np = passive.cpu().numpy()
        mechanistic_np = mechanistic.cpu().numpy()
        net_np = net.cpu().numpy().astype(np.float64)
        intrinsic_np = intrinsic.cpu().numpy().astype(np.float64)
        return {
            "valid": bool(
                np.isfinite(net_np).all()
                and np.isfinite(intrinsic_np).all()
                and np.isfinite(mechanistic_np).all()
            ),
            "fit_transition_count": int(len(rows)),
            "passive_hines_endpoint_rmse_mv": self._rmse(passive_np, target),
            "known_synaptic_hines_endpoint_rmse_mv": self._rmse(
                mechanistic_np, target
            ),
            "known_synaptic_gain_over_passive_fraction": 1.0
            - self._rmse(mechanistic_np, target)
            / max(self._rmse(passive_np, target), 1e-12),
            "net_effective_source": {
                "rms": float(np.sqrt(np.mean(net_np**2))),
                "p99_absolute": float(np.quantile(np.abs(net_np), 0.99)),
                "maximum_absolute": float(np.max(np.abs(net_np))),
            },
            "intrinsic_residual": {
                "rms": float(np.sqrt(np.mean(intrinsic_np**2))),
                "p99_absolute": float(
                    np.quantile(np.abs(intrinsic_np), 0.99)
                ),
                "maximum_absolute": float(np.max(np.abs(intrinsic_np))),
            },
            "target_correlation": self._correlation(net_np, intrinsic_np),
            "selection_eligible": False,
        }

    def _substep_source_support_audit(self) -> Dict[str, Any]:
        """Measure macro-step conditioning from stored voltage microtraces.

        This is deliberately an algebraic diagnostic, not a trainable arm:
        intermediate mechanism and synapse states are unavailable, so a fair
        substep learner cannot yet be built from this dataset.
        """

        indices = self.window_data["fit"]["indices"].reshape(-1)
        indices = indices[: self.config.substep_audit_transition_count]
        parent = np.asarray(self.physical["parent_ids"], dtype=np.int64)
        coupling = np.asarray(
            self.physical["axial_conductance_to_parent_us"], dtype=np.float64
        )
        capacitance = np.asarray(self.physical["capacitance_uf"], dtype=np.float64)
        leak = np.asarray(self.physical["leak_conductance_us"], dtype=np.float64)
        reversal = np.asarray(self.physical["leak_reversal_mv"], dtype=np.float64)
        axial_total = self._axial_total(parent, coupling)
        cpu_solver = self._topology_tensors(
            "authentic", atomic.torch.float64, atomic.torch.device("cpu")
        )["solver"]

        def matrix_apply(voltage: np.ndarray, diagonal: np.ndarray) -> np.ndarray:
            result = diagonal * voltage
            for child, parent_id in enumerate(parent):
                if child == int(parent_id):
                    continue
                edge = coupling[child]
                result[child] -= edge * voltage[int(parent_id)]
                result[int(parent_id)] -= edge * voltage[child]
            return result

        traces = []
        for index in indices:
            trace = np.asarray(self.store.microtrace(int(index)), dtype=np.float64)
            if trace.ndim != 2:
                raise RuntimeError("all-segment voltage microtrace must be 2-D")
            if trace.shape[1] != self.layout.segment_count:
                if trace.shape[0] == self.layout.segment_count:
                    trace = trace.T
                else:
                    raise RuntimeError("all-segment voltage microtrace width mismatch")
            traces.append(trace)
        reports = {}
        for dt in self.config.substep_audit_dt_ms:
            source_values = []
            passive_errors = []
            for trace in traces:
                grid = np.linspace(0.0, 1.0, len(trace))
                requested = np.arange(0.0, 1.0 + 0.5 * dt, dt)
                positions = [int(np.argmin(np.abs(grid - value))) for value in requested]
                teacher = trace[positions]
                mass = (1000.0 / dt) * capacitance
                diagonal = mass + leak + axial_total
                current = teacher[0].copy()
                for step in range(len(teacher) - 1):
                    base_rhs = mass * teacher[step] + leak * reversal
                    source = (
                        matrix_apply(teacher[step + 1], diagonal) - base_rhs
                    ) / (diagonal * self.config.voltage_scale_mv)
                    source_values.append(source)
                    rhs = mass * current + leak * reversal
                    with atomic.torch.no_grad():
                        predicted = cpu_solver(
                            atomic.torch.as_tensor(diagonal[None, :]),
                            atomic.torch.as_tensor(coupling[None, :]),
                            atomic.torch.as_tensor(rhs[None, :]),
                        ).cpu().numpy()[0]
                    current = predicted
                passive_errors.append((current - teacher[-1]) ** 2)
            sources = np.concatenate(source_values)
            reports[str(dt)] = {
                "substeps_per_ms": int(round(1.0 / dt)),
                "source_rms": float(np.sqrt(np.mean(sources**2))),
                "source_p99_absolute": float(np.quantile(np.abs(sources), 0.99)),
                "source_maximum_absolute": float(np.max(np.abs(sources))),
                "recursive_passive_endpoint_rmse_mv": float(
                    np.sqrt(np.mean(np.concatenate(passive_errors)))
                ),
            }
        return {
            "valid": True,
            "transition_count": int(len(indices)),
            "dt_ms": reports,
            "selection_eligible": False,
            "trainable_substep_arm_built": False,
            "limitation": (
                "stored microtraces contain voltage but not complete intermediate "
                "mechanism and synapse state"
            ),
        }

    def prepare_atomic_source_learnability(self) -> Dict[str, Any]:
        base = self.prepare_effective_membrane_source_playground()
        _, source = verified_06bo_artifact_root(
            self.artifact_06bo_source,
            self.output_dir.parent / ".06bp_artifact_cache" / "06bo",
        )
        device = atomic.torch.device("cuda" if atomic.torch.cuda.is_available() else "cpu")
        structured_report = self._materialize_structured_inputs()
        scale_report = self._compute_source_scales(device)
        intrinsic_scale_report = self._compute_intrinsic_source_scales(device)
        source_contract_audit = self._source_contract_audit(device)
        substep_audit = self._substep_source_support_audit()
        fragment_parameter_count = sum(
            value.numel()
            for value in self._new_fragment_model(
                self.config.pilot_seeds[0], device
            ).parameters()
        )
        report = {
            **base,
            "schema_version": "06b-p-adaptive-multifactor-contract-v2",
            "experiment": "atomic_effective_source_learnability",
            "source_06bo": source,
            "scientific_question": (
                "Which atomic cause prevents effective-source learning: scale/objective, "
                "lossy event encoding, missing boundary synapse state, net-source "
                "cancellation, or the one-millisecond macro-step?"
            ),
            "factorial_axes": {
                "source_scaling": list(self.config.scaling_modes),
                "objective": list(self.config.objectives),
            },
            "factor_arm_count": len(self._atomic_specs()),
            "adaptive_fragment_axes": {
                "input_contract": list(self.config.input_contracts),
                "physical_target": list(self.config.physical_targets),
            },
            "adaptive_fragment_arm_count": len(self._fragment_specs()),
            "adaptive_fragment_parameter_count_per_arm": fragment_parameter_count,
            "adaptive_rule": (
                "stage-1 scaling/objective selected on calibration half A; stage-2 "
                "input/target selected on disjoint calibration half B"
            ),
            "teacher_boundary_training": True,
            "frozen_recursive_voltage_boundary_test": True,
            "teacher_state_used_during_boundary_test": True,
            "predicted_state_trained": False,
            "temporal_memory_trained": False,
            "same_numeric_input_tensor": True,
            "same_initialization_within_seed": True,
            "same_minibatch_stream_within_seed": True,
            "source_scale_report": scale_report,
            "intrinsic_source_scale_report": intrinsic_scale_report,
            "structured_input_report": structured_report,
            "source_contract_audit": source_contract_audit,
            "substep_source_support_audit": substep_audit,
            "substep_models_trained": False,
            "teacher_endpoint_used_as_model_input": False,
            "teacher_source_oracle_selectable": False,
            "validation_state_accessed": False,
            "test_state_accessed": False,
            "configuration": asdict(self.config),
        }
        atomic._write_json(self.output_dir / "atomic_source_contract.json", report)
        atomic._write_json(
            self.output_dir / "source_contract_audit.json", source_contract_audit
        )
        atomic._write_json(
            self.output_dir / "substep_source_support_audit.json", substep_audit
        )
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
        self,
        model: Any,
        mode: str,
        role: str,
        device: Any,
        rows: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        if rows is None:
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
                            model,
                            spec[0],
                            "calibration",
                            device,
                            self._calibration_rows("atomic"),
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
            "calibration_partition": "deterministic_even_rows_half_A",
            "development_used_during_training": False,
            "reports": reports,
        }
        self.atomic_training_valid = bool(report["valid"])
        atomic._write_json(self.output_dir / "atomic_source_training.json", report)
        return report

    def _select_adaptive_choice(
        self, atomic_training: Mapping[str, Any]
    ) -> Dict[str, str]:
        scores: Dict[str, List[float]] = {
            self._atomic_key(spec): [] for spec in self._atomic_specs()
        }
        for seed in atomic_training["reports"].values():
            for key, row in seed["selected"].items():
                scores[key].append(
                    float(row["selected_calibration_endpoint_rmse_mv"])
                )
        selected = min(scores, key=lambda key: self._median(scores[key]))
        scaling, objective = selected.split("|")
        self.adaptive_choice = {
            "atomic_arm": selected,
            "scaling": scaling,
            "objective": objective,
            "selection_role": "calibration_half_A",
        }
        return dict(self.adaptive_choice)

    def _fragment_features(
        self, batch: Mapping[str, Any], input_contract: str
    ) -> Any:
        base, _, _ = self._teacher_features(batch)
        return atomic.torch.cat(
            (base, self._structured_tensor(input_contract, batch)), dim=-1
        )

    def _fragment_forward(
        self,
        model: Any,
        spec: Tuple[str, str],
        scaling: str,
        batch: Mapping[str, Any],
    ) -> Tuple[Any, Any, Any, Any]:
        input_contract, physical_target = spec
        features = self._fragment_features(batch, input_contract)
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
        scale = self._target_scale_tensor(physical_target, scaling, standardized)
        decoded = standardized * scale
        target_source = self._physical_source_target(batch, physical_target)
        prediction = self._apply_physical_target(
            decoded, batch["voltage"], physical_target, batch
        )
        return standardized, decoded, target_source, prediction

    def _fragment_loss(
        self,
        model: Any,
        spec: Tuple[str, str],
        scaling: str,
        objective: str,
        batch: Mapping[str, Any],
    ) -> Tuple[Any, Dict[str, Any]]:
        standardized, decoded, target_source, prediction = self._fragment_forward(
            model, spec, scaling, batch
        )
        target_standardized = target_source / self._target_scale_tensor(
            spec[1], scaling, standardized
        )
        native = atomic.torch_functional.smooth_l1_loss(
            standardized, target_standardized
        )
        endpoint_error = (
            prediction - batch["target_voltage"]
        ) / self.config.voltage_scale_mv
        activity = batch["target_voltage"] - batch["voltage"]
        endpoint = atomic.torch.mean(
            self._activity_weight(activity)
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

    def _fragment_metrics(
        self,
        model: Any,
        spec: Tuple[str, str],
        scaling: str,
        role: str,
        device: Any,
        rows: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        if rows is None:
            rows = np.arange(self._flat_count(role), dtype=np.int64)
        batch = self._flat_tensors(role, rows, device)
        model.eval()
        with atomic.torch.no_grad():
            standardized, decoded, target_source, prediction = self._fragment_forward(
                model, spec, scaling, batch
            )
            zeros = atomic.torch.zeros_like(decoded)
            physical_baseline = self._apply_physical_target(
                zeros, batch["voltage"], spec[1], batch
            )
            passive = self._apply_physical_target(
                zeros, batch["voltage"], NET_EFFECTIVE_SOURCE, batch
            )
        prediction_np = prediction.cpu().numpy()
        baseline_np = physical_baseline.cpu().numpy()
        passive_np = passive.cpu().numpy()
        target_np = batch["target_voltage"].cpu().numpy()
        current_np = batch["voltage"].cpu().numpy()
        source_np = target_source.cpu().numpy()
        decoded_np = decoded.cpu().numpy()
        activity = np.abs(target_np - current_np)
        masks = {
            "quiescent_lt_1mV": activity < 1.0,
            "moderate_1_to_5mV": (activity >= 1.0) & (activity < 5.0),
            "active_ge_5mV": activity >= 5.0,
            "regenerative_ge_20mV": activity >= 20.0,
        }
        model_rmse = self._rmse(prediction_np, target_np)
        baseline_rmse = self._rmse(baseline_np, target_np)
        passive_rmse = self._rmse(passive_np, target_np)
        source_rmse = self._rmse(decoded_np, source_np)
        source_rms = float(np.sqrt(np.mean(source_np.astype(np.float64) ** 2)))
        return {
            "endpoint_rmse_mv": model_rmse,
            "physical_baseline_endpoint_rmse_mv": baseline_rmse,
            "passive_endpoint_rmse_mv": passive_rmse,
            "endpoint_gain_over_physical_baseline_fraction": 1.0
            - model_rmse / max(baseline_rmse, 1e-12),
            "endpoint_gain_over_passive_fraction": 1.0
            - model_rmse / max(passive_rmse, 1e-12),
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
    def _gradient_vector(parameters: Sequence[Any]) -> np.ndarray:
        values = [
            parameter.grad.detach().cpu().numpy().reshape(-1)
            for parameter in parameters
            if parameter.grad is not None
        ]
        return np.concatenate(values) if values else np.zeros(1, dtype=np.float32)

    def _regime_gradient_probe(
        self,
        model: Any,
        spec: Tuple[str, str],
        scaling: str,
        objective: str,
        device: Any,
    ) -> Dict[str, Any]:
        values = self.window_data["fit"]
        activity = np.max(
            np.abs(values["voltage_t1"] - values["voltage_t"]), axis=-1
        ).reshape(-1)
        masks = {
            "quiescent": activity < 1.0,
            "moderate": (activity >= 1.0) & (activity < 5.0),
            "active": activity >= 5.0,
        }
        gradients: Dict[str, np.ndarray] = {}
        norms: Dict[str, float] = {}
        for name, mask in masks.items():
            rows = np.flatnonzero(mask)[: self.config.fragment_batch_transition_count]
            if not len(rows):
                continue
            model.zero_grad(set_to_none=True)
            loss, _ = self._fragment_loss(
                model,
                spec,
                scaling,
                objective,
                self._flat_tensors("fit", rows, device),
            )
            loss.backward()
            vector = self._gradient_vector(list(model.parameters())).astype(np.float64)
            gradients[name] = vector
            norms[name] = float(np.linalg.norm(vector))
        cosines = {}
        for left, right in itertools.combinations(sorted(gradients), 2):
            denominator = max(norms[left] * norms[right], 1e-30)
            cosines[f"{left}|{right}"] = float(
                np.dot(gradients[left], gradients[right]) / denominator
            )
        model.zero_grad(set_to_none=True)
        return {"gradient_norms": norms, "gradient_cosines": cosines}

    def train_adaptive_fragment_matrix(
        self, atomic_training: Mapping[str, Any]
    ) -> Dict[str, Any]:
        if not self.atomic_training_valid:
            raise RuntimeError("stage-1 atomic training is incomplete")
        choice = self._select_adaptive_choice(atomic_training)
        scaling, objective = choice["scaling"], choice["objective"]
        specs = self._fragment_specs()
        device = atomic.torch.device("cuda" if atomic.torch.cuda.is_available() else "cpu")
        reports: Dict[str, Any] = {}
        calibration_rows = self._calibration_rows("fragment")
        for seed in self.config.pilot_seeds:
            models = {spec: self._new_fragment_model(seed, device) for spec in specs}
            optimizers = {
                spec: atomic.torch.optim.AdamW(
                    models[spec].parameters(),
                    lr=self.config.matrix_learning_rate,
                    weight_decay=self.config.matrix_weight_decay,
                )
                for spec in specs
            }
            gradient_probes = {
                self._fragment_key(spec): self._regime_gradient_probe(
                    models[spec], spec, scaling, objective, device
                )
                for spec in specs
            }
            trajectories = {self._fragment_key(spec): [] for spec in specs}
            best: Dict[Tuple[str, str], Tuple[float, int, Dict[str, Any]]] = {}
            rng = np.random.default_rng(seed + 681000)
            digest = hashlib.sha256()
            progress = atomic._CompactProgress(
                f"06b-p adaptive 3x2 seed={seed}",
                self.config.fragment_training_steps,
                max(1, self.config.fragment_training_steps // 25),
            )
            for step in range(self.config.fragment_training_steps + 1):
                if step in self.config.fragment_checkpoints:
                    for spec, model in models.items():
                        metrics = self._fragment_metrics(
                            model,
                            spec,
                            scaling,
                            "calibration",
                            device,
                            calibration_rows,
                        )
                        key = self._fragment_key(spec)
                        trajectories[key].append(
                            {"step": step, "calibration": metrics}
                        )
                        score = metrics["endpoint_rmse_mv"]
                        if spec not in best or score < best[spec][0]:
                            best[spec] = (score, step, self._copy_state_dict(model))
                if step == self.config.fragment_training_steps:
                    break
                rows = rng.choice(
                    self._flat_count("fit"),
                    size=self.config.fragment_batch_transition_count,
                    replace=False,
                )
                digest.update(np.asarray(rows, dtype=np.int64).tobytes())
                batch = self._flat_tensors("fit", rows, device)
                losses = []
                for spec, model in models.items():
                    optimizer = optimizers[spec]
                    optimizer.zero_grad(set_to_none=True)
                    loss, _ = self._fragment_loss(
                        model, spec, scaling, objective, batch
                    )
                    if not bool(atomic.torch.isfinite(loss)):
                        raise RuntimeError(
                            f"non-finite adaptive loss seed={seed} step={step} "
                            f"arm={self._fragment_key(spec)}"
                        )
                    loss.backward()
                    atomic.torch.nn.utils.clip_grad_norm_(
                        model.parameters(), self.config.matrix_gradient_clip_norm
                    )
                    optimizer.step()
                    losses.append(float(loss.detach().cpu()))
                progress.update(
                    step + 1,
                    f"loss[min/median/max]={min(losses):.3g}/"
                    f"{float(np.median(losses)):.3g}/{max(losses):.3g}",
                )
            selected = {}
            for spec, model in models.items():
                score, selected_step, state = best[spec]
                model.load_state_dict(state)
                model.eval()
                key = self._fragment_key(spec)
                self.fragment_models[(key, seed)] = model
                path = self.output_dir / f"adaptive_fragment_{key.replace('|','__')}_seed{seed}.pt"
                atomic.torch.save(
                    {
                        "spec": spec,
                        "adaptive_choice": choice,
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
                "regime_gradient_probes_at_initialization": gradient_probes,
                "trajectories": trajectories,
                "selected": selected,
            }
        calibration_arm_scores = {
            self._fragment_key(spec): self._median(
                [
                    float(
                        seed["selected"][self._fragment_key(spec)][
                            "selected_calibration_endpoint_rmse_mv"
                        ]
                    )
                    for seed in reports.values()
                ]
            )
            for spec in specs
        }
        selected_fragment_arm = min(
            calibration_arm_scores, key=calibration_arm_scores.get
        )
        self.adaptive_choice["fragment_arm"] = selected_fragment_arm
        report = {
            "schema_version": "06b-p-adaptive-fragment-training-v1",
            "valid": all(
                len(rows) == len(self.config.fragment_checkpoints)
                for seed in reports.values()
                for rows in seed["trajectories"].values()
            ),
            "adaptive_choice_from_stage_1": choice,
            "selected_fragment_arm": selected_fragment_arm,
            "median_calibration_endpoint_rmse_by_arm": calibration_arm_scores,
            "factor_arm_count": len(specs),
            "same_fixed_width_numeric_tensor": True,
            "same_parameter_count_within_stage": True,
            "same_initialization_within_seed": True,
            "same_minibatch_stream_within_seed": True,
            "calibration_partition": "deterministic_odd_rows_half_B",
            "development_used_during_training": False,
            "reports": reports,
        }
        self.fragment_training_valid = bool(report["valid"])
        atomic._write_json(
            self.output_dir / "adaptive_fragment_training.json", report
        )
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

    def _recursive_fragment_metrics(
        self,
        model: Any,
        spec: Tuple[str, str],
        scaling: str,
        role: str,
        device: Any,
    ) -> Dict[str, Any]:
        values = self.window_data[role]
        state = atomic.torch.as_tensor(values["state_t"], device=device)
        voltage_t = atomic.torch.as_tensor(values["voltage_t"], device=device)
        target = atomic.torch.as_tensor(values["voltage_t1"], device=device)
        drive = atomic.torch.as_tensor(values["drive"], device=device)
        held = atomic.torch.as_tensor(values["held_ions"], device=device)
        current = voltage_t[:, 0]
        passive = voltage_t[:, 0]
        physical_baseline = voltage_t[:, 0]
        center = atomic.torch.as_tensor(
            self.statistics["state_center"], dtype=state.dtype, device=device
        )
        state_scale = atomic.torch.as_tensor(
            self.statistics["state_scale"], dtype=state.dtype, device=device
        )
        horizons = {}
        model.eval()
        with atomic.torch.no_grad():
            for step in range(target.shape[1]):
                logical_indices = values["indices"][:, step]
                raw_state = self.store.read_state(logical_indices, "t").astype(
                    np.float32
                )
                current_np = current.detach().cpu().numpy().astype(np.float32)
                exact = encode_realized_synaptic_drive(
                    self.store,
                    logical_indices,
                    current_np,
                    dt_ms=self.config.macro_step_ms,
                )
                complete = encode_realized_synaptic_drive(
                    self.store,
                    logical_indices,
                    current_np,
                    dt_ms=self.config.macro_step_ms,
                    raw_state_t=raw_state,
                )
                batch = {
                    "state": state[:, step],
                    "voltage": current,
                    "target_voltage": target[:, step],
                    "drive": drive[:, step],
                    "held_ions": held,
                    "structured_exact_events": atomic.torch.as_tensor(
                        exact["synaptic_features"], device=device
                    ),
                    "structured_boundary_complete": atomic.torch.as_tensor(
                        complete["synaptic_features"], device=device
                    ),
                    "synaptic_conductance_us": atomic.torch.as_tensor(
                        complete["synaptic_conductance_us"], device=device
                    ),
                    "synaptic_source_na": atomic.torch.as_tensor(
                        complete["synaptic_source_na"], device=device
                    ),
                    "somatic_current_na": atomic.torch.as_tensor(
                        complete["somatic_current_na"], device=device
                    ),
                }
                features = self._features(
                    (batch["state"] - center) / state_scale,
                    current,
                    atomic.torch.cat((batch["drive"], held), dim=-1),
                    "authentic",
                )
                features = atomic.torch.cat(
                    (features, self._structured_tensor(spec[0], batch)), dim=-1
                )
                region_ids = atomic.torch.as_tensor(
                    self.layout.segment_region_ids,
                    dtype=atomic.torch.long,
                    device=device,
                )
                hidden = atomic.torch.zeros(
                    current.shape[0],
                    self.layout.segment_count,
                    self.config.matrix_hidden_width,
                    dtype=current.dtype,
                    device=device,
                )
                standardized, _ = model(
                    features, region_ids, hidden, recurrent=False
                )
                decoded = standardized * self._target_scale_tensor(
                    spec[1], scaling, standardized
                )
                current = self._apply_physical_target(
                    decoded, current, spec[1], batch
                )
                zeros = atomic.torch.zeros_like(decoded)
                physical_batch = dict(batch)
                physical_batch["voltage"] = physical_baseline
                if spec[1] == INTRINSIC_RESIDUAL:
                    physical_complete = encode_realized_synaptic_drive(
                        self.store,
                        logical_indices,
                        physical_baseline.detach().cpu().numpy().astype(np.float32),
                        dt_ms=self.config.macro_step_ms,
                        raw_state_t=raw_state,
                    )
                    for name in (
                        "synaptic_conductance_us",
                        "synaptic_source_na",
                        "somatic_current_na",
                    ):
                        physical_batch[name] = atomic.torch.as_tensor(
                            physical_complete[name], device=device
                        )
                physical_baseline = self._apply_physical_target(
                    zeros, physical_baseline, spec[1], physical_batch
                )
                passive_batch = dict(batch)
                passive_batch["voltage"] = passive
                passive = self._apply_physical_target(
                    zeros, passive, NET_EFFECTIVE_SOURCE, passive_batch
                )
                horizon = step + 1
                if horizon in self.config.rollout_horizons_ms:
                    prediction_np = current.cpu().numpy()
                    physical_np = physical_baseline.cpu().numpy()
                    passive_np = passive.cpu().numpy()
                    target_np = target[:, step].cpu().numpy()
                    initial_np = voltage_t[:, 0].cpu().numpy()
                    activity = np.abs(target_np - initial_np)
                    masks = {
                        "quiescent_lt_1mV": activity < 1.0,
                        "moderate_1_to_5mV": (activity >= 1.0)
                        & (activity < 5.0),
                        "active_ge_5mV": activity >= 5.0,
                        "regenerative_ge_20mV": activity >= 20.0,
                    }
                    model_rmse = self._rmse(prediction_np, target_np)
                    physical_rmse = self._rmse(physical_np, target_np)
                    passive_rmse = self._rmse(passive_np, target_np)
                    horizons[f"{horizon}_ms"] = {
                        "endpoint_rmse_mv": model_rmse,
                        "physical_baseline_endpoint_rmse_mv": physical_rmse,
                        "passive_endpoint_rmse_mv": passive_rmse,
                        "endpoint_gain_over_physical_baseline_fraction": 1.0
                        - model_rmse / max(physical_rmse, 1e-12),
                        "endpoint_gain_over_passive_fraction": 1.0
                        - model_rmse / max(passive_rmse, 1e-12),
                        "physical_voltage_violation_count": int(
                            (
                                (
                                    prediction_np
                                    < self.config.physical_voltage_minimum_mv
                                )
                                | (
                                    prediction_np
                                    > self.config.physical_voltage_maximum_mv
                                )
                            ).sum()
                        ),
                        "nonfinite_voltage_count": int(
                            (~np.isfinite(prediction_np)).sum()
                        ),
                        "activity": self._masked_gain_against(
                            prediction_np, passive_np, target_np, masks
                        ),
                    }
        return {
            "horizons": horizons,
            "structured_inputs_recomputed_at_predicted_voltage": True,
            "teacher_mechanism_state_retained": True,
            "teacher_endpoint_used_as_input": False,
        }

    def _fragment_voltage_sensitivity(
        self,
        model: Any,
        spec: Tuple[str, str],
        scaling: str,
        device: Any,
    ) -> Dict[str, Any]:
        rows = np.arange(min(16, self._flat_count("development")), dtype=np.int64)
        base = self._flat_tensors("development", rows, device)
        direction = atomic.torch.sign(base["target_voltage"] - base["voltage"])
        direction = atomic.torch.where(
            direction == 0.0, atomic.torch.ones_like(direction), direction
        )
        report = {}
        model.eval()
        with atomic.torch.no_grad():
            for magnitude in self.config.voltage_sensitivity_perturbations_mv:
                predictions = []
                for sign in (-1.0, 1.0):
                    batch = dict(base)
                    batch["voltage"] = base["voltage"] + sign * magnitude * direction
                    predictions.append(
                        self._fragment_forward(model, spec, scaling, batch)[3]
                    )
                derivative = (predictions[1] - predictions[0]) / (2.0 * magnitude)
                report[str(magnitude)] = {
                    "rms_directional_gain": float(
                        atomic.torch.sqrt(atomic.torch.mean(derivative**2)).cpu()
                    ),
                    "maximum_absolute_directional_gain": float(
                        atomic.torch.max(atomic.torch.abs(derivative)).cpu()
                    ),
                    "nonfinite_count": int(
                        (~np.isfinite(derivative.cpu().numpy())).sum()
                    ),
                }
        return report

    def evaluate_atomic_and_recursive_boundaries(self) -> Dict[str, Any]:
        if not self.atomic_training_valid:
            raise RuntimeError("06b-p atomic training is incomplete")
        if not self.fragment_training_valid:
            raise RuntimeError("06b-p adaptive fragment training is incomplete")
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
        fragment_per_seed = {}
        scaling = self.adaptive_choice["scaling"]
        total_fragment = len(self._fragment_specs()) * len(self.config.pilot_seeds)
        fragment_progress = atomic._CompactProgress(
            "06b-p frozen adaptive evaluation",
            total_fragment,
            max(1, total_fragment // 12),
        )
        completed = 0
        for seed in self.config.pilot_seeds:
            rows = {}
            for spec in self._fragment_specs():
                key = self._fragment_key(spec)
                model = self.fragment_models[(key, seed)]
                rows[key] = {
                    "teacher_boundary_one_step": self._fragment_metrics(
                        model, spec, scaling, "development", device
                    ),
                    "recursive_voltage_teacher_STATE": self._recursive_fragment_metrics(
                        model, spec, scaling, "development", device
                    ),
                    "voltage_sensitivity": self._fragment_voltage_sensitivity(
                        model, spec, scaling, device
                    ),
                }
                completed += 1
                fragment_progress.update(completed, f"seed={seed} {key}")
            fragment_per_seed[str(seed)] = rows
        report = {
            "schema_version": "06b-p-boundary-evaluation-v1",
            "valid": all(
                row["teacher_boundary_one_step"]["nonfinite_count"] == 0
                and row["recursive_voltage_teacher_STATE"]["horizons"]["8_ms"]["nonfinite_voltage_count"] == 0
                for seed in per_seed.values()
                for row in seed.values()
            )
            and all(
                row["teacher_boundary_one_step"]["nonfinite_count"] == 0
                and row["recursive_voltage_teacher_STATE"]["horizons"]["8_ms"][
                    "nonfinite_voltage_count"
                ]
                == 0
                for seed in fragment_per_seed.values()
                for row in seed.values()
            ),
            "role": "historically_reused_train_development",
            "teacher_boundary_used_only_for_atomic_evaluation": True,
            "recursive_boundary_uses_teacher_STATE": True,
            "models_retrained_for_boundary_test": False,
            "validation_state_accessed": False,
            "test_state_accessed": False,
            "per_seed": per_seed,
            "adaptive_fragment_per_seed": fragment_per_seed,
            "adaptive_choice": dict(self.adaptive_choice),
            "structured_inputs_recomputed_from_predicted_voltage": True,
            "directional_voltage_sensitivity_probed": True,
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

    def _fragment_summary(
        self, key: str, evaluation: Mapping[str, Any]
    ) -> Dict[str, Any]:
        rows = [
            seed[key] for seed in evaluation["adaptive_fragment_per_seed"].values()
        ]
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
            "median_atomic_gain_over_physical_baseline_fraction": self._median(
                [
                    row["endpoint_gain_over_physical_baseline_fraction"]
                    for row in atomic_rows
                ]
            ),
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
            "median_recursive_8ms_gain_over_physical_baseline_fraction": self._median(
                [
                    row["endpoint_gain_over_physical_baseline_fraction"]
                    for row in recursive_rows
                ]
            ),
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
            "voltage_sensitivity": {
                str(magnitude): {
                    "median_rms_directional_gain": self._median(
                        [
                            row["voltage_sensitivity"][str(magnitude)][
                                "rms_directional_gain"
                            ]
                            for row in rows
                        ]
                    ),
                    "median_maximum_absolute_directional_gain": self._median(
                        [
                            row["voltage_sensitivity"][str(magnitude)][
                                "maximum_absolute_directional_gain"
                            ]
                            for row in rows
                        ]
                    ),
                }
                for magnitude in self.config.voltage_sensitivity_perturbations_mv
            },
        }

    def _fragment_main_effect(
        self,
        summaries: Mapping[str, Any],
        axis: int,
        positive: str,
        negative: str,
        metric: str,
    ) -> float:
        contrasts = []
        for spec in self._fragment_specs():
            if spec[axis] != positive:
                continue
            reference = list(spec)
            reference[axis] = negative
            positive_value = summaries[self._fragment_key(spec)][metric]
            negative_value = summaries[self._fragment_key(tuple(reference))][metric]
            contrasts.append(1.0 - positive_value / max(negative_value, 1e-12))
        return self._median(contrasts)

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
        fragment_training: Mapping[str, Any],
        evaluation: Mapping[str, Any],
    ) -> Dict[str, Any]:
        summaries = {
            self._atomic_key(spec): self._summary(self._atomic_key(spec), evaluation)
            for spec in self._atomic_specs()
        }
        scaling = self._scaling_report(training)
        fragment_summaries = {
            self._fragment_key(spec): self._fragment_summary(
                self._fragment_key(spec), evaluation
            )
            for spec in self._fragment_specs()
        }
        best_observed_fragment_key = min(
            fragment_summaries,
            key=lambda key: fragment_summaries[key][
                "median_atomic_endpoint_rmse_mv"
            ],
        )
        best_fragment_key = str(fragment_training["selected_fragment_arm"])
        best_fragment = fragment_summaries[best_fragment_key]
        best_key = min(
            summaries,
            key=lambda key: summaries[key]["median_atomic_endpoint_rmse_mv"],
        )
        best = summaries[best_key]
        activity = best_fragment["recursive_activity_gain_over_passive"]
        activity_safe = all(
            activity.get(name) is not None and activity[name] >= 0.0
            for name in ("quiescent_lt_1mV", "moderate_1_to_5mV", "active_ge_5mV")
        )
        fragment_selected_steps = [
            int(seed["selected"][best_fragment_key]["selected_step"])
            for seed in fragment_training["reports"].values()
        ]
        trained = (
            float(np.mean(np.asarray(fragment_selected_steps) > 0))
            >= self.config.minimum_trained_selection_fraction
        )
        atomic_signal = (
            best_fragment["median_atomic_gain_over_passive_fraction"]
            >= self.config.minimum_atomic_gain_over_passive_fraction
            and trained
        )
        recursive_signal = (
            best_fragment["median_recursive_8ms_gain_over_passive_fraction"]
            >= self.config.minimum_recursive_gain_over_passive_fraction
        )
        safe = bool(
            atomic_signal
            and recursive_signal
            and activity_safe
            and best_fragment["physical_voltage_violation_count"] == 0
        )
        exact_event_effect = self._fragment_main_effect(
            fragment_summaries,
            0,
            EXACT_EVENTS,
            COMPACT_MOMENTS,
            "median_atomic_endpoint_rmse_mv",
        )
        boundary_state_effect = self._fragment_main_effect(
            fragment_summaries,
            0,
            BOUNDARY_COMPLETE,
            EXACT_EVENTS,
            "median_atomic_endpoint_rmse_mv",
        )
        intrinsic_target_effect = self._fragment_main_effect(
            fragment_summaries,
            1,
            INTRINSIC_RESIDUAL,
            NET_EFFECTIVE_SOURCE,
            "median_atomic_endpoint_rmse_mv",
        )
        materiality = self.config.source_materiality_fraction
        if safe:
            diagnosis = "CAUSAL_SOURCE_FRAGMENT_SURVIVES_RECURSIVE_BOUNDARY"
            next_step = "independent_train_support_fragment_confirmation"
        elif atomic_signal and not recursive_signal:
            diagnosis = "SOURCE_LEARNABLE_ON_TEACHER_BOUNDARY_BUT_FAILS_RECURSIVE_BOUNDARY"
            next_step = "causal_recursive_boundary_exposure_or_scheduled_sampling"
        elif atomic_signal:
            diagnosis = "SOURCE_LEARNABLE_BUT_REGIME_SAFETY_FAILS"
            next_step = "source_regime_decomposition_with_explicit_quiet_residual"
        elif boundary_state_effect >= materiality:
            diagnosis = "BOUNDARY_SYNAPSE_MEMORY_IS_CAUSALLY_MATERIAL_BUT_INSUFFICIENT"
            next_step = "joint_synapse_state_and_intrinsic_source_atomic_playground"
        elif exact_event_effect >= materiality:
            diagnosis = "LOSSY_EVENT_AGGREGATION_IS_CAUSALLY_MATERIAL_BUT_INSUFFICIENT"
            next_step = "ordered_event_operator_atomic_playground"
        elif intrinsic_target_effect >= materiality:
            diagnosis = "NET_SOURCE_CANCELLATION_IS_CAUSALLY_MATERIAL_BUT_INSUFFICIENT"
            next_step = "mechanism_factored_intrinsic_current_operator"
        else:
            diagnosis = "SOURCE_FAILURE_SURVIVES_SCALE_INPUT_AND_TARGET_REPAIRS"
            next_step = "mechanism_factored_current_operator_or_new_intermediate_state_data"
        report = {
            "schema_version": "06b-p-final-report-v2",
            "valid": bool(
                contract.get("valid")
                and training.get("valid")
                and fragment_training.get("valid")
                and evaluation.get("valid")
            ),
            "component_playground_grade": True,
            "diagnosis": diagnosis,
            "best_stage_1_arm": best_key,
            "adaptive_choice": dict(self.adaptive_choice),
            "best_observed_development_arm": best_observed_fragment_key,
            "best_observed_arm": best_fragment_key,
            "selected_calibration_fragment_arm": best_fragment_key,
            "selected_candidate": best_fragment_key if safe else None,
            "best_observed_arm_metrics": best_fragment,
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
            "adaptive_fragment_main_effects": {
                "exact_events_over_compact": exact_event_effect,
                "boundary_complete_over_exact_events": boundary_state_effect,
                "intrinsic_residual_over_net_effective_source": intrinsic_target_effect,
            },
            "causal_materiality": {
                "lossy_event_aggregation": exact_event_effect >= materiality,
                "missing_boundary_synapse_state": boundary_state_effect >= materiality,
                "net_source_cancellation": intrinsic_target_effect >= materiality,
            },
            "checkpoint_scaling": scaling,
            "best_fragment_selected_steps": fragment_selected_steps,
            "atomic_source_signal": atomic_signal,
            "recursive_boundary_signal": recursive_signal,
            "regime_safety_passed": activity_safe,
            "summaries": summaries,
            "adaptive_fragment_summaries": fragment_summaries,
            "source_contract_audit": contract.get("source_contract_audit"),
            "substep_source_support_audit": contract.get(
                "substep_source_support_audit"
            ),
            "teacher_source_oracle_selectable": False,
            "teacher_endpoint_used_as_model_input": False,
            "predicted_state_trained": False,
            "temporal_memory_trained": False,
            "new_independent_confirmation_claimed": False,
            "development_used_for_checkpoint_or_arm_selection": False,
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
    "COMPACT_MOMENTS",
    "EXACT_EVENTS",
    "BOUNDARY_COMPLETE",
    "NET_EFFECTIVE_SOURCE",
    "INTRINSIC_RESIDUAL",
    "INPUT_CONTRACTS",
    "PHYSICAL_TARGETS",
    "AtomicEffectiveSourceConfig",
    "AtomicEffectiveSourceLearnability",
    "verified_06bo_artifact_root",
]
