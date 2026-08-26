"""06b-r: causal event utilization x recursive exposure x local stability.

The experiment is deliberately train-only.  It keeps the passive Hines solve
fixed, reuses the 06b-q ordered event contract, and tests three orthogonal
interventions in a paired 2x2x2 matrix.  Development data are opened only
after checkpoint and arm selection on disjoint calibration halves.
"""

from __future__ import annotations

import hashlib
import itertools
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np

from ..hayflow_data.hines_inputs import encode_realized_synaptic_drive
from . import atomic_state_dynamics_playground as atomic
from .atomic_effective_source_learnability import HYBRID, NET_EFFECTIVE_SOURCE
from .event_supported_jump_playground import (
    CHRONOLOGICAL_JUMP,
    EVENT_FEATURE_NAMES,
    NONZERO_ROBUST_LOG,
    UNGATED_RESIDUAL,
    EventEncoderBank,
    EventSupportedJumpConfig,
    EventSupportedJumpPlayground,
    normalize_event_tensor,
)


EXPECTED_06BQ_ARCHIVE_SHA256 = (
    "5f2a8ddd0402fcd556138f9838e330e97fc0f1b2e8dc8c8a2d330a53b8bd117e"
)
EXPECTED_06BQ_INDEX_SHA256 = (
    "c39fa24b2b96cfba29ab348fb139c7f3156decef48d09b53d040ce2e384b9a08"
)
EXPECTED_06BQ_FINAL_SHA256 = (
    "f3d4614c46c95918ec1ac6f5afce9e4242e35a92ef183406c20bc56789e64c17"
)

AUXILIARY_OFF = "auxiliary_off"
CAUSAL_SYNAPTIC_AUXILIARY = "causal_synaptic_auxiliary"
AUXILIARY_MODES = (AUXILIARY_OFF, CAUSAL_SYNAPTIC_AUXILIARY)

TEACHER_BOUNDARY = "teacher_boundary"
PUSHFORWARD_4MS = "pushforward_4ms"
EXPOSURE_MODES = (TEACHER_BOUNDARY, PUSHFORWARD_4MS)

STABILITY_OFF = "stability_off"
PASSIVE_RELATIVE_DIRECTIONAL = "passive_relative_directional"
STABILITY_MODES = (STABILITY_OFF, PASSIVE_RELATIVE_DIRECTIONAL)

CAUSAL_TARGET = "causal_target"
PERMUTED_TARGET = "permuted_target"
AUXILIARY_PROBE_ARMS = (CAUSAL_TARGET, PERMUTED_TARGET)

ORDERED_EVENT_PATH_DELETION = "ordered_event_path_deletion"
ALL_CURRENT_U_DELETION = "all_current_U_deletion"
TIMESTAMP_REVERSAL = "timestamp_reversal"
RECEPTOR_PERMUTATION = "receptor_permutation"
CAUSAL_CONTROLS = (
    ORDERED_EVENT_PATH_DELETION,
    ALL_CURRENT_U_DELETION,
    TIMESTAMP_REVERSAL,
    RECEPTOR_PERMUTATION,
)


def verified_06bq_artifact_root(
    source: Path, cache_dir: Path
) -> Tuple[Path, Dict[str, Any], Dict[str, Any]]:
    """Verify the registered 06b-q archive and every indexed member."""

    source = Path(source).expanduser().resolve()
    cache_dir = Path(cache_dir).resolve()
    if not source.exists():
        raise FileNotFoundError(source)
    archive_hash = "extracted-directory"
    if source.is_file():
        if source.suffix.lower() != ".zip":
            raise RuntimeError("06b-q source must be a ZIP or extracted directory")
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
        if atomic._sha256_file(path) == EXPECTED_06BQ_INDEX_SHA256
    ]
    if len(roots) != 1:
        raise RuntimeError(f"expected one exact 06b-q artifact; found {len(roots)}")
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
        raise RuntimeError(f"06b-q indexed member verification failed: {failures}")
    final_path = root / "final_report.json"
    if atomic._sha256_file(final_path) != EXPECTED_06BQ_FINAL_SHA256:
        raise RuntimeError("06b-q final report SHA-256 mismatch")
    final = json.loads(final_path.read_text(encoding="utf-8"))
    if (
        final.get("valid") is not True
        or final.get("selected_candidate") is not None
        or final.get("validation_state_accessed") is not False
        or final.get("test_state_accessed") is not False
    ):
        raise RuntimeError("06b-q result does not authorize 06b-r")
    if source.is_file() and archive_hash != EXPECTED_06BQ_ARCHIVE_SHA256:
        archive_hash = "kaggle-repacked"
    return root, final, {
        "valid": True,
        "archive_sha256": archive_hash,
        "artifact_index_sha256": EXPECTED_06BQ_INDEX_SHA256,
        "final_report_sha256": EXPECTED_06BQ_FINAL_SHA256,
        "indexed_member_count": len(index.get("artifacts", [])),
        "reported_diagnosis": final.get("diagnosis"),
    }


@dataclass(frozen=True)
class RecursiveEventExposureConfig(EventSupportedJumpConfig):
    auxiliary_modes: Tuple[str, ...] = AUXILIARY_MODES
    exposure_modes: Tuple[str, ...] = EXPOSURE_MODES
    stability_modes: Tuple[str, ...] = STABILITY_MODES
    auxiliary_target_names: Tuple[str, ...] = (
        "synaptic_conductance_us",
        "synaptic_source_na",
        "somatic_current_na",
    )
    auxiliary_scale_quantile: float = 0.99
    auxiliary_scale_floor: float = 1e-6
    auxiliary_probe_steps: int = 120
    auxiliary_probe_checkpoints: Tuple[int, ...] = (0, 20, 60, 120)
    auxiliary_probe_batch_window_count: int = 8
    recursive_training_steps: int = 240
    recursive_checkpoints: Tuple[int, ...] = (0, 40, 120, 240)
    recursive_training_horizon_ms: int = 4
    recursive_batch_window_count: int = 4
    auxiliary_loss_weight: float = 0.25
    directional_stability_weight: float = 0.05
    directional_perturbation_mv: float = 0.25
    passive_relative_margin: float = 1.05
    minimum_auxiliary_gain_over_permuted_fraction: float = 0.10
    minimum_candidate_gain_over_passive_fraction: float = 0.02
    maximum_per_seed_regression_fraction: float = 0.0
    minimum_event_materiality_fraction: float = 0.02

    def validate(self) -> None:
        super().validate()
        if tuple(self.auxiliary_modes) != AUXILIARY_MODES:
            raise ValueError("06b-r auxiliary factor changed")
        if tuple(self.exposure_modes) != EXPOSURE_MODES:
            raise ValueError("06b-r exposure factor changed")
        if tuple(self.stability_modes) != STABILITY_MODES:
            raise ValueError("06b-r stability factor changed")
        if self.recursive_training_horizon_ms != 4:
            raise ValueError("06b-r pushforward horizon must remain four milliseconds")
        if (
            self.recursive_checkpoints[0] != 0
            or self.recursive_checkpoints[-1] != self.recursive_training_steps
        ):
            raise ValueError("recursive checkpoints must span the full budget")
        if (
            self.auxiliary_probe_checkpoints[0] != 0
            or self.auxiliary_probe_checkpoints[-1] != self.auxiliary_probe_steps
        ):
            raise ValueError("auxiliary checkpoints must span the full budget")

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "RecursiveEventExposureConfig":
        payload = dict(values)
        tuple_names = {
            "pilot_seeds", "output_parameterizations", "state_feedback_contracts",
            "temporal_contracts", "matrix_checkpoints", "rollout_horizons_ms",
            "scaling_modes", "objectives", "atomic_checkpoints", "input_contracts",
            "physical_targets", "fragment_checkpoints", "substep_audit_dt_ms",
            "voltage_sensitivity_perturbations_mv", "event_representations",
            "event_normalizations", "safety_gates", "support_fit_components",
            "support_calibration_components", "support_development_components",
            "jump_checkpoints", "gradient_probe_steps", "safety_checkpoints",
            "synthetic_checkpoints", "auxiliary_modes", "exposure_modes",
            "stability_modes", "auxiliary_target_names",
            "auxiliary_probe_checkpoints", "recursive_checkpoints",
        }
        for name in tuple_names:
            if name in payload:
                payload[name] = tuple(payload[name])
        result = cls(**payload)
        result.validate()
        return result


if atomic.nn is not None:

    class RecursiveEventSourceCell(atomic.nn.Module):
        """Parameter-matched source cell with an always-present auxiliary head."""

        def __init__(
            self,
            *,
            base_feature_width: int,
            event_embedding_width: int,
            region_count: int,
            region_width: int,
            hidden_width: int,
            output_limit: float,
            auxiliary_width: int,
        ) -> None:
            super().__init__()
            from .effective_membrane_source_playground import CausalMembraneSourceCell

            self.events = EventEncoderBank(len(EVENT_FEATURE_NAMES), event_embedding_width)
            self.source = CausalMembraneSourceCell(
                base_feature_width + event_embedding_width,
                region_count,
                region_width,
                hidden_width,
                output_limit,
            )
            self.auxiliary = atomic.nn.Sequential(
                atomic.nn.Linear(event_embedding_width, hidden_width),
                atomic.nn.SiLU(),
                atomic.nn.Linear(hidden_width, auxiliary_width),
            )

        def forward(
            self,
            base: Any,
            event_values: Any,
            event_mask: Any,
            region_ids: Any,
        ) -> Tuple[Any, Any, Any]:
            event_embedding = self.events(
                event_values, event_mask, CHRONOLOGICAL_JUMP
            )
            hidden = atomic.torch.zeros(
                base.shape[0],
                base.shape[1],
                self.source.hidden_width,
                dtype=base.dtype,
                device=base.device,
            )
            output, next_hidden = self.source(
                atomic.torch.cat((base, event_embedding), dim=-1),
                region_ids,
                hidden,
                recurrent=False,
            )
            return output, next_hidden, self.auxiliary(event_embedding)


else:  # pragma: no cover

    class RecursiveEventSourceCell:
        def __init__(self, *_: Any, **__: Any) -> None:
            raise RuntimeError("06b-r requires PyTorch")


class RecursiveEventExposurePlayground(EventSupportedJumpPlayground):
    """Run the atomic auxiliary probe and paired recursive 2x2x2 matrix."""

    config: RecursiveEventExposureConfig

    def __init__(
        self,
        bundle: Any,
        output_dir: Path,
        config: RecursiveEventExposureConfig,
        artifact_05t_source: Path,
        artifact_06bn_source: Path,
        artifact_06bo_source: Path,
        artifact_06bp_source: Path,
        artifact_06bq_source: Path,
        *,
        code_revision: str,
    ) -> None:
        super().__init__(
            bundle,
            output_dir,
            config,
            artifact_05t_source,
            artifact_06bn_source,
            artifact_06bo_source,
            artifact_06bp_source,
            code_revision=code_revision,
        )
        self.artifact_06bq_source = Path(artifact_06bq_source)
        self.source_06bq_root: Optional[Path] = None
        self.auxiliary_scale = np.empty(0, dtype=np.float32)
        self.recursive_models: Dict[Tuple[str, int], Any] = {}

    @staticmethod
    def _spec_key(spec: Tuple[str, str, str]) -> str:
        return "|".join(spec)

    def _specs(self) -> Tuple[Tuple[str, str, str], ...]:
        return tuple(
            itertools.product(
                self.config.auxiliary_modes,
                self.config.exposure_modes,
                self.config.stability_modes,
            )
        )

    def _new_recursive_model(self, seed: int, device: Any) -> Any:
        atomic.torch.manual_seed(seed + 706000)
        return RecursiveEventSourceCell(
            base_feature_width=self._feature_width(),
            event_embedding_width=self.config.event_embedding_width,
            region_count=len(self.layout.region_names),
            region_width=self.config.matrix_region_embedding_width,
            hidden_width=self.config.matrix_hidden_width,
            output_limit=self.config.normalized_output_limit,
            auxiliary_width=len(self.config.auxiliary_target_names),
        ).to(device)

    def _materialize_causal_auxiliary_targets(self) -> Dict[str, Any]:
        fit_values = None
        reports: Dict[str, Any] = {}
        for role, values in self.window_data.items():
            flat = values["indices"].reshape(-1)
            voltage = values["voltage_t"].reshape(-1, self.layout.segment_count)
            raw_state = self.store.read_state(flat, "t").astype(np.float32)
            encoded = encode_realized_synaptic_drive(
                self.store,
                flat,
                voltage,
                dt_ms=self.config.macro_step_ms,
                raw_state_t=raw_state,
            )
            target = np.stack(
                [encoded[name] for name in self.config.auxiliary_target_names],
                axis=-1,
            ).astype(np.float32)
            shape = values["voltage_t"].shape[:2]
            values["causal_auxiliary_raw"] = target.reshape(
                *shape, self.layout.segment_count, -1
            )
            if role == "fit":
                fit_values = target
            reports[role] = {
                "transition_count": int(len(flat)),
                "nonzero_fraction": float(np.mean(target != 0.0)),
                "rms": float(np.sqrt(np.mean(target.astype(np.float64) ** 2))),
            }
        if fit_values is None:
            raise RuntimeError("06b-r fit auxiliary target was not materialized")
        scales = []
        for channel in range(fit_values.shape[-1]):
            nonzero = np.abs(fit_values[..., channel])
            nonzero = nonzero[nonzero > 0.0]
            value = (
                float(np.quantile(nonzero, self.config.auxiliary_scale_quantile))
                if len(nonzero)
                else self.config.auxiliary_scale_floor
            )
            scales.append(max(value, self.config.auxiliary_scale_floor))
        self.auxiliary_scale = np.asarray(scales, dtype=np.float32)
        for values in self.window_data.values():
            values["causal_auxiliary_target"] = np.arcsinh(
                values["causal_auxiliary_raw"]
                / self.auxiliary_scale.reshape(1, 1, 1, -1)
            ).astype(np.float32)
        return {
            "valid": bool(np.isfinite(self.auxiliary_scale).all()),
            "target_names": list(self.config.auxiliary_target_names),
            "target_is_function_of": ["S_t", "U_realized"],
            "teacher_endpoint_used": False,
            "normalization": "fit-only per-channel asinh after nonzero p99 scaling",
            "scale": self.auxiliary_scale.tolist(),
            "roles": reports,
        }

    def prepare_recursive_event_exposure_playground(self) -> Dict[str, Any]:
        base = self.prepare_event_supported_jump_playground()
        root, prior, source = verified_06bq_artifact_root(
            self.artifact_06bq_source,
            self.output_dir.parent / ".06br_artifact_cache" / "06bq",
        )
        self.source_06bq_root = root
        auxiliary = self._materialize_causal_auxiliary_targets()
        device = atomic.torch.device("cpu")
        parameter_counts = {
            self._spec_key(spec): sum(
                parameter.numel()
                for parameter in self._new_recursive_model(
                    self.config.pilot_seeds[0], device
                ).parameters()
            )
            for spec in self._specs()
        }
        blockers = []
        if len(set(parameter_counts.values())) != 1:
            blockers.append("factorial arms are not parameter matched")
        if not auxiliary["valid"]:
            blockers.append("causal auxiliary targets are nonfinite")
        report = {
            **base,
            "schema_version": "06b-r-contract-v1",
            "experiment": "recursive_event_exposure_playground",
            "valid": not blockers,
            "blockers": blockers,
            "source_06bq": source,
            "prior_formal_diagnosis": prior.get("diagnosis"),
            "fixed_event_representation": CHRONOLOGICAL_JUMP,
            "fixed_event_normalization": NONZERO_ROBUST_LOG,
            "causal_auxiliary_contract": auxiliary,
            "factorial_axes": {
                "causal_event_auxiliary": list(self.config.auxiliary_modes),
                "boundary_exposure": list(self.config.exposure_modes),
                "local_stability": list(self.config.stability_modes),
            },
            "factorial_arm_count": len(self._specs()),
            "parameter_counts": parameter_counts,
            "passive_hines_prior_fixed": True,
            "teacher_mechanism_state_during_rollout": True,
            "teacher_mechanism_state_is_deployment_ready": False,
            "pushforward_previous_prediction_detached": True,
            "validation_state_accessed": False,
            "test_state_accessed": False,
            "fresh_test_accessed": False,
            "teacher_endpoint_used_as_model_input": False,
            "configuration": asdict(self.config),
        }
        atomic._write_json(
            self.output_dir / "recursive_event_exposure_contract.json", report
        )
        if blockers:
            raise RuntimeError(f"06b-r preflight failed: {blockers}")
        return report

    def _window_batch(
        self, role: str, windows: np.ndarray, step: int, current: Any, device: Any
    ) -> Dict[str, Any]:
        values = self.window_data[role]
        windows = np.asarray(windows, dtype=np.int64)
        payload = {
            "state": values["state_t"][windows, step],
            "voltage": current,
            "target_voltage": values["voltage_t1"][windows, step],
            "drive": values["drive"][windows, step],
            "held_ions": values["held_ions"][windows],
            "ordered_events": values["ordered_events"][windows, step],
            "ordered_event_mask": values["ordered_event_mask"][windows, step],
            "causal_auxiliary_target": values["causal_auxiliary_target"][windows, step],
        }
        return {
            name: value
            if atomic.torch.is_tensor(value)
            else atomic.torch.as_tensor(value, device=device)
            for name, value in payload.items()
        }

    def _forward_recursive(
        self, model: Any, batch: Mapping[str, Any]
    ) -> Tuple[Any, Any, Any, Any]:
        center = atomic.torch.as_tensor(
            self.statistics["state_center"],
            dtype=batch["state"].dtype,
            device=batch["state"].device,
        )
        state_scale = atomic.torch.as_tensor(
            self.statistics["state_scale"],
            dtype=batch["state"].dtype,
            device=batch["state"].device,
        )
        context = atomic.torch.cat((batch["drive"], batch["held_ions"]), dim=-1)
        base = self._features(
            (batch["state"] - center) / state_scale,
            batch["voltage"],
            context,
            "authentic",
        )
        event_scale = atomic.torch.as_tensor(
            self.event_scales[NONZERO_ROBUST_LOG],
            dtype=batch["ordered_events"].dtype,
            device=batch["ordered_events"].device,
        )
        events = normalize_event_tensor(
            batch["ordered_events"],
            batch["ordered_event_mask"],
            event_scale,
            NONZERO_ROBUST_LOG,
            clip=self.config.event_normalized_clip,
        )
        region_ids = atomic.torch.as_tensor(
            self.layout.segment_region_ids,
            dtype=atomic.torch.long,
            device=batch["voltage"].device,
        )
        standardized, hidden, auxiliary = model(
            base, events, batch["ordered_event_mask"], region_ids
        )
        decoded = standardized * self._scale_tensor(
            self.selected_source_scaling, standardized
        )
        prediction = self._apply_physical_target(
            decoded, batch["voltage"], NET_EFFECTIVE_SOURCE, batch
        )
        return decoded, hidden, auxiliary, prediction

    @staticmethod
    def _rmse_tensor(left: Any, right: Any) -> Any:
        return atomic.torch.sqrt(atomic.torch.mean((left - right) ** 2))

    def _auxiliary_metrics(
        self, model: Any, role: str, rows: np.ndarray, device: Any
    ) -> Dict[str, float]:
        values = self.window_data[role]
        horizon = values["voltage_t"].shape[1]
        rows = np.asarray(rows, dtype=np.int64)
        windows, steps = rows // horizon, rows % horizon
        predictions = []
        targets = []
        model.eval()
        with atomic.torch.no_grad():
            for window, step in zip(windows, steps):
                current = atomic.torch.as_tensor(
                    values["voltage_t"][[window], step], device=device
                )
                batch = self._window_batch(
                    role, np.asarray([window]), int(step), current, device
                )
                _, _, auxiliary, _ = self._forward_recursive(model, batch)
                predictions.append(auxiliary)
                targets.append(batch["causal_auxiliary_target"])
        prediction = atomic.torch.cat(predictions).cpu().numpy()
        target = atomic.torch.cat(targets).cpu().numpy()
        rmse = float(np.sqrt(np.mean((prediction - target) ** 2)))
        variance = float(np.var(target))
        return {
            "normalized_rmse": rmse,
            "r2": float(1.0 - np.mean((prediction - target) ** 2) / max(variance, 1e-12)),
        }

    def run_causal_auxiliary_probe(self) -> Dict[str, Any]:
        """Known-answer target versus deliberately permuted negative control."""

        device = atomic.torch.device("cuda" if atomic.torch.cuda.is_available() else "cpu")
        fit_count = self._flat_count("fit")
        calibration_rows = np.arange(self._flat_count("calibration"), dtype=np.int64)[::2]
        reports: Dict[str, Any] = {}
        for seed in self.config.pilot_seeds:
            models = {
                arm: self._new_recursive_model(seed, device)
                for arm in AUXILIARY_PROBE_ARMS
            }
            optimizers = {
                arm: atomic.torch.optim.AdamW(
                    model.parameters(),
                    lr=self.config.matrix_learning_rate,
                    weight_decay=self.config.matrix_weight_decay,
                )
                for arm, model in models.items()
            }
            rng = np.random.default_rng(seed + 707000)
            histories = {arm: [] for arm in AUXILIARY_PROBE_ARMS}
            progress = atomic._CompactProgress(
                f"06b-r auxiliary probe seed={seed}",
                self.config.auxiliary_probe_steps,
                max(1, self.config.auxiliary_probe_steps // 6),
            )
            for step in range(self.config.auxiliary_probe_steps + 1):
                if step in self.config.auxiliary_probe_checkpoints:
                    for arm, model in models.items():
                        histories[arm].append(
                            {
                                "step": step,
                                "calibration_half_A": self._auxiliary_metrics(
                                    model, "calibration", calibration_rows, device
                                ),
                            }
                        )
                if step == self.config.auxiliary_probe_steps:
                    break
                rows = rng.choice(
                    fit_count,
                    self.config.auxiliary_probe_batch_window_count,
                    replace=False,
                )
                values = self.window_data["fit"]
                horizon = values["voltage_t"].shape[1]
                windows, steps = rows // horizon, rows % horizon
                losses = []
                for arm, model in models.items():
                    optimizer = optimizers[arm]
                    optimizer.zero_grad(set_to_none=True)
                    arm_losses = []
                    for window, local_step in zip(windows, steps):
                        current = atomic.torch.as_tensor(
                            values["voltage_t"][[window], local_step], device=device
                        )
                        batch = self._window_batch(
                            "fit",
                            np.asarray([window]),
                            int(local_step),
                            current,
                            device,
                        )
                        _, _, auxiliary, _ = self._forward_recursive(model, batch)
                        target = batch["causal_auxiliary_target"]
                        if arm == PERMUTED_TARGET:
                            target = atomic.torch.roll(target, shifts=1, dims=1)
                            target = atomic.torch.roll(target, shifts=1, dims=2)
                        arm_losses.append(
                            atomic.torch_functional.smooth_l1_loss(auxiliary, target)
                        )
                    loss = atomic.torch.stack(arm_losses).mean()
                    loss.backward()
                    atomic.torch.nn.utils.clip_grad_norm_(
                        model.parameters(), self.config.matrix_gradient_clip_norm
                    )
                    optimizer.step()
                    losses.append(float(loss.detach().cpu()))
                progress.update(step + 1, f"loss={float(np.median(losses)):.3g}")
            reports[str(seed)] = {"histories": histories}
        causal = self._median(
            row["histories"][CAUSAL_TARGET][-1]["calibration_half_A"]["normalized_rmse"]
            for row in reports.values()
        )
        permuted = self._median(
            row["histories"][PERMUTED_TARGET][-1]["calibration_half_A"]["normalized_rmse"]
            for row in reports.values()
        )
        gain = 1.0 - causal / max(permuted, 1e-12)
        report = {
            "schema_version": "06b-r-auxiliary-probe-v1",
            "valid": bool(np.isfinite([causal, permuted, gain]).all()),
            "selection_eligible": False,
            "known_causal_target": True,
            "teacher_endpoint_used": False,
            "same_initialization_within_seed": True,
            "same_minibatch_stream_within_seed": True,
            "causal_target_median_normalized_rmse": causal,
            "permuted_target_median_normalized_rmse": permuted,
            "gain_over_permuted_fraction": gain,
            "registered_gate_passed": bool(
                gain >= self.config.minimum_auxiliary_gain_over_permuted_fraction
            ),
            "reports": reports,
        }
        atomic._write_json(self.output_dir / "causal_auxiliary_probe.json", report)
        return report

    def _apply_control(self, batch: Dict[str, Any], control: Optional[str]) -> None:
        if control is None:
            return
        if control in (ORDERED_EVENT_PATH_DELETION, ALL_CURRENT_U_DELETION):
            batch["ordered_events"] = atomic.torch.zeros_like(batch["ordered_events"])
            batch["ordered_event_mask"] = atomic.torch.zeros_like(
                batch["ordered_event_mask"]
            )
            if control == ALL_CURRENT_U_DELETION:
                batch["drive"] = atomic.torch.zeros_like(batch["drive"])
            return
        batch["ordered_events"] = batch["ordered_events"].clone()
        if control == TIMESTAMP_REVERSAL:
            column = EVENT_FEATURE_NAMES.index("offset_ms")
            batch["ordered_events"][..., column] = atomic.torch.where(
                batch["ordered_event_mask"],
                1.0 - batch["ordered_events"][..., column],
                batch["ordered_events"][..., column],
            )
            return
        if control == RECEPTOR_PERMUTATION:
            columns = [
                EVENT_FEATURE_NAMES.index(name)
                for name in (
                    "ampa_state_increment",
                    "nmda_state_increment",
                    "gabaa_state_increment",
                    "gabab_state_increment",
                    "has_ampa",
                    "has_nmda",
                    "has_gabaa",
                    "has_gabab",
                )
            ]
            batch["ordered_events"][..., columns] = batch["ordered_events"][
                ..., columns
            ].roll(1, dims=-1)
            return
        raise ValueError(control)

    def _direction(self, reference: Any) -> Any:
        index = atomic.torch.arange(
            self.layout.segment_count,
            dtype=reference.dtype,
            device=reference.device,
        )
        direction = atomic.torch.sin(index * 1.61803398875 + 0.5)
        direction = direction / direction.square().mean().sqrt().clamp_min(1e-6)
        return direction[None, :].expand_as(reference)

    def _passive_relative_directional_penalty(
        self,
        model: Any,
        batch: Mapping[str, Any],
        prediction: Any,
    ) -> Tuple[Any, Dict[str, float]]:
        epsilon = self.config.directional_perturbation_mv
        perturbed = dict(batch)
        perturbed["voltage"] = batch["voltage"] + epsilon * self._direction(
            batch["voltage"]
        )
        _, _, _, perturbed_prediction = self._forward_recursive(model, perturbed)
        passive = self._apply_physical_target(
            atomic.torch.zeros_like(prediction),
            batch["voltage"],
            NET_EFFECTIVE_SOURCE,
            batch,
        )
        passive_perturbed = self._apply_physical_target(
            atomic.torch.zeros_like(perturbed_prediction),
            perturbed["voltage"],
            NET_EFFECTIVE_SOURCE,
            perturbed,
        )
        learned_gain = ((perturbed_prediction - prediction) / epsilon).square().mean()
        passive_gain = ((passive_perturbed - passive) / epsilon).square().mean()
        penalty = atomic.torch.relu(
            learned_gain - self.config.passive_relative_margin * passive_gain
        )
        return penalty, {
            "learned_directional_gain": float(learned_gain.detach().cpu()),
            "passive_directional_gain": float(passive_gain.detach().cpu()),
        }

    def _recursive_training_loss(
        self,
        model: Any,
        spec: Tuple[str, str, str],
        windows: np.ndarray,
        device: Any,
    ) -> Tuple[Any, Dict[str, float]]:
        auxiliary_mode, exposure_mode, stability_mode = spec
        values = self.window_data["fit"]
        current = atomic.torch.as_tensor(values["voltage_t"][windows, 0], device=device)
        endpoint_losses = []
        native_losses = []
        auxiliary_losses = []
        stability_losses = []
        directional = {"learned_directional_gain": 0.0, "passive_directional_gain": 0.0}
        for step in range(self.config.recursive_training_horizon_ms):
            if exposure_mode == TEACHER_BOUNDARY:
                model_voltage = atomic.torch.as_tensor(
                    values["voltage_t"][windows, step], device=device
                )
            elif exposure_mode == PUSHFORWARD_4MS:
                model_voltage = current
            else:
                raise ValueError(exposure_mode)
            batch = self._window_batch("fit", windows, step, model_voltage, device)
            decoded, _, auxiliary, prediction = self._forward_recursive(model, batch)
            target_source = self._normalized_source_target(
                model_voltage, batch["target_voltage"], "authentic"
            )
            source_scale = self._scale_tensor(self.selected_source_scaling, decoded)
            native_losses.append(
                atomic.torch_functional.smooth_l1_loss(
                    decoded / source_scale, target_source / source_scale
                )
            )
            endpoint_losses.append(
                atomic.torch.mean(
                    self._activity_weight(batch["target_voltage"] - model_voltage)
                    * atomic.torch_functional.smooth_l1_loss(
                        (prediction - batch["target_voltage"])
                        / self.config.voltage_scale_mv,
                        atomic.torch.zeros_like(prediction),
                        reduction="none",
                    )
                )
            )
            if auxiliary_mode == CAUSAL_SYNAPTIC_AUXILIARY:
                auxiliary_losses.append(
                    atomic.torch_functional.smooth_l1_loss(
                        auxiliary, batch["causal_auxiliary_target"]
                    )
                )
            elif auxiliary_mode == AUXILIARY_OFF:
                auxiliary_losses.append(auxiliary.sum() * 0.0)
            else:
                raise ValueError(auxiliary_mode)
            if stability_mode == PASSIVE_RELATIVE_DIRECTIONAL and step == 0:
                penalty, directional = self._passive_relative_directional_penalty(
                    model, batch, prediction
                )
                stability_losses.append(penalty)
            elif stability_mode == STABILITY_OFF:
                stability_losses.append(prediction.sum() * 0.0)
            elif stability_mode != PASSIVE_RELATIVE_DIRECTIONAL:
                raise ValueError(stability_mode)
            if exposure_mode == PUSHFORWARD_4MS:
                # Brandstetter-style pushforward: expose the next call to the
                # model distribution without backpropagating through history.
                current = prediction.detach()
        endpoint = atomic.torch.stack(endpoint_losses).mean()
        native = atomic.torch.stack(native_losses).mean()
        auxiliary = atomic.torch.stack(auxiliary_losses).mean()
        stability = (
            atomic.torch.stack(stability_losses).mean()
            if stability_losses
            else endpoint * 0.0
        )
        physical = atomic.torch.stack(
            [
                atomic.torch.relu(prediction - self.config.physical_voltage_maximum_mv)
                .square()
                .mean(),
                atomic.torch.relu(self.config.physical_voltage_minimum_mv - prediction)
                .square()
                .mean(),
            ]
        ).sum() / (self.config.voltage_scale_mv**2)
        loss = endpoint
        if self.selected_objective == HYBRID:
            loss = loss + self.config.hybrid_native_weight * native
        loss = (
            loss
            + self.config.auxiliary_loss_weight * auxiliary
            + self.config.directional_stability_weight * stability
            + self.config.physical_penalty_weight * physical
        )
        return loss, {
            "endpoint": float(endpoint.detach().cpu()),
            "native": float(native.detach().cpu()),
            "auxiliary": float(auxiliary.detach().cpu()),
            "stability": float(stability.detach().cpu()),
            **directional,
        }

    def _recursive_metrics_06br(
        self,
        model: Any,
        role: str,
        windows: np.ndarray,
        device: Any,
        *,
        control: Optional[str] = None,
    ) -> Dict[str, Any]:
        values = self.window_data[role]
        windows = np.asarray(windows, dtype=np.int64)
        horizon = values["voltage_t"].shape[1]
        current = atomic.torch.as_tensor(values["voltage_t"][windows, 0], device=device)
        passive_current = current.clone()
        predictions = []
        passive_predictions = []
        auxiliary_predictions = []
        auxiliary_targets = []
        model.eval()
        with atomic.torch.no_grad():
            for step in range(horizon):
                batch = self._window_batch(role, windows, step, current, device)
                self._apply_control(batch, control)
                _, _, auxiliary, current = self._forward_recursive(model, batch)
                passive_current = self._apply_physical_target(
                    atomic.torch.zeros_like(passive_current),
                    passive_current,
                    NET_EFFECTIVE_SOURCE,
                    batch,
                )
                predictions.append(current)
                passive_predictions.append(passive_current)
                auxiliary_predictions.append(auxiliary)
                auxiliary_targets.append(batch["causal_auxiliary_target"])
        prediction = atomic.torch.stack(predictions, dim=1).cpu().numpy()
        passive = atomic.torch.stack(passive_predictions, dim=1).cpu().numpy()
        target = values["voltage_t1"][windows]
        initial = values["voltage_t"][windows, 0]
        result: Dict[str, Any] = {}
        for step in (1, 2, 4, 8):
            if step > horizon:
                continue
            error = prediction[:, step - 1] - target[:, step - 1]
            baseline = passive[:, step - 1] - target[:, step - 1]
            rmse = float(np.sqrt(np.mean(error.astype(np.float64) ** 2)))
            passive_rmse = float(np.sqrt(np.mean(baseline.astype(np.float64) ** 2)))
            activity = np.abs(target[:, step - 1] - initial)
            regimes = {}
            for label, mask in (
                ("quiescent_lt_1mV", activity < 1.0),
                ("moderate_1_to_5mV", (activity >= 1.0) & (activity < 5.0)),
                ("active_ge_5mV", activity >= 5.0),
            ):
                regimes[label] = (
                    float(np.sqrt(np.mean(error[mask].astype(np.float64) ** 2)))
                    if mask.any()
                    else None
                )
            result[str(step)] = {
                "endpoint_rmse_mv": rmse,
                "passive_endpoint_rmse_mv": passive_rmse,
                "gain_over_passive_fraction": 1.0
                - rmse / max(passive_rmse, 1e-12),
                "mean_drift_mv": float(np.mean(error)),
                "physical_voltage_violation_count": int(
                    (
                        (prediction[:, :step] < self.config.physical_voltage_minimum_mv)
                        | (
                            prediction[:, :step]
                            > self.config.physical_voltage_maximum_mv
                        )
                    ).sum()
                ),
                "regime_rmse_mv": regimes,
            }
        aux_prediction = atomic.torch.stack(auxiliary_predictions, dim=1).cpu().numpy()
        aux_target = atomic.torch.stack(auxiliary_targets, dim=1).cpu().numpy()
        result["auxiliary_normalized_rmse"] = float(
            np.sqrt(np.mean((aux_prediction - aux_target) ** 2))
        )
        return result

    def _balanced_window_rows(
        self, role: str
    ) -> Tuple[np.ndarray, np.ndarray]:
        values = self.window_data[role]
        mask = values["ordered_event_mask"][:, 0]
        event = mask.reshape(len(mask), -1).any(axis=1)
        return np.flatnonzero(event), np.flatnonzero(~event)

    def train_recursive_factorial_matrix(self) -> Dict[str, Any]:
        device = atomic.torch.device("cuda" if atomic.torch.cuda.is_available() else "cpu")
        specs = self._specs()
        cal_count = len(self.window_data["calibration"]["indices"])
        calibration_a = np.arange(cal_count, dtype=np.int64)[::2]
        calibration_b = np.arange(cal_count, dtype=np.int64)[1::2]
        positive, negative = self._balanced_window_rows("fit")
        half = self.config.recursive_batch_window_count // 2
        if min(len(positive), len(negative)) < half:
            raise RuntimeError("06b-r balanced fit windows are insufficient")
        reports: Dict[str, Any] = {}
        for seed in self.config.pilot_seeds:
            models = {spec: self._new_recursive_model(seed, device) for spec in specs}
            optimizers = {
                spec: atomic.torch.optim.AdamW(
                    model.parameters(),
                    lr=self.config.matrix_learning_rate,
                    weight_decay=self.config.matrix_weight_decay,
                )
                for spec, model in models.items()
            }
            best: Dict[Tuple[str, str, str], Tuple[float, int, Dict[str, Any]]] = {}
            histories = {self._spec_key(spec): [] for spec in specs}
            rng = np.random.default_rng(seed + 708000)
            digest = hashlib.sha256()
            progress = atomic._CompactProgress(
                f"06b-r 2x2x2 seed={seed}",
                self.config.recursive_training_steps,
                max(1, self.config.recursive_training_steps // 12),
            )
            for step in range(self.config.recursive_training_steps + 1):
                if step in self.config.recursive_checkpoints:
                    for spec, model in models.items():
                        metrics = self._recursive_metrics_06br(
                            model, "calibration", calibration_a, device
                        )
                        key = self._spec_key(spec)
                        histories[key].append(
                            {"step": step, "calibration_half_A": metrics}
                        )
                        score = metrics["8"]["endpoint_rmse_mv"]
                        if spec not in best or score < best[spec][0]:
                            best[spec] = (
                                score,
                                step,
                                self._copy_state_dict(model),
                            )
                if step == self.config.recursive_training_steps:
                    break
                windows = np.concatenate(
                    (
                        rng.choice(positive, half, replace=False),
                        rng.choice(
                            negative,
                            self.config.recursive_batch_window_count - half,
                            replace=False,
                        ),
                    )
                )
                rng.shuffle(windows)
                digest.update(windows.astype(np.int64).tobytes())
                losses = []
                for spec, model in models.items():
                    optimizer = optimizers[spec]
                    optimizer.zero_grad(set_to_none=True)
                    loss, _ = self._recursive_training_loss(
                        model, spec, windows, device
                    )
                    if not bool(atomic.torch.isfinite(loss)):
                        raise RuntimeError(
                            f"nonfinite 06b-r loss: {self._spec_key(spec)}"
                        )
                    loss.backward()
                    atomic.torch.nn.utils.clip_grad_norm_(
                        model.parameters(), self.config.matrix_gradient_clip_norm
                    )
                    optimizer.step()
                    losses.append(float(loss.detach().cpu()))
                progress.update(step + 1, f"loss={float(np.median(losses)):.3g}")
            selected: Dict[str, Any] = {}
            for spec, model in models.items():
                score, selected_step, state = best[spec]
                model.load_state_dict(state)
                model.eval()
                key = self._spec_key(spec)
                self.recursive_models[(key, seed)] = model
                path = self.output_dir / f"recursive_{key}_seed{seed}.pt"
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
                    "calibration_half_A_8ms_rmse_mv": score,
                    "calibration_half_B": self._recursive_metrics_06br(
                        model, "calibration", calibration_b, device
                    ),
                    "checkpoint": path.name,
                    "checkpoint_sha256": atomic._sha256_file(path),
                }
            reports[str(seed)] = {
                "batch_stream_sha256": digest.hexdigest(),
                "histories": histories,
                "selected": selected,
            }
        scores = {
            self._spec_key(spec): self._median(
                report["selected"][self._spec_key(spec)]["calibration_half_B"]["8"][
                    "endpoint_rmse_mv"
                ]
                for report in reports.values()
            )
            for spec in specs
        }
        passive_scores = {
            self._spec_key(spec): self._median(
                report["selected"][self._spec_key(spec)]["calibration_half_B"]["8"][
                    "passive_endpoint_rmse_mv"
                ]
                for report in reports.values()
            )
            for spec in specs
        }
        eligible = []
        eligibility: Dict[str, Any] = {}
        for spec in specs:
            key = self._spec_key(spec)
            per_seed = []
            for report in reports.values():
                metrics = report["selected"][key]["calibration_half_B"]["8"]
                per_seed.append(
                    metrics["endpoint_rmse_mv"]
                    <= metrics["passive_endpoint_rmse_mv"]
                    * (1.0 + self.config.maximum_per_seed_regression_fraction)
                )
            median_gain = 1.0 - scores[key] / max(passive_scores[key], 1e-12)
            passed = bool(
                median_gain >= self.config.minimum_candidate_gain_over_passive_fraction
                and all(per_seed)
            )
            eligibility[key] = {
                "median_gain_over_passive_fraction": median_gain,
                "no_seed_regressed": all(per_seed),
                "passed": passed,
            }
            if passed:
                eligible.append(key)
        diagnostic_best = min(scores, key=scores.get)
        selected_candidate = min(eligible, key=scores.get) if eligible else None
        report = {
            "schema_version": "06b-r-training-v1",
            "valid": True,
            "factorial_arm_count": len(specs),
            "same_parameter_count": True,
            "same_initialization_within_seed": True,
            "same_minibatch_stream_within_seed": True,
            "pushforward_previous_prediction_detached": True,
            "calibration_half_A_selects_checkpoints": True,
            "calibration_half_B_selects_arm": True,
            "development_used_during_training": False,
            "median_calibration_half_B_8ms_rmse_mv": scores,
            "median_calibration_half_B_passive_8ms_rmse_mv": passive_scores,
            "eligibility": eligibility,
            "diagnostic_best_arm": diagnostic_best,
            "selected_candidate": selected_candidate,
            "reports": reports,
        }
        atomic._write_json(self.output_dir / "recursive_factorial_training.json", report)
        return report

    def _frozen_06bq_reference(self, device: Any) -> Dict[str, Any]:
        if self.source_06bq_root is None:
            raise RuntimeError("06b-q source was not prepared")
        spec = (CHRONOLOGICAL_JUMP, NONZERO_ROBUST_LOG)
        per_seed = {}
        for seed in self.config.pilot_seeds:
            model = EventSupportedJumpPlayground._new_jump_model(self, seed, device)
            path = (
                self.source_06bq_root
                / f"jump_chronological_jump__nonzero_robust_log_seed{seed}.pt"
            )
            checkpoint = atomic.torch.load(path, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint["state_dict"])
            model.eval()
            per_seed[str(seed)] = self._recursive_metrics(
                model, spec, device, gate_mode=UNGATED_RESIDUAL
            )
        return {
            "valid": True,
            "frozen": True,
            "retraining_performed": False,
            "same_development_support": True,
            "per_seed": per_seed,
            "median_8ms_rmse_mv": self._median(
                row["8"]["endpoint_rmse_mv"] for row in per_seed.values()
            ),
            "median_passive_8ms_rmse_mv": self._median(
                row["8"]["passive_endpoint_rmse_mv"] for row in per_seed.values()
            ),
        }

    def _factor_contrasts(self, reports: Mapping[str, Any]) -> Dict[str, Any]:
        values = {
            key: self._median(
                row["recursive"]["8"]["endpoint_rmse_mv"] for row in rows
            )
            for key, rows in reports.items()
        }
        result = {}
        axes = (
            ("causal_event_auxiliary", 0, AUXILIARY_OFF, CAUSAL_SYNAPTIC_AUXILIARY),
            ("boundary_exposure", 1, TEACHER_BOUNDARY, PUSHFORWARD_4MS),
            ("local_stability", 2, STABILITY_OFF, PASSIVE_RELATIVE_DIRECTIONAL),
        )
        for name, index, control, intervention in axes:
            control_values = [
                score
                for key, score in values.items()
                if key.split("|")[index] == control
            ]
            intervention_values = [
                score
                for key, score in values.items()
                if key.split("|")[index] == intervention
            ]
            control_median = self._median(control_values)
            intervention_median = self._median(intervention_values)
            result[name] = {
                "control": control,
                "intervention": intervention,
                "control_median_8ms_rmse_mv": control_median,
                "intervention_median_8ms_rmse_mv": intervention_median,
                "marginal_gain_fraction": 1.0
                - intervention_median / max(control_median, 1e-12),
            }
        return result

    def evaluate_recursive_factorial_matrix(
        self, training: Mapping[str, Any]
    ) -> Dict[str, Any]:
        """Open development only after calibration selection is frozen."""

        device = atomic.torch.device("cuda" if atomic.torch.cuda.is_available() else "cpu")
        windows = np.arange(len(self.window_data["development"]["indices"]), dtype=np.int64)
        reports: Dict[str, Any] = {}
        for spec in self._specs():
            key = self._spec_key(spec)
            rows = []
            for seed in self.config.pilot_seeds:
                model = self.recursive_models[(key, seed)]
                rows.append(
                    {
                        "seed": seed,
                        "recursive": self._recursive_metrics_06br(
                            model, "development", windows, device
                        ),
                    }
                )
            reports[key] = rows
        diagnostic_best = str(training["diagnostic_best_arm"])
        control_reports = {}
        for control in CAUSAL_CONTROLS:
            control_reports[control] = []
            for seed in self.config.pilot_seeds:
                model = self.recursive_models[(diagnostic_best, seed)]
                control_reports[control].append(
                    {
                        "seed": seed,
                        "recursive": self._recursive_metrics_06br(
                            model,
                            "development",
                            windows,
                            device,
                            control=control,
                        ),
                    }
                )
        reference = self._frozen_06bq_reference(device)
        report = {
            "schema_version": "06b-r-evaluation-v1",
            "valid": True,
            "development_opened_after_selection": True,
            "development_used_for_selection": False,
            "validation_state_accessed": False,
            "test_state_accessed": False,
            "fresh_test_accessed": False,
            "diagnostic_best_arm": diagnostic_best,
            "selected_candidate_from_calibration": training.get("selected_candidate"),
            "factor_contrasts": self._factor_contrasts(reports),
            "reports": reports,
            "causal_controls_for_diagnostic_best": control_reports,
            "frozen_06bq_reference": reference,
        }
        atomic._write_json(self.output_dir / "recursive_factorial_evaluation.json", report)
        return report

    def _control_materiality(
        self, evaluation: Mapping[str, Any], arm: str, control: str
    ) -> Dict[str, float]:
        ordinary = self._median(
            row["recursive"]["8"]["endpoint_rmse_mv"]
            for row in evaluation["reports"][arm]
        )
        ablated = self._median(
            row["recursive"]["8"]["endpoint_rmse_mv"]
            for row in evaluation["causal_controls_for_diagnostic_best"][control]
        )
        return {
            "baseline_8ms_rmse_mv": ordinary,
            "ablated_8ms_rmse_mv": ablated,
            "absolute_increase_mv": ablated - ordinary,
            "relative_increase_fraction": (ablated - ordinary) / max(ordinary, 1e-12),
        }

    def finalize_recursive_event_exposure_playground(
        self,
        contract: Mapping[str, Any],
        auxiliary_probe: Mapping[str, Any],
        training: Mapping[str, Any],
        evaluation: Mapping[str, Any],
    ) -> Dict[str, Any]:
        arm = str(training["diagnostic_best_arm"])
        rows = evaluation["reports"][arm]
        recursive_rmse = self._median(
            row["recursive"]["8"]["endpoint_rmse_mv"] for row in rows
        )
        passive_rmse = self._median(
            row["recursive"]["8"]["passive_endpoint_rmse_mv"] for row in rows
        )
        recursive_gain = 1.0 - recursive_rmse / max(passive_rmse, 1e-12)
        per_seed_non_regression = all(
            row["recursive"]["8"]["endpoint_rmse_mv"]
            <= row["recursive"]["8"]["passive_endpoint_rmse_mv"]
            * (1.0 + self.config.maximum_per_seed_regression_fraction)
            for row in rows
        )
        physical_violations = int(
            sum(
                row["recursive"]["8"]["physical_voltage_violation_count"]
                for row in rows
            )
        )
        controls = {
            control: self._control_materiality(evaluation, arm, control)
            for control in CAUSAL_CONTROLS
        }
        event_material = bool(
            controls[ORDERED_EVENT_PATH_DELETION]["relative_increase_fraction"]
            >= self.config.minimum_event_materiality_fraction
        )
        development_gate = bool(
            recursive_gain >= self.config.minimum_candidate_gain_over_passive_fraction
            and per_seed_non_regression
            and physical_violations == 0
            and auxiliary_probe["registered_gate_passed"]
            and event_material
        )
        calibration_candidate = training.get("selected_candidate")
        selected_candidate = arm if calibration_candidate == arm and development_gate else None
        contrasts = evaluation["factor_contrasts"]
        if selected_candidate is not None:
            diagnosis = "RECURSIVE_EVENT_EXPOSURE_CANDIDATE_PASSES_TRAIN_ONLY_GATES"
            next_step = "freeze candidate and run a new sealed-test confirmation"
        elif not auxiliary_probe["registered_gate_passed"]:
            diagnosis = "CAUSAL_EVENT_AUXILIARY_FAILS_PERMUTED_TARGET_CONTROL"
            next_step = "redesign the event-side target or encoder before another coupled model"
        elif contrasts["boundary_exposure"]["marginal_gain_fraction"] > 0.0:
            diagnosis = "PUSHFORWARD_HELPS_BUT_PASSIVE_NON_REGRESSION_GATE_REMAINS_UNMET"
            next_step = "retain pushforward exposure and isolate the remaining local source error"
        elif contrasts["local_stability"]["marginal_gain_fraction"] > 0.0:
            diagnosis = "DIRECTIONAL_STABILITY_HELPS_BUT_EXPOSURE_MISMATCH_REMAINS"
            next_step = "retain passive-relative stability and redesign boundary exposure"
        else:
            diagnosis = "NO_FACTORIAL_ARM_BEATS_THE_SUPPORT_MATCHED_PASSIVE_PRIOR"
            next_step = "retain passive Hines baseline and redesign the learned residual contract"
        report = {
            "schema_version": "06b-r-final-report-v1",
            "valid": True,
            "diagnosis": diagnosis,
            "diagnostic_best_arm": arm,
            "selected_candidate": selected_candidate,
            "candidate_reinstated": False,
            "sealed_test_authorized": selected_candidate is not None,
            "full_training_authorized": False,
            "mass_dataset_generation_authorized": False,
            "median_development_8ms_rmse_mv": recursive_rmse,
            "median_support_matched_passive_8ms_rmse_mv": passive_rmse,
            "median_gain_over_passive_fraction": recursive_gain,
            "per_seed_non_regression": per_seed_non_regression,
            "physical_voltage_violation_count": physical_violations,
            "causal_auxiliary_probe": {
                "gain_over_permuted_fraction": auxiliary_probe[
                    "gain_over_permuted_fraction"
                ],
                "registered_gate_passed": auxiliary_probe["registered_gate_passed"],
            },
            "factor_contrasts": contrasts,
            "causal_controls": controls,
            "ordered_event_path_materiality_passed": event_material,
            "frozen_06bq_reference": evaluation["frozen_06bq_reference"],
            "teacher_mechanism_state_during_rollout": True,
            "teacher_mechanism_state_is_deployment_ready": False,
            "development_used_for_selection": False,
            "validation_state_accessed": False,
            "test_state_accessed": False,
            "fresh_test_accessed": False,
            "next_step": next_step,
            "code_revision": self.code_revision,
        }
        atomic._write_json(self.output_dir / "final_report.json", report)
        self._write_artifact_index()
        return report


__all__ = [
    "EXPECTED_06BQ_ARCHIVE_SHA256",
    "EXPECTED_06BQ_INDEX_SHA256",
    "EXPECTED_06BQ_FINAL_SHA256",
    "AUXILIARY_OFF",
    "CAUSAL_SYNAPTIC_AUXILIARY",
    "AUXILIARY_MODES",
    "TEACHER_BOUNDARY",
    "PUSHFORWARD_4MS",
    "EXPOSURE_MODES",
    "STABILITY_OFF",
    "PASSIVE_RELATIVE_DIRECTIONAL",
    "STABILITY_MODES",
    "CAUSAL_CONTROLS",
    "RecursiveEventSourceCell",
    "RecursiveEventExposureConfig",
    "RecursiveEventExposurePlayground",
    "verified_06bq_artifact_root",
]
