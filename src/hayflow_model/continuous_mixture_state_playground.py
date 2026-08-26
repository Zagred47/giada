"""06b-m: train-only continuous mixture-state architecture playground.

The experiment keeps the persistence and frozen dynamic voltage experts from
06b-l fixed.  It trains only a small causal controller whose recurrent hidden
state emits a convex mixing coefficient for every segment and millisecond.
The aligned matrix isolates input observability, temporal recurrence,
morphological messages and oracle-target distillation.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from . import atomic_state_dynamics_playground as atomic
from .topology_controlled_recurrence_expansion import (
    topology_relabelled_parent_ids,
)
from .voltage_error_model_revision import (
    STATIC_REFERENCE,
    TEACHER_OPTIMAL_BLEND_ORACLE,
    VoltageErrorModelRevision,
    VoltageErrorModelRevisionConfig,
)


EXPECTED_06BL_ARCHIVE_SHA256 = (
    "8af4278ec0eb9cd0594cd5ad5cc7ff3b0f5050e20be5b17df9013348ca14b41e"
)
EXPECTED_06BL_INDEX_SHA256 = (
    "959aea3c0f60613210452ddfa3d0fdb1f1be06ca8b113ac50a582e37b38ae529"
)
EXPECTED_06BL_FINAL_SHA256 = (
    "884e2d7b7a5f1f6796048618d727ba6267c8d2bfe92287ec89826995ccbdda90"
)

VOLTAGE_INSTANTANEOUS = "voltage_instantaneous"
PHYSIOLOGY_INSTANTANEOUS = "physiology_instantaneous"
LOCAL_RECURRENT = "local_recurrent"
TREE_RECURRENT = "tree_recurrent"
SHUFFLED_TREE_RECURRENT = "shuffled_tree_recurrent"
TREE_RECURRENT_NO_ORACLE_AUX = "tree_recurrent_no_oracle_aux"

FULL_MATRIX_ARMS = (
    VOLTAGE_INSTANTANEOUS,
    PHYSIOLOGY_INSTANTANEOUS,
    LOCAL_RECURRENT,
    TREE_RECURRENT,
    SHUFFLED_TREE_RECURRENT,
    TREE_RECURRENT_NO_ORACLE_AUX,
)
SCALING_ARMS = (LOCAL_RECURRENT, TREE_RECURRENT, SHUFFLED_TREE_RECURRENT)


def verified_06bl_artifact_root(
    source: Path, cache_dir: Path
) -> Tuple[Path, Dict[str, Any]]:
    """Verify the exact terminal 06b-l diagnostic that authorizes 06b-m."""

    source = Path(source).expanduser().resolve()
    cache_dir = Path(cache_dir).resolve()
    if not source.exists():
        raise FileNotFoundError(source)
    archive_hash = "extracted-directory"
    if source.is_file():
        if source.suffix.lower() != ".zip":
            raise RuntimeError("06b-l source must be a ZIP or extracted directory")
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
    matches = [
        path.parent
        for path in search_root.rglob("artifact_index.json")
        if atomic._sha256_file(path) == EXPECTED_06BL_INDEX_SHA256
    ]
    if len(matches) != 1:
        raise RuntimeError(f"expected one exact 06b-l artifact; found {len(matches)}")
    root = matches[0]
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
        raise RuntimeError(f"06b-l indexed member verification failed: {failures}")
    final_path = root / "final_report.json"
    if atomic._sha256_file(final_path) != EXPECTED_06BL_FINAL_SHA256:
        raise RuntimeError("06b-l final report SHA-256 mismatch")
    final = json.loads(final_path.read_text(encoding="utf-8"))
    if (
        final.get("diagnosis")
        != "OPTIMAL_BLEND_ORACLE_WORKS_BUT_REGIME_GATE_FAILS"
        or final.get("next_step") != "architecture_revision_continuous_mixture_state"
        or final.get("teacher_optimal_blend_oracle_passed") is not True
        or final.get("coupled_06c_canary_authorized") is not False
    ):
        raise RuntimeError("06b-l result does not authorize 06b-m")
    if source.is_file() and archive_hash != EXPECTED_06BL_ARCHIVE_SHA256:
        archive_hash = "kaggle-repacked"
    return root, {
        "valid": True,
        "source_kind": "zip" if source.is_file() else "directory",
        "archive_sha256": archive_hash,
        "artifact_index_sha256": EXPECTED_06BL_INDEX_SHA256,
        "final_report_sha256": EXPECTED_06BL_FINAL_SHA256,
        "indexed_member_count": len(index.get("artifacts", [])),
        "diagnosis": final["diagnosis"],
        "next_step": final["next_step"],
    }


@dataclass(frozen=True)
class ContinuousMixtureStateConfig(VoltageErrorModelRevisionConfig):
    mixture_hidden_widths: Tuple[int, ...] = (8, 16, 32)
    full_matrix_hidden_width: int = 16
    mixture_training_steps: int = 400
    mixture_checkpoints: Tuple[int, ...] = (0, 100, 200, 400)
    mixture_batch_window_count: int = 4
    mixture_learning_rate: float = 3e-4
    mixture_weight_decay: float = 1e-5
    mixture_gradient_clip_norm: float = 1.0
    mixture_region_embedding_width: int = 4
    mixture_oracle_auxiliary_weight: float = 0.25
    mixture_STATE_loss_weight: float = 0.25
    mixture_physical_penalty_weight: float = 1.0
    mixture_drift_penalty_weight: float = 0.1
    maximum_mixture_parameter_count: int = 20000
    topology_relabel_seed_offset: int = 606400
    minimum_topology_gain_fraction: float = 0.01
    minimum_memory_gain_fraction: float = 0.01
    minimum_observability_gain_fraction: float = 0.01
    minimum_oracle_alpha_correlation: float = 0.10

    def validate(self) -> None:
        super().validate()
        if tuple(sorted(set(self.mixture_hidden_widths))) != self.mixture_hidden_widths:
            raise ValueError("06b-m hidden widths must be sorted and unique")
        if self.full_matrix_hidden_width not in self.mixture_hidden_widths:
            raise ValueError("06b-m full-matrix width must belong to scaling widths")
        if self.mixture_checkpoints[0] != 0:
            raise ValueError("06b-m checkpoints must start at zero")
        if tuple(sorted(set(self.mixture_checkpoints))) != self.mixture_checkpoints:
            raise ValueError("06b-m checkpoints must be sorted and unique")
        if self.mixture_checkpoints[-1] != self.mixture_training_steps:
            raise ValueError("06b-m final checkpoint must equal training steps")
        positive = (
            *self.mixture_hidden_widths,
            self.mixture_training_steps,
            self.mixture_batch_window_count,
            self.mixture_learning_rate,
            self.mixture_gradient_clip_norm,
            self.mixture_region_embedding_width,
            self.maximum_mixture_parameter_count,
        )
        if any(value <= 0 for value in positive):
            raise ValueError("06b-m positive configuration value is invalid")
        if self.mixture_oracle_auxiliary_weight < 0 or self.mixture_STATE_loss_weight < 0:
            raise ValueError("06b-m loss weights cannot be negative")

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "ContinuousMixtureStateConfig":
        payload = dict(values)
        for name in (
            "rollout_horizons_ms",
            "voltage_path_sample_indices",
            "pilot_seeds",
            "scaling_checkpoints",
            "repair_checkpoints",
            "scheduled_checkpoints",
            "objective_checkpoints",
            "mixture_hidden_widths",
            "mixture_checkpoints",
        ):
            if name in payload:
                payload[name] = tuple(map(int, payload[name]))
        for name in (
            "voltage_shrinkage_grid",
            "activity_edges_mv",
            "analytic_shrinkage_strengths",
            "analytic_voltage_edges_mv",
            "temporal_ridge_strengths",
            "gate_ridge_strengths",
            "hurdle_probability_thresholds",
        ):
            if name in payload:
                payload[name] = tuple(map(float, payload[name]))
        result = cls(**payload)
        result.validate()
        return result


if atomic.torch is not None:

    class ContinuousMixtureCell(atomic.nn.Module):
        """Parameter-matched causal controller used by every aligned arm."""

        def __init__(
            self,
            feature_width: int,
            region_count: int,
            region_width: int,
            hidden_width: int,
        ) -> None:
            super().__init__()
            self.hidden_width = int(hidden_width)
            self.region = atomic.nn.Embedding(region_count, region_width)
            self.encoder = atomic.nn.Sequential(
                atomic.nn.Linear(feature_width + region_width, hidden_width),
                atomic.nn.SiLU(),
            )
            self.message = atomic.nn.Linear(hidden_width, hidden_width, bias=False)
            self.recurrent = atomic.nn.GRUCell(hidden_width, hidden_width)
            self.readout = atomic.nn.Linear(hidden_width, 1)
            atomic.nn.init.zeros_(self.region.weight)
            atomic.nn.init.zeros_(self.readout.weight)
            atomic.nn.init.zeros_(self.readout.bias)

        @staticmethod
        def _tree_message(
            hidden: Any, parent: Any, children: Any, child_mask: Any
        ) -> Any:
            parent_value = hidden[:, parent]
            child_value = (
                hidden[:, children] * child_mask[None, :, :, None]
            ).sum(dim=2)
            child_value = child_value / child_mask.sum(dim=1).clamp_min(1.0)[
                None, :, None
            ]
            return 0.5 * (parent_value + child_value)

        def forward(
            self,
            features: Any,
            region_ids: Any,
            hidden: Any,
            parent: Any,
            children: Any,
            child_mask: Any,
            *,
            recurrent: bool,
            topology: str,
        ) -> Tuple[Any, Any]:
            batch, segments, _ = features.shape
            if not recurrent:
                hidden = atomic.torch.zeros_like(hidden)
            region = self.region(region_ids)[None, :, :].expand(batch, -1, -1)
            encoded = self.encoder(atomic.torch.cat((features, region), dim=-1))
            if topology == "none":
                message = atomic.torch.zeros_like(hidden)
            elif topology == "local":
                message = hidden
            elif topology == "tree":
                message = self._tree_message(hidden, parent, children, child_mask)
            else:
                raise ValueError(topology)
            recurrent_input = encoded + self.message(message)
            next_hidden = self.recurrent(
                recurrent_input.reshape(batch * segments, -1),
                hidden.reshape(batch * segments, -1),
            ).reshape(batch, segments, -1)
            alpha = atomic.torch.sigmoid(self.readout(next_hidden).squeeze(-1))
            return alpha, next_hidden

else:  # pragma: no cover
    ContinuousMixtureCell = None


class ContinuousMixtureStatePlayground(VoltageErrorModelRevision):
    """Aligned component playground for a causal continuous mixture state."""

    config: ContinuousMixtureStateConfig

    def __init__(
        self,
        bundle: Any,
        output_dir: Path,
        config: ContinuousMixtureStateConfig,
        artifact_05t_source: Path,
        artifact_06b_source: Path,
        artifact_06bb_source: Path,
        artifact_06bc_source: Path,
        artifact_06bd_source: Path,
        artifact_06be_source: Path,
        artifact_06bf_source: Path,
        artifact_06bg_source: Path,
        artifact_06bh_source: Path,
        artifact_06bi_source: Path,
        artifact_06bj_source: Path,
        artifact_06bk_source: Path,
        artifact_06bl_source: Path,
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
            artifact_06bd_source,
            artifact_06be_source,
            artifact_06bf_source,
            artifact_06bg_source,
            artifact_06bh_source,
            artifact_06bi_source,
            artifact_06bj_source,
            artifact_06bk_source,
            code_revision=code_revision,
        )
        self.artifact_06bl_source = Path(artifact_06bl_source)
        self.mixture_models: Dict[Tuple[str, int, int], Any] = {}
        self.topologies: Dict[Tuple[str, int], np.ndarray] = {}
        self.mixture_feature_width: Optional[int] = None
        self.training_valid = False

    @staticmethod
    def _arm_contract(arm: str) -> Dict[str, Any]:
        return {
            VOLTAGE_INSTANTANEOUS: {
                "input": "voltage_only",
                "recurrent": False,
                "topology": "none",
                "oracle_auxiliary": True,
            },
            PHYSIOLOGY_INSTANTANEOUS: {
                "input": "full_physiology",
                "recurrent": False,
                "topology": "none",
                "oracle_auxiliary": True,
            },
            LOCAL_RECURRENT: {
                "input": "full_physiology",
                "recurrent": True,
                "topology": "local",
                "oracle_auxiliary": True,
            },
            TREE_RECURRENT: {
                "input": "full_physiology",
                "recurrent": True,
                "topology": "authentic_tree",
                "oracle_auxiliary": True,
            },
            SHUFFLED_TREE_RECURRENT: {
                "input": "full_physiology",
                "recurrent": True,
                "topology": "relabelled_tree",
                "oracle_auxiliary": True,
            },
            TREE_RECURRENT_NO_ORACLE_AUX: {
                "input": "full_physiology",
                "recurrent": True,
                "topology": "authentic_tree",
                "oracle_auxiliary": False,
            },
        }[arm]

    def _run_specs(self) -> Tuple[Tuple[str, int], ...]:
        specs = [
            (arm, self.config.full_matrix_hidden_width) for arm in FULL_MATRIX_ARMS
        ]
        for width in self.config.mixture_hidden_widths:
            if width == self.config.full_matrix_hidden_width:
                continue
            specs.extend((arm, width) for arm in SCALING_ARMS)
        return tuple(specs)

    @staticmethod
    def _children(parent: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        children = [[] for _ in range(len(parent))]
        for child, value in enumerate(parent):
            if child != int(value):
                children[int(value)].append(child)
        width = max(1, max(map(len, children)))
        ids = np.zeros((len(parent), width), dtype=np.int64)
        mask = np.zeros((len(parent), width), dtype=np.float32)
        for index, values in enumerate(children):
            if values:
                ids[index, : len(values)] = values
                mask[index, : len(values)] = 1.0
        return ids, mask

    def _topology(self, arm: str, seed: int) -> np.ndarray:
        key = (arm, seed)
        if key not in self.topologies:
            parent = np.asarray(self.layout.parent_ids, dtype=np.int64).copy()
            if arm == SHUFFLED_TREE_RECURRENT:
                parent = topology_relabelled_parent_ids(
                    parent, seed=self.config.topology_relabel_seed_offset + seed
                )
            self.topologies[key] = parent
        return self.topologies[key]

    def _feature_width(self) -> int:
        return int(
            5
            + 2 * len(self.coordinate_groups)
            + len(atomic.CAUSAL_DRIVE_FEATURES)
            + len(self.ion_feature_names)
            + self.layout.segment_static.shape[1]
        )

    def _new_mixture_model(self, width: int, device: Any) -> Any:
        model = ContinuousMixtureCell(
            self._feature_width(),
            len(self.layout.region_names),
            self.config.mixture_region_embedding_width,
            width,
        ).to(device)
        count = int(sum(value.numel() for value in model.parameters()))
        if count > self.config.maximum_mixture_parameter_count:
            raise RuntimeError(
                f"06b-m mixture controller has {count} parameters; ceiling is "
                f"{self.config.maximum_mixture_parameter_count}"
            )
        return model

    @staticmethod
    def _state_dict_sha256(state_dict: Mapping[str, Any]) -> str:
        digest = hashlib.sha256()
        for name, value in sorted(state_dict.items()):
            array = value.detach().cpu().contiguous().numpy()
            digest.update(name.encode("utf-8"))
            digest.update(str(array.dtype).encode("ascii"))
            digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
            digest.update(array.tobytes())
        return digest.hexdigest()

    def prepare_continuous_mixture_state(self) -> Dict[str, Any]:
        base = self.prepare_voltage_error_model_revision()
        _, source = verified_06bl_artifact_root(
            self.artifact_06bl_source,
            self.output_dir.parent / ".06bm_artifact_cache" / "06bl",
        )
        self.mixture_feature_width = self._feature_width()
        device = next(iter(self.frozen_state_models.values())).proposal.weight.device
        parameter_counts = {}
        for width in self.config.mixture_hidden_widths:
            parameter_counts[str(width)] = int(
                sum(value.numel() for value in self._new_mixture_model(width, device).parameters())
            )
        arm_contracts = {arm: self._arm_contract(arm) for arm in FULL_MATRIX_ARMS}
        report = {
            **base,
            "schema_version": "06b-m-continuous-mixture-contract-v1",
            "experiment": "continuous_mixture_state_playground",
            "source_06bl": source,
            "architecture_revision": "causal_continuous_mixture_state",
            "experts": ["persistence_zero_delta", "frozen_static_dynamic_update"],
            "arms": arm_contracts,
            "full_matrix_hidden_width": self.config.full_matrix_hidden_width,
            "scaling_hidden_widths": list(self.config.mixture_hidden_widths),
            "scaling_arms": list(SCALING_ARMS),
            "checkpoints": list(self.config.mixture_checkpoints),
            "parameter_counts_by_width": parameter_counts,
            "parameter_matched_within_width": True,
            "same_initialization_within_seed_and_width": True,
            "same_minibatch_stream_within_seed": True,
            "same_frozen_experts_within_seed": True,
            "teacher_endpoint_used_as_input": False,
            "teacher_optimal_blend_used_only_as_training_target": True,
            "mechanism_STATE_updater_frozen": True,
            "voltage_expert_frozen": True,
            "validation_state_accessed": False,
            "test_state_accessed": False,
            "new_independent_confirmation_claimed": False,
            "coupled_06c_canary_authorized": False,
            "full_training_authorized": False,
            "physical_parallelism": (
                "synchronized deterministic sequential arm updates on one GPU"
            ),
        }
        for stale in (
            "terminal_diagnostic_before_architecture_revision",
            "neural_training_performed",
            "optimizer_used",
        ):
            report.pop(stale, None)
        atomic._write_json(self.output_dir / "continuous_mixture_contract.json", report)
        return report

    def _features(
        self,
        arm: str,
        normalized_state: Any,
        current_voltage: Any,
        initial_voltage: Any,
        raw_delta: Any,
        baseline_delta: Any,
        context: Any,
    ) -> Any:
        batch = current_voltage.shape[0]
        state = self._segment_state_tensor(normalized_state)
        presence = atomic.torch.as_tensor(
            self.semantic_presence,
            dtype=current_voltage.dtype,
            device=current_voltage.device,
        )[None, :, :].expand(batch, -1, -1)
        static = atomic.torch.as_tensor(
            self.layout.segment_static,
            dtype=current_voltage.dtype,
            device=current_voltage.device,
        )[None, :, :].expand(batch, -1, -1)
        basic = atomic.torch.stack(
            (
                raw_delta / self.config.bridge_voltage_scale_mv,
                raw_delta.abs() / self.config.bridge_voltage_scale_mv,
                current_voltage / 100.0,
                baseline_delta / self.config.bridge_voltage_scale_mv,
                (current_voltage - initial_voltage) / 100.0,
            ),
            dim=-1,
        )
        rich = atomic.torch.cat((basic, state, presence, context, static), dim=-1)
        if self._arm_contract(arm)["input"] == "voltage_only":
            rich = atomic.torch.cat((basic, atomic.torch.zeros_like(rich[..., 5:])), dim=-1)
        return rich

    def _topology_tensors(
        self, arm: str, seed: int, dtype: Any, device: Any
    ) -> Tuple[Any, Any, Any]:
        parent_np = self._topology(arm, seed)
        children_np, mask_np = self._children(parent_np)
        return (
            atomic.torch.as_tensor(parent_np, dtype=atomic.torch.long, device=device),
            atomic.torch.as_tensor(children_np, dtype=atomic.torch.long, device=device),
            atomic.torch.as_tensor(mask_np, dtype=dtype, device=device),
        )

    def _mixture_loss_weight(self, target_delta: Any) -> Any:
        edges = atomic.torch.as_tensor(
            self.config.activity_edges_mv,
            dtype=target_delta.dtype,
            device=target_delta.device,
        )
        activity = atomic.torch.bucketize(target_delta.abs(), edges)
        region = atomic.torch.as_tensor(
            self.layout.segment_region_ids,
            dtype=atomic.torch.long,
            device=target_delta.device,
        )[None, :].expand_as(activity)
        weights = atomic.torch.as_tensor(
            self.joint_weights,
            dtype=target_delta.dtype,
            device=target_delta.device,
        )
        return weights[activity, region]

    @staticmethod
    def _safe_correlation(left: np.ndarray, right: np.ndarray) -> float:
        left = np.asarray(left, dtype=np.float64).reshape(-1)
        right = np.asarray(right, dtype=np.float64).reshape(-1)
        if np.std(left) < 1e-12 or np.std(right) < 1e-12:
            return 0.0
        return float(np.corrcoef(left, right)[0, 1])

    def _mixture_unroll(
        self,
        model: Any,
        arm: str,
        width: int,
        seed: int,
        batch: Mapping[str, Any],
        *,
        collect: bool,
    ) -> Tuple[Any, Dict[str, Any]]:
        pair = self.source_models[("full_feedback_scalar", seed)]
        current_state = batch["state_t"][:, 0]
        current_voltage = batch["voltage_t"][:, 0]
        initial_voltage = current_voltage
        hidden = atomic.torch.zeros(
            current_voltage.shape[0],
            self.layout.segment_count,
            width,
            dtype=current_voltage.dtype,
            device=current_voltage.device,
        )
        state_center = atomic.torch.as_tensor(
            self.statistics["state_center"], device=current_state.device
        )
        state_scale = atomic.torch.as_tensor(
            self.statistics["state_scale"], device=current_state.device
        )
        delta_scale = atomic.torch.as_tensor(
            self.statistics["delta_scale"], device=current_state.device
        )
        parent, children, child_mask = self._topology_tensors(
            arm, seed, current_voltage.dtype, current_voltage.device
        )
        contract = self._arm_contract(arm)
        topology = (
            "tree"
            if contract["topology"] in ("authentic_tree", "relabelled_tree")
            else contract["topology"]
        )
        losses = []
        outputs: Dict[str, Any] = {}
        previous_alpha = None
        for step in range(self.config.objective_unroll_horizon_ms):
            context = atomic.torch.cat(
                (batch["drive"][:, step], batch["held_ions"]), dim=-1
            )
            normalized = (current_state - state_center) / state_scale
            raw = self._bridge_forward(pair[0], normalized, current_voltage, context)
            baseline = raw * self._static_gain(seed, raw, current_voltage)
            features = self._features(
                arm,
                normalized,
                current_voltage,
                initial_voltage,
                raw,
                baseline,
                context,
            )
            alpha, next_hidden = model(
                features,
                atomic.torch.as_tensor(
                    self.layout.segment_region_ids,
                    dtype=atomic.torch.long,
                    device=current_voltage.device,
                ),
                hidden,
                parent,
                children,
                child_mask,
                recurrent=bool(contract["recurrent"]),
                topology=topology,
            )
            voltage_delta = baseline * alpha
            next_voltage = current_voltage + voltage_delta
            state_delta = self._state_forward(
                pair[1], normalized, current_voltage, voltage_delta, context
            )
            next_state = current_state + state_delta * delta_scale
            target_voltage = batch["voltage_t1"][:, step]
            target_state = batch["state_t1"][:, step]
            target_delta = target_voltage - batch["voltage_t"][:, step]
            voltage_error = (
                next_voltage - target_voltage
            ) / self.config.bridge_voltage_scale_mv
            voltage_point = atomic.torch_functional.smooth_l1_loss(
                voltage_error,
                atomic.torch.zeros_like(voltage_error),
                reduction="none",
            )
            voltage_loss = atomic.torch.mean(
                self._mixture_loss_weight(target_delta) * voltage_point
            )
            state_error = (next_state - target_state) / state_scale
            state_loss = atomic.torch_functional.smooth_l1_loss(
                state_error,
                atomic.torch.zeros_like(state_error),
            )
            required = target_voltage - current_voltage
            oracle_alpha = atomic.torch.clamp(
                required * baseline / (baseline * baseline + 1e-8), 0.0, 1.0
            ).detach()
            oracle_loss = atomic.torch.mean((alpha - oracle_alpha) ** 2)
            high = atomic.torch.relu(
                (next_voltage - self.config.physical_voltage_maximum_mv)
                / self.config.bridge_voltage_scale_mv
            )
            low = atomic.torch.relu(
                (self.config.physical_voltage_minimum_mv - next_voltage)
                / self.config.bridge_voltage_scale_mv
            )
            physical = atomic.torch.mean(high * high + low * low)
            drift = atomic.torch.mean(atomic.torch.mean(voltage_error, dim=1) ** 2)
            auxiliary_weight = (
                self.config.mixture_oracle_auxiliary_weight
                if contract["oracle_auxiliary"]
                else 0.0
            )
            losses.append(
                voltage_loss
                + self.config.mixture_STATE_loss_weight * state_loss
                + auxiliary_weight * oracle_loss
                + self.config.mixture_physical_penalty_weight * physical
                + self.config.mixture_drift_penalty_weight * drift
            )
            if collect and step + 1 in self.config.rollout_horizons_ms:
                hidden_change = atomic.torch.mean((next_hidden - hidden) ** 2).sqrt()
                alpha_change = (
                    atomic.torch.zeros((), device=alpha.device)
                    if previous_alpha is None
                    else atomic.torch.mean((alpha - previous_alpha) ** 2).sqrt()
                )
                outputs[f"{step + 1}_ms"] = {
                    "state": next_state,
                    "voltage": next_voltage,
                    "alpha": alpha,
                    "oracle_alpha": oracle_alpha,
                    "hidden": next_hidden,
                    "hidden_change_rms": hidden_change,
                    "alpha_change_rms": alpha_change,
                }
            hidden = next_hidden if contract["recurrent"] else atomic.torch.zeros_like(next_hidden)
            previous_alpha = alpha
            current_state, current_voltage = next_state, next_voltage
        return atomic.torch.stack(losses).mean(), outputs

    def _evaluate_model(
        self, model: Any, arm: str, width: int, seed: int, role: str, device: Any
    ) -> Dict[str, Any]:
        rows = np.arange(len(self.window_data[role]["indices"]), dtype=np.int64)
        batch = self._batch_tensors(role, rows, device)
        was_training = bool(model.training)
        model.eval()
        with atomic.torch.no_grad():
            _, outputs = self._mixture_unroll(
                model, arm, width, seed, batch, collect=True
            )
        result: Dict[str, Any] = {"horizons": {}}
        for horizon, output in outputs.items():
            step = int(horizon[:-3]) - 1
            metrics = self._metric(
                output["state"], output["voltage"], batch, step
            )
            alpha = output["alpha"].cpu().numpy()
            oracle = output["oracle_alpha"].cpu().numpy()
            metrics["alpha_mean"] = float(np.mean(alpha))
            metrics["alpha_standard_deviation"] = float(np.std(alpha))
            metrics["alpha_near_zero_fraction"] = float(np.mean(alpha <= 0.05))
            metrics["alpha_near_one_fraction"] = float(np.mean(alpha >= 0.95))
            metrics["oracle_alpha_rmse"] = float(np.sqrt(np.mean((alpha - oracle) ** 2)))
            metrics["oracle_alpha_correlation"] = self._safe_correlation(alpha, oracle)
            metrics["hidden_norm_mean"] = float(
                output["hidden"].norm(dim=-1).mean().cpu()
            )
            metrics["hidden_change_rms"] = float(output["hidden_change_rms"].cpu())
            metrics["alpha_change_rms"] = float(output["alpha_change_rms"].cpu())
            result["horizons"][horizon] = metrics
        endpoint = outputs["8_ms"]
        row = {
            "voltage": endpoint["voltage"].cpu().numpy(),
            "target_voltage": batch["voltage_t1"][:, 7].cpu().numpy(),
            "initial_voltage": batch["voltage_t"][:, 0].cpu().numpy(),
        }
        result["activity_at_8ms"] = self._activity_metrics(row)
        result["region_at_8ms"] = self._region_metrics(row)
        model.train(was_training)
        return result

    def _save_checkpoint(
        self,
        model: Any,
        arm: str,
        width: int,
        seed: int,
        step: int,
        device: Any,
    ) -> Dict[str, Any]:
        path = self.output_dir / f"mixture_{arm}_w{width}_seed{seed}_step{step}.pt"
        topology = self._topology(arm, seed)
        payload = {
            "arm": arm,
            "width": width,
            "seed": seed,
            "step": step,
            "state_dict": {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            },
            "topology_sha256": hashlib.sha256(topology.tobytes()).hexdigest(),
            "configuration": asdict(self.config),
        }
        atomic.torch.save(payload, path)
        return {
            "step": step,
            "checkpoint": path.name,
            "checkpoint_sha256": atomic._sha256_file(path),
            "calibration": self._evaluate_model(
                model, arm, width, seed, "calibration", device
            ),
        }

    def train_synchronized_mixture_matrix(self) -> Dict[str, Any]:
        device = next(iter(self.frozen_state_models.values())).proposal.weight.device
        specs = self._run_specs()
        reports: Dict[str, Any] = {}
        for seed in self.config.pilot_seeds:
            base_states = {}
            initialization_hashes = {}
            for width in self.config.mixture_hidden_widths:
                atomic.torch.manual_seed(seed + 640000 + width)
                base = self._new_mixture_model(width, device)
                base_states[width] = copy.deepcopy(base.state_dict())
                initialization_hashes[str(width)] = self._state_dict_sha256(
                    base_states[width]
                )
            models = {}
            optimizers = {}
            for arm, width in specs:
                model = self._new_mixture_model(width, device)
                model.load_state_dict(copy.deepcopy(base_states[width]))
                model.train()
                models[(arm, width)] = model
                optimizers[(arm, width)] = atomic.torch.optim.AdamW(
                    model.parameters(),
                    lr=self.config.mixture_learning_rate,
                    weight_decay=self.config.mixture_weight_decay,
                )
            seed_report = {
                f"{arm}|w{width}": [
                    self._save_checkpoint(model, arm, width, seed, 0, device)
                ]
                for (arm, width), model in models.items()
            }
            rng = np.random.default_rng(seed + 641000)
            batch_stream_digest = hashlib.sha256()
            progress = atomic._CompactProgress(
                f"06b-m synchronized mixture seed={seed}",
                self.config.mixture_training_steps,
                max(1, self.config.mixture_training_steps // 25),
            )
            for step in range(1, self.config.mixture_training_steps + 1):
                rows = rng.choice(
                    len(self.window_data["fit"]["indices"]),
                    size=self.config.mixture_batch_window_count,
                    replace=False,
                )
                batch_stream_digest.update(
                    np.asarray(rows, dtype=np.int64).tobytes()
                )
                batch = self._batch_tensors("fit", rows, device)
                losses = []
                for (arm, width), model in models.items():
                    optimizer = optimizers[(arm, width)]
                    optimizer.zero_grad(set_to_none=True)
                    loss, _ = self._mixture_unroll(
                        model, arm, width, seed, batch, collect=False
                    )
                    if not bool(atomic.torch.isfinite(loss)):
                        raise RuntimeError(
                            f"non-finite 06b-m loss: seed={seed} step={step} "
                            f"arm={arm} width={width}"
                        )
                    loss.backward()
                    gradient_norm = atomic.torch.nn.utils.clip_grad_norm_(
                        model.parameters(), self.config.mixture_gradient_clip_norm
                    )
                    if not bool(atomic.torch.isfinite(gradient_norm)):
                        raise RuntimeError(
                            f"non-finite 06b-m gradient: seed={seed} step={step} "
                            f"arm={arm} width={width}"
                        )
                    optimizer.step()
                    losses.append(float(loss.detach().cpu()))
                if step in self.config.mixture_checkpoints:
                    for (arm, width), model in models.items():
                        seed_report[f"{arm}|w{width}"].append(
                            self._save_checkpoint(
                                model, arm, width, seed, step, device
                            )
                        )
                progress.update(
                    step,
                    "loss[min/median/max]="
                    f"{min(losses):.3g}/{float(np.median(losses)):.3g}/{max(losses):.3g}",
                )
            for (arm, width), model in models.items():
                model.eval()
                self.mixture_models[(arm, width, seed)] = model
            reports[str(seed)] = {
                "initialization_sha256_by_width": initialization_hashes,
                "batch_stream_sha256": batch_stream_digest.hexdigest(),
                "runs": seed_report,
            }
            atomic._write_json(
                self.output_dir / f"mixture_training_seed{seed}.json",
                reports[str(seed)],
            )
        expected_checkpoints = len(self.config.mixture_checkpoints)
        report = {
            "schema_version": "06b-m-synchronized-training-v1",
            "valid": all(
                len(rows) == expected_checkpoints
                for seed_rows in reports.values()
                for rows in seed_rows["runs"].values()
            ),
            "device": str(device),
            "run_specs": [f"{arm}|w{width}" for arm, width in specs],
            "same_minibatch_stream_within_seed": True,
            "same_initialization_within_seed_and_width": True,
            "teacher_endpoint_used_as_input": False,
            "teacher_optimal_blend_used_only_as_training_target": True,
            "development_used_during_training": False,
            "reports": reports,
        }
        self.training_valid = bool(report["valid"])
        atomic._write_json(self.output_dir / "mixture_training_report.json", report)
        return report

    def evaluate_mixture_matrix(self) -> Dict[str, Any]:
        if not self.training_valid:
            raise RuntimeError("06b-m synchronized training is not complete")
        device = next(iter(self.frozen_state_models.values())).proposal.weight.device
        specs = self._run_specs()
        per_seed: Dict[str, Any] = {}
        progress = atomic._CompactProgress(
            "06b-m development evaluation",
            len(self.config.pilot_seeds) * (len(specs) + 2),
            1,
        )
        completed = 0
        for seed in self.config.pilot_seeds:
            rows = {
                STATIC_REFERENCE: self._recursive_gate_evaluation(
                    seed, STATIC_REFERENCE, "development", device
                ),
                TEACHER_OPTIMAL_BLEND_ORACLE: self._recursive_gate_evaluation(
                    seed, TEACHER_OPTIMAL_BLEND_ORACLE, "development", device
                ),
            }
            completed += 2
            progress.update(completed, f"seed={seed} frozen references")
            for arm, width in specs:
                key = f"{arm}|w{width}"
                rows[key] = self._evaluate_model(
                    self.mixture_models[(arm, width, seed)],
                    arm,
                    width,
                    seed,
                    "development",
                    device,
                )
                completed += 1
                progress.update(completed, f"seed={seed} {key}")
            per_seed[str(seed)] = rows
        valid = all(
            row["horizons"]["8_ms"]["nonfinite_voltage_count"] == 0
            and row["horizons"]["8_ms"]["nonfinite_state_count"] == 0
            for seed_rows in per_seed.values()
            for row in seed_rows.values()
        )
        report = {
            "schema_version": "06b-m-development-v1",
            "valid": bool(valid),
            "role": "historically_reused_train_development",
            "new_independent_confirmation_claimed": False,
            "validation_state_accessed": False,
            "test_state_accessed": False,
            "per_seed": per_seed,
        }
        atomic._write_json(self.output_dir / "mixture_development.json", report)
        return report

    @staticmethod
    def _median(values: Sequence[float]) -> float:
        return float(np.median(np.asarray(values, dtype=np.float64)))

    def _summary(self, key: str, evaluation: Mapping[str, Any]) -> Dict[str, Any]:
        rows = [seed[key] for seed in evaluation["per_seed"].values()]
        static_rows = [
            seed[STATIC_REFERENCE] for seed in evaluation["per_seed"].values()
        ]
        endpoint = [row["horizons"]["8_ms"] for row in rows]
        static_endpoint = [row["horizons"]["8_ms"] for row in static_rows]
        return {
            "median_recursive_gain_over_static_fraction": self._median(
                [
                    1 - row["voltage_rmse_mv"] / max(reference["voltage_rmse_mv"], 1e-12)
                    for row, reference in zip(endpoint, static_endpoint)
                ]
            ),
            "median_voltage_gain_vs_persistence_fraction": self._median(
                [row["voltage_improvement_vs_persistence_fraction"] for row in endpoint]
            ),
            "median_STATE_gain_vs_persistence_fraction": self._median(
                [row["state_improvement_vs_persistence_fraction"] for row in endpoint]
            ),
            "minimum_seed_STATE_gain_vs_persistence_fraction": float(
                min(row["state_improvement_vs_persistence_fraction"] for row in endpoint)
            ),
            "median_oracle_alpha_correlation": self._median(
                [row.get("oracle_alpha_correlation", math.nan) for row in endpoint]
            ),
            "median_oracle_alpha_rmse": self._median(
                [row.get("oracle_alpha_rmse", math.nan) for row in endpoint]
            ),
            "median_alpha_standard_deviation": self._median(
                [row.get("alpha_standard_deviation", math.nan) for row in endpoint]
            ),
            "activity_gain_vs_persistence": {
                name: self._median(
                    [
                        row["activity_at_8ms"][name][
                            "voltage_gain_vs_persistence_fraction"
                        ]
                        for row in rows
                    ]
                )
                for name in rows[0]["activity_at_8ms"]
            },
            "region_gain_vs_persistence": {
                name: self._median(
                    [
                        row["region_at_8ms"][name][
                            "voltage_gain_vs_persistence_fraction"
                        ]
                        for row in rows
                    ]
                )
                for name in rows[0]["region_at_8ms"]
            },
            "all_seed_voltage_gain_positive": all(
                row["voltage_improvement_vs_persistence_fraction"] > 0
                for row in endpoint
            ),
            "physical_voltage_violation_count": int(
                sum(row["physical_voltage_violation_count"] for row in endpoint)
            ),
        }

    def _paired_gain(
        self, better: str, reference: str, evaluation: Mapping[str, Any]
    ) -> float:
        values = []
        for rows in evaluation["per_seed"].values():
            candidate = rows[better]["horizons"]["8_ms"]["voltage_rmse_mv"]
            baseline = rows[reference]["horizons"]["8_ms"]["voltage_rmse_mv"]
            values.append(1 - candidate / max(baseline, 1e-12))
        return self._median(values)

    def finalize_continuous_mixture_state(
        self, evaluation: Mapping[str, Any]
    ) -> Dict[str, Any]:
        specs = self._run_specs()
        summaries = {
            f"{arm}|w{width}": self._summary(
                f"{arm}|w{width}", evaluation
            )
            for arm, width in specs
        }
        width = max(self.config.mixture_hidden_widths)
        matrix_width = self.config.full_matrix_hidden_width
        primary_key = f"{TREE_RECURRENT}|w{width}"
        local_key = f"{LOCAL_RECURRENT}|w{width}"
        shuffled_key = f"{SHUFFLED_TREE_RECURRENT}|w{width}"
        contrasts = {
            "physiology_over_voltage_instantaneous": self._paired_gain(
                f"{PHYSIOLOGY_INSTANTANEOUS}|w{matrix_width}",
                f"{VOLTAGE_INSTANTANEOUS}|w{matrix_width}",
                evaluation,
            ),
            "local_recurrence_over_physiology_instantaneous": self._paired_gain(
                f"{LOCAL_RECURRENT}|w{matrix_width}",
                f"{PHYSIOLOGY_INSTANTANEOUS}|w{matrix_width}",
                evaluation,
            ),
            "authentic_tree_over_local": self._paired_gain(
                primary_key, local_key, evaluation
            ),
            "authentic_tree_over_shuffled_tree": self._paired_gain(
                primary_key, shuffled_key, evaluation
            ),
            "oracle_auxiliary_over_rollout_only": self._paired_gain(
                f"{TREE_RECURRENT}|w{matrix_width}",
                f"{TREE_RECURRENT_NO_ORACLE_AUX}|w{matrix_width}",
                evaluation,
            ),
            "tree_width_max_over_min": self._paired_gain(
                primary_key,
                f"{TREE_RECURRENT}|w{min(self.config.mixture_hidden_widths)}",
                evaluation,
            ),
        }

        def absolute_pass(key: str) -> bool:
            row = summaries[key]
            return bool(
                row["median_recursive_gain_over_static_fraction"]
                >= self.config.minimum_recursive_gain_over_static_fraction
                and row["all_seed_voltage_gain_positive"]
                and row["minimum_seed_STATE_gain_vs_persistence_fraction"]
                > self.config.minimum_STATE_gain_fraction
                and row["activity_gain_vs_persistence"]["active_ge_5mV"]
                >= self.config.minimum_active_gain_fraction
                and row["activity_gain_vs_persistence"]["moderate_1_to_5mV"]
                >= self.config.minimum_moderate_gain_fraction
                and row["activity_gain_vs_persistence"]["quiescent_lt_1mV"]
                >= self.config.minimum_quiescent_gain_fraction
                and row["region_gain_vs_persistence"]["soma"]
                >= self.config.minimum_soma_gain_fraction
                and row["physical_voltage_violation_count"] == 0
            )

        primary_pass = bool(
            absolute_pass(primary_key)
            and contrasts["authentic_tree_over_local"]
            >= self.config.minimum_topology_gain_fraction
            and contrasts["authentic_tree_over_shuffled_tree"]
            >= self.config.minimum_topology_gain_fraction
            and summaries[primary_key]["median_oracle_alpha_correlation"]
            >= self.config.minimum_oracle_alpha_correlation
        )
        local_pass = bool(
            absolute_pass(local_key)
            and contrasts["local_recurrence_over_physiology_instantaneous"]
            >= self.config.minimum_memory_gain_fraction
            and summaries[local_key]["median_oracle_alpha_correlation"]
            >= self.config.minimum_oracle_alpha_correlation
        )
        instantaneous_key = f"{PHYSIOLOGY_INSTANTANEOUS}|w{matrix_width}"
        instantaneous_pass = bool(
            absolute_pass(instantaneous_key)
            and contrasts["physiology_over_voltage_instantaneous"]
            >= self.config.minimum_observability_gain_fraction
            and summaries[instantaneous_key]["median_oracle_alpha_correlation"]
            >= self.config.minimum_oracle_alpha_correlation
        )
        if primary_pass:
            diagnosis = "TOPOLOGY_AWARE_CONTINUOUS_MIXTURE_STATE_IDENTIFIED"
            selected = primary_key
            next_step = "fresh_train_support_continuous_mixture_confirmation"
        elif local_pass:
            diagnosis = "LOCAL_CONTINUOUS_MIXTURE_STATE_IDENTIFIED"
            selected = local_key
            next_step = "fresh_train_support_local_mixture_confirmation"
        elif instantaneous_pass:
            diagnosis = "INSTANTANEOUS_PHYSIOLOGY_MIXTURE_IDENTIFIED"
            selected = instantaneous_key
            next_step = "fresh_train_support_instantaneous_mixture_confirmation"
        else:
            selected = None
            best_correlation = max(
                row["median_oracle_alpha_correlation"] for row in summaries.values()
            )
            if best_correlation >= self.config.minimum_oracle_alpha_correlation:
                diagnosis = "MIXTURE_TARGET_LEARNABLE_BUT_RECURSIVE_COMPOSITION_FAILS"
                next_step = "continuous_mixture_objective_and_coupling_revision"
            else:
                diagnosis = "CAUSAL_MIXTURE_STATE_NOT_OBSERVABLE_FROM_CURRENT_CONTRACT"
                next_step = "synaptic_and_hidden_state_observability_revision"
        report = {
            "schema_version": "06b-m-final-report-v1",
            "valid": bool(self.training_valid and evaluation.get("valid")),
            "component_playground_grade": True,
            "architecture_revision_executed": True,
            "diagnosis": diagnosis,
            "selected_candidate": selected,
            "primary_passed": primary_pass,
            "local_fallback_passed": local_pass,
            "instantaneous_fallback_passed": instantaneous_pass,
            "summaries": summaries,
            "factorial_contrasts": contrasts,
            "new_independent_confirmation_claimed": False,
            "teacher_endpoint_used_as_input": False,
            "teacher_optimal_blend_used_only_as_training_target": True,
            "validation_state_accessed": False,
            "test_state_accessed": False,
            "coupled_06c_canary_authorized": False,
            "full_training_authorized": False,
            "mass_dataset_generation_authorized": False,
            "fresh_train_support_confirmation_authorized": bool(selected),
            "next_step": next_step,
            "code_revision": self.code_revision,
        }
        atomic._write_json(self.output_dir / "final_report.json", report)
        self._write_artifact_index()
        return report


__all__ = [
    "ContinuousMixtureCell",
    "ContinuousMixtureStateConfig",
    "ContinuousMixtureStatePlayground",
    "EXPECTED_06BL_ARCHIVE_SHA256",
    "EXPECTED_06BL_FINAL_SHA256",
    "EXPECTED_06BL_INDEX_SHA256",
    "FULL_MATRIX_ARMS",
    "LOCAL_RECURRENT",
    "PHYSIOLOGY_INSTANTANEOUS",
    "SCALING_ARMS",
    "SHUFFLED_TREE_RECURRENT",
    "TREE_RECURRENT",
    "TREE_RECURRENT_NO_ORACLE_AUX",
    "VOLTAGE_INSTANTANEOUS",
    "verified_06bl_artifact_root",
]
