"""06b-n: aligned objective/coupling/state-structure forensic.

The experiment is deliberately train-only and multifactorial.  It reuses the
same windows, frozen voltage expert, random seeds and minibatch stream to test
four independent interventions in one notebook:

* neutral versus carry-first gate initialization;
* ordinary recursive loss versus positive regret over persistence;
* state update after voltage mixing versus a mixture of complete state flows;
* the registered generic STATE updater versus a bounded relaxation updater.

Frozen counterfactual sweeps and a teacher-microtrace upper bound precede the
training matrix.  They are diagnostics and are never eligible for selection.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from . import atomic_state_dynamics_playground as atomic
from .continuous_mixture_state_playground import (
    LOCAL_RECURRENT,
    PHYSIOLOGY_INSTANTANEOUS,
    ContinuousMixtureStateConfig,
    ContinuousMixtureStatePlayground,
)
from .voltage_error_model_revision import STATIC_REFERENCE


EXPECTED_06BM_ARCHIVE_SHA256 = (
    "9b97c3b4a465f376f98b97a9408f7accd0268a0681b8be8c432f40327f97bfee"
)
EXPECTED_06BM_INDEX_SHA256 = (
    "319bacdeece4749407643816977550b3b326694f787503bb50dcc3d3e1ef73f6"
)
EXPECTED_06BM_FINAL_SHA256 = (
    "5f35dc5e4a4d3b308d8c1ccaeed2cba4da780ced201a42f9ee36640bb5c1628b"
)

PRE_MIXED_STATE = "pre_mixed_state"
ENDPOINT_MIXED_STATE = "endpoint_mixed_state"
STANDARD_ROLLOUT = "standard_rollout"
PERSISTENCE_REGRET = "persistence_regret"
GENERIC_STATE = "generic_state"
RELAXATION_STATE = "relaxation_state"


def verified_06bm_artifact_root(
    source: Path, cache_dir: Path
) -> Tuple[Path, Dict[str, Any]]:
    """Verify the exact registered 06b-m artifact and every indexed member."""

    source = Path(source).expanduser().resolve()
    cache_dir = Path(cache_dir).resolve()
    if not source.exists():
        raise FileNotFoundError(source)
    archive_hash = "extracted-directory"
    if source.is_file():
        if source.suffix.lower() != ".zip":
            raise RuntimeError("06b-m source must be a ZIP or extracted directory")
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
        if atomic._sha256_file(path) == EXPECTED_06BM_INDEX_SHA256
    ]
    if len(matches) != 1:
        raise RuntimeError(f"expected one exact 06b-m artifact; found {len(matches)}")
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
        raise RuntimeError(f"06b-m indexed member verification failed: {failures}")
    final_path = root / "final_report.json"
    if atomic._sha256_file(final_path) != EXPECTED_06BM_FINAL_SHA256:
        raise RuntimeError("06b-m final report SHA-256 mismatch")
    final = json.loads(final_path.read_text(encoding="utf-8"))
    if (
        final.get("diagnosis")
        != "MIXTURE_TARGET_LEARNABLE_BUT_RECURSIVE_COMPOSITION_FAILS"
        or final.get("next_step")
        != "continuous_mixture_objective_and_coupling_revision"
        or final.get("coupled_06c_canary_authorized") is not False
    ):
        raise RuntimeError("06b-m result does not authorize 06b-n")
    if source.is_file() and archive_hash != EXPECTED_06BM_ARCHIVE_SHA256:
        archive_hash = "kaggle-repacked"
    return root, {
        "valid": True,
        "source_kind": "zip" if source.is_file() else "directory",
        "archive_sha256": archive_hash,
        "artifact_index_sha256": EXPECTED_06BM_INDEX_SHA256,
        "final_report_sha256": EXPECTED_06BM_FINAL_SHA256,
        "indexed_member_count": len(index.get("artifacts", [])),
        "diagnosis": final["diagnosis"],
        "next_step": final["next_step"],
    }


def audit_cnexp_teacher_contract(teacher_root: Path, mechanism_names: Sequence[str]) -> Dict[str, Any]:
    """Audit, without pretending, whether exact cnexp replay is data-identifiable."""

    teacher_root = Path(teacher_root)
    mod_files = sorted(teacher_root.rglob("*.mod")) if teacher_root.is_dir() else []
    rows = []
    for path in mod_files:
        text = path.read_text(encoding="utf-8", errors="replace")
        if not re.search(r"\bMETHOD\s+cnexp\b", text, flags=re.IGNORECASE):
            continue
        suffix = re.search(r"\b(?:SUFFIX|POINT_PROCESS)\s+([A-Za-z0-9_]+)", text)
        derivatives = re.findall(r"^\s*([A-Za-z0-9_]+)'\s*=", text, flags=re.MULTILINE)
        rows.append(
            {
                "file": path.name,
                "mechanism": suffix.group(1) if suffix else path.stem,
                "derivative_states": sorted(set(derivatives)),
                "sha256": atomic._sha256_file(path),
            }
        )
    dataset_names = {str(name) for name in mechanism_names}
    matched = sorted({row["mechanism"] for row in rows} & dataset_names)
    blockers = []
    if not rows:
        blockers.append("canonical teacher .mod files with METHOD cnexp were not found")
    blockers.extend(
        [
            "the transition schema does not snapshot every per-instance kinetic parameter needed to re-evaluate all .mod rate functions",
            "an independently verified Python implementation of every teacher rate function is not registered",
        ]
    )
    return {
        "schema_version": "06b-n-cnexp-audit-v1",
        "teacher_root": str(teacher_root),
        "mod_file_count": len(mod_files),
        "cnexp_file_count": len(rows),
        "cnexp_mechanisms": rows,
        "dataset_mechanism_match_count": len(matched),
        "dataset_mechanism_matches": matched,
        "exact_cnexp_replay_executed": False,
        "exact_cnexp_replay_eligible": False,
        "blockers": blockers,
        "fallback_executed": "bounded_learned_relaxation_plus_teacher_microtrace_upper_bound",
    }


@dataclass(frozen=True)
class StructurePreservingCouplingConfig(ContinuousMixtureStateConfig):
    frozen_gate_priors: Tuple[float, ...] = (0.01, 0.05, 0.10, 0.50)
    factor_gate_priors: Tuple[float, ...] = (0.02, 0.50)
    factor_objectives: Tuple[str, ...] = (STANDARD_ROLLOUT, PERSISTENCE_REGRET)
    factor_couplings: Tuple[str, ...] = (PRE_MIXED_STATE, ENDPOINT_MIXED_STATE)
    factor_state_updaters: Tuple[str, ...] = (GENERIC_STATE, RELAXATION_STATE)
    factor_hidden_width: int = 16
    factor_training_steps: int = 200
    factor_checkpoints: Tuple[int, ...] = (0, 50, 100, 200)
    factor_batch_window_count: int = 4
    factor_learning_rate: float = 3e-4
    factor_weight_decay: float = 1e-5
    factor_gradient_clip_norm: float = 1.0
    persistence_regret_weight: float = 0.5
    persistence_regret_temperature: float = 0.05
    persistence_regret_margin: float = 0.0
    quiescent_regret_multiplier: float = 4.0
    quiescent_threshold_mv: float = 1.0
    relaxation_hidden_width: int = 20
    relaxation_training_steps: int = 400
    relaxation_checkpoints: Tuple[int, ...] = (0, 100, 200, 400)
    relaxation_learning_rate: float = 3e-4
    relaxation_weight_decay: float = 1e-5
    relaxation_gradient_clip_norm: float = 1.0
    relaxation_embedding_width: int = 4
    maximum_relaxation_parameter_count: int = 12000
    minimum_material_factor_gain: float = 0.01

    def validate(self) -> None:
        super().validate()
        for priors in (self.frozen_gate_priors, self.factor_gate_priors):
            if not priors or any(not 0 < value < 1 for value in priors):
                raise ValueError("06b-n gate priors must lie strictly inside (0,1)")
        if tuple(self.factor_objectives) != (STANDARD_ROLLOUT, PERSISTENCE_REGRET):
            raise ValueError("06b-n objective axis changed")
        if tuple(self.factor_couplings) != (PRE_MIXED_STATE, ENDPOINT_MIXED_STATE):
            raise ValueError("06b-n coupling axis changed")
        if tuple(self.factor_state_updaters) != (GENERIC_STATE, RELAXATION_STATE):
            raise ValueError("06b-n STATE-updater axis changed")
        if self.factor_checkpoints[0] != 0 or self.factor_checkpoints[-1] != self.factor_training_steps:
            raise ValueError("06b-n factor checkpoints must span the training trajectory")
        if self.relaxation_checkpoints[0] != 0 or self.relaxation_checkpoints[-1] != self.relaxation_training_steps:
            raise ValueError("06b-n relaxation checkpoints must span training")
        positive = (
            self.factor_hidden_width,
            self.factor_training_steps,
            self.factor_batch_window_count,
            self.factor_learning_rate,
            self.factor_gradient_clip_norm,
            self.persistence_regret_weight,
            self.persistence_regret_temperature,
            self.quiescent_regret_multiplier,
            self.quiescent_threshold_mv,
            self.relaxation_hidden_width,
            self.relaxation_training_steps,
            self.relaxation_learning_rate,
            self.relaxation_gradient_clip_norm,
            self.maximum_relaxation_parameter_count,
            self.minimum_material_factor_gain,
        )
        if any(value <= 0 for value in positive):
            raise ValueError("06b-n positive configuration value is invalid")

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "StructurePreservingCouplingConfig":
        payload = dict(values)
        integer_tuples = (
            "rollout_horizons_ms",
            "voltage_path_sample_indices",
            "pilot_seeds",
            "scaling_checkpoints",
            "repair_checkpoints",
            "scheduled_checkpoints",
            "objective_checkpoints",
            "mixture_hidden_widths",
            "mixture_checkpoints",
            "factor_checkpoints",
            "relaxation_checkpoints",
        )
        float_tuples = (
            "voltage_shrinkage_grid",
            "activity_edges_mv",
            "analytic_shrinkage_strengths",
            "analytic_voltage_edges_mv",
            "temporal_ridge_strengths",
            "gate_ridge_strengths",
            "hurdle_probability_thresholds",
            "frozen_gate_priors",
            "factor_gate_priors",
        )
        for name in integer_tuples:
            if name in payload:
                payload[name] = tuple(map(int, payload[name]))
        for name in float_tuples:
            if name in payload:
                payload[name] = tuple(map(float, payload[name]))
        for name in ("factor_objectives", "factor_couplings", "factor_state_updaters"):
            if name in payload:
                payload[name] = tuple(map(str, payload[name]))
        result = cls(**payload)
        result.validate()
        return result


if atomic.nn is not None:

    class BoundedRelaxationStateUpdater(atomic.nn.Module):
        """Rush-Larsen-shaped updater in the canonical bounded STATE domain."""

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
        ) -> None:
            super().__init__()
            self.mechanism_embedding = atomic.nn.Embedding(mechanism_count, embedding_width)
            self.variable_embedding = atomic.nn.Embedding(variable_count, embedding_width)
            self.kind_embedding = atomic.nn.Embedding(kind_count, embedding_width)
            self.region_embedding = atomic.nn.Embedding(region_count, embedding_width)
            width = path_width + static_width + drive_width + 4 * embedding_width
            self.encoder = atomic.nn.Sequential(
                atomic.nn.Linear(width, hidden_width),
                atomic.nn.SiLU(),
                atomic.nn.Linear(hidden_width, hidden_width),
                atomic.nn.SiLU(),
            )
            self.equilibrium_logit = atomic.nn.Linear(hidden_width, 1)
            self.log_rate = atomic.nn.Linear(hidden_width, 1)
            atomic.nn.init.zeros_(self.equilibrium_logit.weight)
            atomic.nn.init.zeros_(self.equilibrium_logit.bias)
            atomic.nn.init.zeros_(self.log_rate.weight)
            atomic.nn.init.constant_(self.log_rate.bias, -6.0)

        def forward(
            self,
            voltage_path: Any,
            drive: Any,
            static: Any,
            mechanism_id: Any,
            variable_id: Any,
            kind_id: Any,
            region_id: Any,
        ) -> Tuple[Any, Any]:
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
                atomic.torch.cat((voltage_path / 100.0, drive, static, embedded), dim=-1)
            )
            equilibrium = atomic.torch.sigmoid(self.equilibrium_logit(hidden).squeeze(-1))
            rate = atomic.torch_functional.softplus(self.log_rate(hidden).squeeze(-1))
            return equilibrium, rate

else:  # pragma: no cover

    class BoundedRelaxationStateUpdater:
        def __init__(self, *_: Any, **__: Any) -> None:
            raise RuntimeError("06b-n requires PyTorch")


class StructurePreservingCouplingForensic(ContinuousMixtureStatePlayground):
    """One-run paired matrix for the authorized 06b-m continuation."""

    config: StructurePreservingCouplingConfig

    def __init__(
        self,
        bundle: Any,
        output_dir: Path,
        config: StructurePreservingCouplingConfig,
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
        artifact_06bm_source: Path,
        teacher_root: Path,
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
            artifact_06bl_source,
            code_revision=code_revision,
        )
        self.artifact_06bm_source = Path(artifact_06bm_source)
        self.teacher_root = Path(teacher_root)
        self.source_06bm_root: Optional[Path] = None
        self.frozen_controllers: Dict[Tuple[str, int, int], Any] = {}
        self.relaxation_models: Dict[int, Any] = {}
        self.factor_models: Dict[Tuple[str, int], Any] = {}
        self.factor_training_valid = False

    def _factor_specs(self) -> Tuple[Tuple[float, str, str, str], ...]:
        return tuple(
            (prior, objective, coupling, state)
            for prior in self.config.factor_gate_priors
            for objective in self.config.factor_objectives
            for coupling in self.config.factor_couplings
            for state in self.config.factor_state_updaters
        )

    @staticmethod
    def _factor_key(spec: Tuple[float, str, str, str]) -> str:
        prior, objective, coupling, state = spec
        return f"p{prior:.2f}|{objective}|{coupling}|{state}"

    def _new_relaxation_model(self, device: Any) -> Any:
        model = BoundedRelaxationStateUpdater(
            mechanism_count=len(self.layout.mechanism_names),
            variable_count=len(self.layout.variable_names),
            kind_count=len(self.layout.kind_names),
            region_count=len(self.layout.region_names),
            static_width=self.layout.segment_static.shape[1],
            drive_width=len(atomic.CAUSAL_DRIVE_FEATURES) + len(self.ion_feature_names),
            path_width=len(self.config.voltage_path_sample_indices),
            hidden_width=self.config.relaxation_hidden_width,
            embedding_width=self.config.relaxation_embedding_width,
        ).to(device)
        count = int(sum(value.numel() for value in model.parameters()))
        if count > self.config.maximum_relaxation_parameter_count:
            raise RuntimeError(
                f"06b-n relaxation updater has {count} parameters; ceiling is "
                f"{self.config.maximum_relaxation_parameter_count}"
            )
        return model

    def _load_frozen_controller(self, root: Path, arm: str, width: int, seed: int, device: Any) -> Any:
        path = root / f"mixture_{arm}_w{width}_seed{seed}_step400.pt"
        checkpoint = atomic.torch.load(path, map_location=device, weights_only=False)
        if (
            str(checkpoint.get("arm")) != arm
            or int(checkpoint.get("width", -1)) != width
            or int(checkpoint.get("seed", -1)) != seed
            or int(checkpoint.get("step", -1)) != 400
        ):
            raise RuntimeError(f"06b-m controller identity mismatch: {path.name}")
        model = self._new_mixture_model(width, device)
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        return model

    def prepare_structure_preserving_forensic(self) -> Dict[str, Any]:
        base = self.prepare_continuous_mixture_state()
        root, source = verified_06bm_artifact_root(
            self.artifact_06bm_source,
            self.output_dir.parent / ".06bn_artifact_cache" / "06bm",
        )
        self.source_06bm_root = root
        device = next(iter(self.frozen_state_models.values())).proposal.weight.device
        for seed in self.config.pilot_seeds:
            for arm, width in ((LOCAL_RECURRENT, 32), (PHYSIOLOGY_INSTANTANEOUS, 16)):
                self.frozen_controllers[(arm, width, seed)] = self._load_frozen_controller(
                    root, arm, width, seed, device
                )
        cnexp = audit_cnexp_teacher_contract(self.teacher_root, self.layout.mechanism_names)
        atomic._write_json(self.output_dir / "cnexp_teacher_contract.json", cnexp)
        relaxation_probe = self._new_relaxation_model(device)
        relaxation_count = int(sum(value.numel() for value in relaxation_probe.parameters()))
        report = {
            **base,
            "schema_version": "06b-n-structure-preserving-contract-v1",
            "experiment": "structure_preserving_coupling_forensic",
            "source_06bm": source,
            "frozen_counterfactual_gate_priors": list(self.config.frozen_gate_priors),
            "factor_axes": {
                "gate_prior": list(self.config.factor_gate_priors),
                "objective": list(self.config.factor_objectives),
                "coupling": list(self.config.factor_couplings),
                "state_updater": list(self.config.factor_state_updaters),
            },
            "factor_arm_count": len(self._factor_specs()),
            "factor_hidden_width": self.config.factor_hidden_width,
            "factor_checkpoints": list(self.config.factor_checkpoints),
            "relaxation_parameter_count": relaxation_count,
            "relaxation_parameter_ceiling": self.config.maximum_relaxation_parameter_count,
            "exact_cnexp_replay_eligible": cnexp["exact_cnexp_replay_eligible"],
            "teacher_microtrace_upper_bound_selectable": False,
            "same_initialization_except_registered_gate_bias": True,
            "same_minibatch_stream_within_seed": True,
            "same_frozen_voltage_expert": True,
            "validation_state_accessed": False,
            "test_state_accessed": False,
            "new_independent_confirmation_claimed": False,
            "coupled_06c_canary_authorized": False,
            "full_training_authorized": False,
        }
        atomic._write_json(self.output_dir / "structure_preserving_contract.json", report)
        return report

    @staticmethod
    def _shift_gate_prior(alpha: Any, prior: float) -> Any:
        eps = 1e-6
        clipped = alpha.clamp(eps, 1.0 - eps)
        shift = math.log(prior / (1.0 - prior))
        return atomic.torch.sigmoid(atomic.torch.logit(clipped) + shift)

    def _state_forward_path(
        self, model: Any, normalized_state: Any, voltage: Any, voltage_path: Any, context: Any
    ) -> Any:
        batch, coordinate_count = normalized_state.shape
        device = normalized_state.device
        # NumPy path constructions such as ceil/floor promote float32 arrays to
        # float64.  Keep the model boundary explicit so every causal and
        # privileged path matches the registered float32 checkpoint.
        voltage_path = atomic.torch.as_tensor(
            voltage_path, dtype=normalized_state.dtype, device=device
        )
        segments = atomic.torch.as_tensor(
            self.coordinate["segment"], dtype=atomic.torch.long, device=device
        )
        static = atomic.torch.as_tensor(
            self.layout.segment_static[self.coordinate["segment"]],
            dtype=normalized_state.dtype,
            device=device,
        )[None, :, :].expand(batch, -1, -1)
        ids = lambda name: atomic.torch.as_tensor(
            self.coordinate[name], dtype=atomic.torch.long, device=device
        )[None, :].expand(batch, -1).reshape(-1)
        prediction = model(
            normalized_state.reshape(-1),
            voltage[:, segments].reshape(-1),
            voltage_path[:, segments].reshape(batch * coordinate_count, -1),
            context[:, segments].reshape(batch * coordinate_count, -1),
            static.reshape(batch * coordinate_count, -1),
            ids("mechanism"),
            ids("variable"),
            ids("kind"),
            ids("region"),
        )
        return prediction.reshape(batch, coordinate_count)

    def _relaxation_state_forward(
        self,
        model: Any,
        normalized_state: Any,
        voltage: Any,
        voltage_delta: Any,
        context: Any,
        semantic_permutation: Optional[np.ndarray] = None,
    ) -> Any:
        batch, coordinate_count = normalized_state.shape
        device = normalized_state.device
        segments = atomic.torch.as_tensor(
            self.coordinate["segment"], dtype=atomic.torch.long, device=device
        )
        fractions = atomic.torch.as_tensor(
            np.asarray(self.config.voltage_path_sample_indices, dtype=np.float32)
            / float(self.config.expected_microtrace_sample_count - 1),
            device=device,
        )
        path = voltage_delta[:, segments, None] * fractions[None, None, :]
        static = atomic.torch.as_tensor(
            self.layout.segment_static[self.coordinate["segment"]],
            dtype=normalized_state.dtype,
            device=device,
        )[None, :, :].expand(batch, -1, -1)
        def ids(name: str) -> Any:
            values = self.coordinate[name]
            if semantic_permutation is not None and name in ("mechanism", "variable", "kind"):
                values = values[np.asarray(semantic_permutation, dtype=np.int64)]
            return atomic.torch.as_tensor(
                values, dtype=atomic.torch.long, device=device
            )[None, :].expand(batch, -1).reshape(-1)
        equilibrium, rate = model(
            path.reshape(batch * coordinate_count, -1),
            context[:, segments].reshape(batch * coordinate_count, -1),
            static.reshape(batch * coordinate_count, -1),
            ids("mechanism"),
            ids("variable"),
            ids("kind"),
            ids("region"),
        )
        state_center = atomic.torch.as_tensor(self.statistics["state_center"], device=device)
        state_scale = atomic.torch.as_tensor(self.statistics["state_scale"], device=device)
        delta_scale = atomic.torch.as_tensor(self.statistics["delta_scale"], device=device)
        current_logit = normalized_state * state_scale + state_center
        current = atomic.torch.sigmoid(current_logit).reshape(-1)
        decay = atomic.torch.exp(-rate.clamp(max=20.0))
        next_value = (equilibrium + (current - equilibrium) * decay).clamp(1e-6, 1.0 - 1e-6)
        next_logit = atomic.torch.logit(next_value).reshape(batch, coordinate_count)
        return (next_logit - current_logit) / delta_scale

    def _state_delta(
        self,
        state_kind: str,
        seed: int,
        normalized: Any,
        voltage: Any,
        voltage_delta: Any,
        context: Any,
    ) -> Any:
        if state_kind == GENERIC_STATE:
            model = self.source_models[("full_feedback_scalar", seed)][1]
            return self._state_forward(model, normalized, voltage, voltage_delta, context)
        if state_kind == RELAXATION_STATE:
            return self._relaxation_state_forward(
                self.relaxation_models[seed], normalized, voltage, voltage_delta, context
            )
        raise ValueError(state_kind)

    def _coupled_state_delta(
        self,
        coupling: str,
        state_kind: str,
        seed: int,
        normalized: Any,
        voltage: Any,
        baseline: Any,
        alpha: Any,
        context: Any,
    ) -> Any:
        if coupling == PRE_MIXED_STATE:
            return self._state_delta(
                state_kind, seed, normalized, voltage, baseline * alpha, context
            )
        if coupling == ENDPOINT_MIXED_STATE:
            carry = self._state_delta(
                state_kind, seed, normalized, voltage, atomic.torch.zeros_like(baseline), context
            )
            dynamic = self._state_delta(
                state_kind, seed, normalized, voltage, baseline, context
            )
            coordinate_alpha = alpha[:, atomic.torch.as_tensor(
                self.coordinate["segment"], dtype=atomic.torch.long, device=alpha.device
            )]
            return carry + coordinate_alpha * (dynamic - carry)
        raise ValueError(coupling)

    def _counterfactual_unroll(
        self,
        model: Any,
        source_arm: str,
        width: int,
        seed: int,
        batch: Mapping[str, Any],
        prior: float,
        coupling: str,
    ) -> Dict[str, Any]:
        current_state = batch["state_t"][:, 0]
        current_voltage = batch["voltage_t"][:, 0]
        initial_voltage = current_voltage
        hidden = atomic.torch.zeros(
            current_voltage.shape[0], self.layout.segment_count, width,
            dtype=current_voltage.dtype, device=current_voltage.device,
        )
        center = atomic.torch.as_tensor(self.statistics["state_center"], device=current_state.device)
        scale = atomic.torch.as_tensor(self.statistics["state_scale"], device=current_state.device)
        delta_scale = atomic.torch.as_tensor(self.statistics["delta_scale"], device=current_state.device)
        parent, children, child_mask = self._topology_tensors(
            source_arm, seed, current_voltage.dtype, current_voltage.device
        )
        outputs = {}
        pair = self.source_models[("full_feedback_scalar", seed)]
        for step in range(self.config.objective_unroll_horizon_ms):
            context = atomic.torch.cat((batch["drive"][:, step], batch["held_ions"]), dim=-1)
            normalized = (current_state - center) / scale
            raw = self._bridge_forward(pair[0], normalized, current_voltage, context)
            baseline = raw * self._static_gain(seed, raw, current_voltage)
            features = self._features(
                source_arm, normalized, current_voltage, initial_voltage, raw, baseline, context
            )
            alpha, hidden_next = model(
                features,
                atomic.torch.as_tensor(self.layout.segment_region_ids, dtype=atomic.torch.long, device=current_voltage.device),
                hidden, parent, children, child_mask,
                recurrent=source_arm == LOCAL_RECURRENT,
                topology="local" if source_arm == LOCAL_RECURRENT else "none",
            )
            shifted = self._shift_gate_prior(alpha, prior)
            next_voltage = current_voltage + shifted * baseline
            state_delta = self._coupled_state_delta(
                coupling, GENERIC_STATE, seed, normalized, current_voltage, baseline, shifted, context
            )
            next_state = current_state + state_delta * delta_scale
            if step + 1 in self.config.rollout_horizons_ms:
                outputs[f"{step + 1}_ms"] = {
                    "state": next_state,
                    "voltage": next_voltage,
                    "alpha": shifted,
                }
            hidden = hidden_next if source_arm == LOCAL_RECURRENT else atomic.torch.zeros_like(hidden_next)
            current_state, current_voltage = next_state, next_voltage
        return outputs

    def _teacher_paths(self, indices: np.ndarray, device: Any) -> Any:
        sample_indices = np.asarray(self.config.voltage_path_sample_indices, dtype=np.int64)
        paths = []
        for index in np.asarray(indices, dtype=np.int64).reshape(-1):
            trace = np.asarray(self.store.microtrace(int(index)), dtype=np.float32)
            if trace.shape != (self.config.expected_microtrace_sample_count, self.layout.segment_count):
                raise RuntimeError(f"06b-n unexpected microtrace shape {trace.shape}")
            paths.append(trace[sample_indices].T - trace[0][..., None])
        return atomic.torch.as_tensor(np.asarray(paths), device=device)

    def run_frozen_counterfactual_matrix(self) -> Dict[str, Any]:
        device = next(iter(self.frozen_state_models.values())).proposal.weight.device
        rows = np.arange(len(self.window_data["development"]["indices"]), dtype=np.int64)
        batch = self._batch_tensors("development", rows, device)
        per_seed: Dict[str, Any] = {}
        total = len(self.config.pilot_seeds) * 2 * len(self.config.frozen_gate_priors) * 2
        progress = atomic._CompactProgress("06b-n frozen counterfactuals", total, max(1, total // 20))
        completed = 0
        for seed in self.config.pilot_seeds:
            seed_rows = {}
            for source_arm, width in ((LOCAL_RECURRENT, 32), (PHYSIOLOGY_INSTANTANEOUS, 16)):
                model = self.frozen_controllers[(source_arm, width, seed)]
                for prior in self.config.frozen_gate_priors:
                    for coupling in self.config.factor_couplings:
                        with atomic.torch.no_grad():
                            outputs = self._counterfactual_unroll(
                                model, source_arm, width, seed, batch, prior, coupling
                            )
                        endpoint = outputs["8_ms"]
                        metrics = self._metric(endpoint["state"], endpoint["voltage"], batch, 7)
                        row = {
                            **metrics,
                            "alpha_mean": float(endpoint["alpha"].mean().cpu()),
                            "alpha_standard_deviation": float(endpoint["alpha"].std().cpu()),
                        }
                        key = f"{source_arm}|w{width}|p{prior:.2f}|{coupling}"
                        seed_rows[key] = row
                        completed += 1
                        progress.update(completed, f"seed={seed} {key}")
            pair = self.source_models[("full_feedback_scalar", seed)]
            current_state = batch["state_t"][:, 0]
            current_voltage = batch["voltage_t"][:, 0]
            context = atomic.torch.cat((batch["drive"][:, 0], batch["held_ions"]), dim=-1)
            normalized = (current_state - atomic.torch.as_tensor(self.statistics["state_center"], device=device)) / atomic.torch.as_tensor(self.statistics["state_scale"], device=device)
            raw = self._bridge_forward(pair[0], normalized, current_voltage, context)
            endpoint = raw * self._static_gain(seed, raw, current_voltage)
            fractions = np.asarray(self.config.voltage_path_sample_indices, dtype=np.float32) / float(self.config.expected_microtrace_sample_count - 1)
            path_variants = {
                "causal_linear_8_support": endpoint[:, :, None] * atomic.torch.as_tensor(fractions, dtype=current_voltage.dtype, device=device)[None, None, :],
                "causal_coarse_4_support": endpoint[:, :, None] * atomic.torch.as_tensor(np.ceil(fractions * 4.0) / 4.0, dtype=current_voltage.dtype, device=device)[None, None, :],
                "causal_endpoint_2_support": endpoint[:, :, None] * atomic.torch.as_tensor(np.where(fractions < 1.0, 0.0, 1.0), dtype=current_voltage.dtype, device=device)[None, None, :],
                "teacher_microtrace_upper_bound": self._teacher_paths(self.window_data["development"]["indices"][:, 0], device),
            }
            target_state = batch["state_t1"][:, 0]
            state_scale = atomic.torch.as_tensor(self.statistics["state_scale"], device=device)
            path_diagnostics = {}
            for path_name, voltage_path in path_variants.items():
                with atomic.torch.no_grad():
                    state_delta = self._state_forward_path(
                        pair[1], normalized, current_voltage, voltage_path, context
                    )
                predicted_state = current_state + state_delta * atomic.torch.as_tensor(self.statistics["delta_scale"], device=device)
                path_diagnostics[path_name] = {
                    "normalized_state_rmse": float(atomic.torch.mean(((predicted_state - target_state) / state_scale) ** 2).sqrt().cpu()),
                    "selectable": path_name != "teacher_microtrace_upper_bound",
                    "teacher_future_used": path_name == "teacher_microtrace_upper_bound",
                }
            seed_rows["state_path_diagnostics"] = path_diagnostics
            per_seed[str(seed)] = seed_rows
        report = {
            "schema_version": "06b-n-frozen-counterfactual-v1",
            "valid": all(
                np.isfinite(value)
                for rows_ in per_seed.values()
                for row in rows_.values()
                for value in row.values()
                if isinstance(value, (int, float))
            ),
            "role": "historically_reused_train_development",
            "gate_prior_transform_is_posthoc_diagnostic": True,
            "teacher_microtrace_upper_bound_selectable": False,
            "per_seed": per_seed,
        }
        atomic._write_json(self.output_dir / "frozen_counterfactual_matrix.json", report)
        return report

    def _relaxation_one_step_loss(
        self,
        model: Any,
        seed: int,
        batch: Mapping[str, Any],
        semantic_permutation: Optional[np.ndarray] = None,
    ) -> Any:
        current_state = batch["state_t"][:, 0]
        current_voltage = batch["voltage_t"][:, 0]
        target_state = batch["state_t1"][:, 0]
        center = atomic.torch.as_tensor(self.statistics["state_center"], device=current_state.device)
        scale = atomic.torch.as_tensor(self.statistics["state_scale"], device=current_state.device)
        delta_scale = atomic.torch.as_tensor(self.statistics["delta_scale"], device=current_state.device)
        normalized = (current_state - center) / scale
        context = atomic.torch.cat((batch["drive"][:, 0], batch["held_ions"]), dim=-1)
        pair = self.source_models[("full_feedback_scalar", seed)]
        with atomic.torch.no_grad():
            raw = self._bridge_forward(pair[0], normalized, current_voltage, context)
            baseline = raw * self._static_gain(seed, raw, current_voltage)
        prediction = self._relaxation_state_forward(
            model,
            normalized,
            current_voltage,
            baseline.detach(),
            context,
            semantic_permutation=semantic_permutation,
        )
        target_delta = (target_state - current_state) / delta_scale
        return atomic.torch_functional.smooth_l1_loss(prediction, target_delta)

    def _evaluate_relaxation(
        self, model: Any, seed: int, role: str, device: Any, *, shuffled: bool = False
    ) -> Dict[str, Any]:
        rows = np.arange(len(self.window_data[role]["indices"]), dtype=np.int64)
        batch = self._batch_tensors(role, rows, device)
        was_training = bool(model.training)
        model.eval()
        permutation = None
        if shuffled:
            permutation = np.random.default_rng(seed + 650900).permutation(
                len(self.mechanism_records)
            )
        with atomic.torch.no_grad():
            loss = self._relaxation_one_step_loss(
                model, seed, batch, semantic_permutation=permutation
            )
        model.train(was_training)
        return {"normalized_smooth_l1": float(loss.cpu())}

    def train_relaxation_updaters(self) -> Dict[str, Any]:
        device = next(iter(self.frozen_state_models.values())).proposal.weight.device
        reports = {}
        for seed in self.config.pilot_seeds:
            atomic.torch.manual_seed(seed + 650000)
            model = self._new_relaxation_model(device)
            optimizer = atomic.torch.optim.AdamW(
                model.parameters(), lr=self.config.relaxation_learning_rate,
                weight_decay=self.config.relaxation_weight_decay,
            )
            rng = np.random.default_rng(seed + 650100)
            checkpoints = []
            progress = atomic._CompactProgress(
                f"06b-n relaxation seed={seed}", self.config.relaxation_training_steps,
                max(1, self.config.relaxation_training_steps // 20),
            )
            for step in range(self.config.relaxation_training_steps + 1):
                if step in self.config.relaxation_checkpoints:
                    checkpoints.append({
                        "step": step,
                        "calibration": self._evaluate_relaxation(model, seed, "calibration", device),
                    })
                if step == self.config.relaxation_training_steps:
                    break
                rows = rng.choice(
                    len(self.window_data["fit"]["indices"]),
                    size=self.config.factor_batch_window_count,
                    replace=False,
                )
                batch = self._batch_tensors("fit", rows, device)
                optimizer.zero_grad(set_to_none=True)
                loss = self._relaxation_one_step_loss(model, seed, batch)
                if not bool(atomic.torch.isfinite(loss)):
                    raise RuntimeError(f"non-finite relaxation loss seed={seed} step={step}")
                loss.backward()
                atomic.torch.nn.utils.clip_grad_norm_(
                    model.parameters(), self.config.relaxation_gradient_clip_norm
                )
                optimizer.step()
                progress.update(step + 1, f"loss={float(loss.detach().cpu()):.4g}")
            model.eval()
            for parameter in model.parameters():
                parameter.requires_grad_(False)
            self.relaxation_models[seed] = model
            path = self.output_dir / f"relaxation_state_seed{seed}.pt"
            atomic.torch.save({
                "seed": seed,
                "state_dict": {name: value.detach().cpu() for name, value in model.state_dict().items()},
                "configuration": asdict(self.config),
            }, path)
            reports[str(seed)] = {
                "checkpoints": checkpoints,
                "development": self._evaluate_relaxation(model, seed, "development", device),
                "development_shuffled_semantics": self._evaluate_relaxation(
                    model, seed, "development", device, shuffled=True
                ),
                "checkpoint": path.name,
                "checkpoint_sha256": atomic._sha256_file(path),
            }
        report = {
            "schema_version": "06b-n-relaxation-training-v1",
            "valid": len(reports) == len(self.config.pilot_seeds),
            "teacher_microtrace_used_for_training": False,
            "development_used_for_selection": False,
            "reports": reports,
        }
        atomic._write_json(self.output_dir / "relaxation_training_report.json", report)
        return report

    def _new_factor_controller(self, prior: float, seed: int, device: Any) -> Any:
        atomic.torch.manual_seed(seed + 651000 + self.config.factor_hidden_width)
        model = self._new_mixture_model(self.config.factor_hidden_width, device)
        with atomic.torch.no_grad():
            model.readout.bias.fill_(math.log(prior / (1.0 - prior)))
        return model

    def _factor_unroll(
        self,
        model: Any,
        spec: Tuple[float, str, str, str],
        seed: int,
        batch: Mapping[str, Any],
        *,
        collect: bool,
    ) -> Tuple[Any, Dict[str, Any]]:
        _, objective, coupling, state_kind = spec
        current_state = batch["state_t"][:, 0]
        current_voltage = batch["voltage_t"][:, 0]
        initial_voltage = current_voltage
        width = self.config.factor_hidden_width
        hidden = atomic.torch.zeros(
            current_voltage.shape[0], self.layout.segment_count, width,
            dtype=current_voltage.dtype, device=current_voltage.device,
        )
        center = atomic.torch.as_tensor(self.statistics["state_center"], device=current_state.device)
        scale = atomic.torch.as_tensor(self.statistics["state_scale"], device=current_state.device)
        delta_scale = atomic.torch.as_tensor(self.statistics["delta_scale"], device=current_state.device)
        parent, children, child_mask = self._topology_tensors(
            LOCAL_RECURRENT, seed, current_voltage.dtype, current_voltage.device
        )
        pair = self.source_models[("full_feedback_scalar", seed)]
        losses = []
        outputs = {}
        for step in range(self.config.objective_unroll_horizon_ms):
            context = atomic.torch.cat((batch["drive"][:, step], batch["held_ions"]), dim=-1)
            normalized = (current_state - center) / scale
            raw = self._bridge_forward(pair[0], normalized, current_voltage, context)
            baseline = raw * self._static_gain(seed, raw, current_voltage)
            features = self._features(
                LOCAL_RECURRENT, normalized, current_voltage, initial_voltage, raw, baseline, context
            )
            alpha, next_hidden = model(
                features,
                atomic.torch.as_tensor(self.layout.segment_region_ids, dtype=atomic.torch.long, device=current_voltage.device),
                hidden, parent, children, child_mask, recurrent=True, topology="local",
            )
            voltage_delta = alpha * baseline
            next_voltage = current_voltage + voltage_delta
            state_delta = self._coupled_state_delta(
                coupling, state_kind, seed, normalized, current_voltage, baseline, alpha, context
            )
            next_state = current_state + state_delta * delta_scale
            target_voltage = batch["voltage_t1"][:, step]
            target_state = batch["state_t1"][:, step]
            target_delta = target_voltage - batch["voltage_t"][:, step]
            voltage_error = (next_voltage - target_voltage) / self.config.bridge_voltage_scale_mv
            voltage_point = atomic.torch_functional.smooth_l1_loss(
                voltage_error, atomic.torch.zeros_like(voltage_error), reduction="none"
            )
            voltage_loss = atomic.torch.mean(self._mixture_loss_weight(target_delta) * voltage_point)
            state_loss = atomic.torch_functional.smooth_l1_loss(
                (next_state - target_state) / scale,
                atomic.torch.zeros_like(next_state),
            )
            high = atomic.torch.relu((next_voltage - self.config.physical_voltage_maximum_mv) / self.config.bridge_voltage_scale_mv)
            low = atomic.torch.relu((self.config.physical_voltage_minimum_mv - next_voltage) / self.config.bridge_voltage_scale_mv)
            loss = voltage_loss + self.config.mixture_STATE_loss_weight * state_loss
            loss = loss + self.config.mixture_physical_penalty_weight * atomic.torch.mean(high * high + low * low)
            loss = loss + self.config.mixture_drift_penalty_weight * atomic.torch.mean(
                atomic.torch.mean(voltage_error, dim=1) ** 2
            )
            if objective == PERSISTENCE_REGRET:
                model_sq = voltage_error * voltage_error
                persistence_sq = ((current_voltage - target_voltage) / self.config.bridge_voltage_scale_mv) ** 2
                regret = atomic.torch_functional.softplus(
                    (model_sq - persistence_sq - self.config.persistence_regret_margin)
                    / self.config.persistence_regret_temperature
                ) * self.config.persistence_regret_temperature
                quiet = (target_delta.abs() < self.config.quiescent_threshold_mv).to(regret.dtype)
                regret_weight = 1.0 + (self.config.quiescent_regret_multiplier - 1.0) * quiet
                loss = loss + self.config.persistence_regret_weight * atomic.torch.mean(regret_weight * regret)
            losses.append(loss)
            if collect and step + 1 in self.config.rollout_horizons_ms:
                outputs[f"{step + 1}_ms"] = {"state": next_state, "voltage": next_voltage, "alpha": alpha}
            hidden = next_hidden
            current_state, current_voltage = next_state, next_voltage
        return atomic.torch.stack(losses).mean(), outputs

    def _evaluate_factor_model(
        self, model: Any, spec: Tuple[float, str, str, str], seed: int, role: str, device: Any
    ) -> Dict[str, Any]:
        rows = np.arange(len(self.window_data[role]["indices"]), dtype=np.int64)
        batch = self._batch_tensors(role, rows, device)
        was_training = bool(model.training)
        model.eval()
        with atomic.torch.no_grad():
            _, outputs = self._factor_unroll(model, spec, seed, batch, collect=True)
        result = {"horizons": {}}
        for horizon, output in outputs.items():
            step = int(horizon[:-3]) - 1
            metric = self._metric(output["state"], output["voltage"], batch, step)
            metric["alpha_mean"] = float(output["alpha"].mean().cpu())
            metric["alpha_standard_deviation"] = float(output["alpha"].std().cpu())
            result["horizons"][horizon] = metric
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

    def train_synchronized_factorial_matrix(self) -> Dict[str, Any]:
        if len(self.relaxation_models) != len(self.config.pilot_seeds):
            raise RuntimeError("06b-n relaxation updaters must be trained first")
        device = next(iter(self.frozen_state_models.values())).proposal.weight.device
        specs = self._factor_specs()
        reports = {}
        for seed in self.config.pilot_seeds:
            models = {spec: self._new_factor_controller(spec[0], seed, device) for spec in specs}
            optimizers = {
                spec: atomic.torch.optim.AdamW(
                    model.parameters(), lr=self.config.factor_learning_rate,
                    weight_decay=self.config.factor_weight_decay,
                )
                for spec, model in models.items()
            }
            seed_rows = {self._factor_key(spec): [] for spec in specs}
            rng = np.random.default_rng(seed + 652000)
            digest = hashlib.sha256()
            progress = atomic._CompactProgress(
                f"06b-n 2x2x2x2 seed={seed}", self.config.factor_training_steps,
                max(1, self.config.factor_training_steps // 25),
            )
            for step in range(self.config.factor_training_steps + 1):
                if step in self.config.factor_checkpoints:
                    for spec, model in models.items():
                        seed_rows[self._factor_key(spec)].append({
                            "step": step,
                            "calibration": self._evaluate_factor_model(model, spec, seed, "calibration", device),
                        })
                if step == self.config.factor_training_steps:
                    break
                rows = rng.choice(
                    len(self.window_data["fit"]["indices"]),
                    size=self.config.factor_batch_window_count,
                    replace=False,
                )
                digest.update(np.asarray(rows, dtype=np.int64).tobytes())
                batch = self._batch_tensors("fit", rows, device)
                losses = []
                for spec, model in models.items():
                    optimizer = optimizers[spec]
                    optimizer.zero_grad(set_to_none=True)
                    loss, _ = self._factor_unroll(model, spec, seed, batch, collect=False)
                    if not bool(atomic.torch.isfinite(loss)):
                        raise RuntimeError(
                            f"non-finite 06b-n loss seed={seed} step={step} arm={self._factor_key(spec)}"
                        )
                    loss.backward()
                    atomic.torch.nn.utils.clip_grad_norm_(
                        model.parameters(), self.config.factor_gradient_clip_norm
                    )
                    optimizer.step()
                    losses.append(float(loss.detach().cpu()))
                progress.update(
                    step + 1,
                    f"loss[min/median/max]={min(losses):.3g}/{float(np.median(losses)):.3g}/{max(losses):.3g}",
                )
            for spec, model in models.items():
                model.eval()
                self.factor_models[(self._factor_key(spec), seed)] = model
                path = self.output_dir / f"factor_{self._factor_key(spec).replace('|','__')}_seed{seed}.pt"
                atomic.torch.save({
                    "spec": spec,
                    "seed": seed,
                    "state_dict": {name: value.detach().cpu() for name, value in model.state_dict().items()},
                }, path)
                seed_rows[self._factor_key(spec)][-1]["checkpoint"] = path.name
                seed_rows[self._factor_key(spec)][-1]["checkpoint_sha256"] = atomic._sha256_file(path)
            reports[str(seed)] = {
                "batch_stream_sha256": digest.hexdigest(),
                "runs": seed_rows,
            }
        expected = len(self.config.factor_checkpoints)
        report = {
            "schema_version": "06b-n-factorial-training-v1",
            "valid": all(
                len(rows) == expected
                for seed in reports.values()
                for rows in seed["runs"].values()
            ),
            "factor_arm_count": len(specs),
            "same_minibatch_stream_within_seed": True,
            "same_initialization_except_registered_gate_bias": True,
            "teacher_endpoint_used_as_input": False,
            "development_used_during_training": False,
            "reports": reports,
        }
        self.factor_training_valid = bool(report["valid"])
        atomic._write_json(self.output_dir / "factorial_training_report.json", report)
        return report

    def evaluate_factorial_matrix(self) -> Dict[str, Any]:
        if not self.factor_training_valid:
            raise RuntimeError("06b-n factorial training is not complete")
        device = next(iter(self.frozen_state_models.values())).proposal.weight.device
        per_seed = {}
        specs = self._factor_specs()
        progress = atomic._CompactProgress(
            "06b-n development factorial", len(specs) * len(self.config.pilot_seeds),
            max(1, len(specs) * len(self.config.pilot_seeds) // 20),
        )
        completed = 0
        for seed in self.config.pilot_seeds:
            rows = {}
            for spec in specs:
                key = self._factor_key(spec)
                rows[key] = self._evaluate_factor_model(
                    self.factor_models[(key, seed)], spec, seed, "development", device
                )
                completed += 1
                progress.update(completed, f"seed={seed} {key}")
            rows[STATIC_REFERENCE] = self._recursive_gate_evaluation(
                seed, STATIC_REFERENCE, "development", device
            )
            per_seed[str(seed)] = rows
        report = {
            "schema_version": "06b-n-factorial-development-v1",
            "valid": all(
                row["horizons"]["8_ms"]["nonfinite_voltage_count"] == 0
                and row["horizons"]["8_ms"]["nonfinite_state_count"] == 0
                for seed in per_seed.values()
                for key, row in seed.items()
                if key != STATIC_REFERENCE
            ),
            "role": "historically_reused_train_development",
            "validation_state_accessed": False,
            "test_state_accessed": False,
            "per_seed": per_seed,
        }
        atomic._write_json(self.output_dir / "factorial_development.json", report)
        return report

    def _factor_summary(self, key: str, evaluation: Mapping[str, Any]) -> Dict[str, Any]:
        rows = [seed[key] for seed in evaluation["per_seed"].values()]
        references = [seed[STATIC_REFERENCE] for seed in evaluation["per_seed"].values()]
        endpoint = [row["horizons"]["8_ms"] for row in rows]
        reference_endpoint = [row["horizons"]["8_ms"] for row in references]
        return {
            "median_gain_over_static_fraction": self._median([
                1.0 - row["voltage_rmse_mv"] / max(ref["voltage_rmse_mv"], 1e-12)
                for row, ref in zip(endpoint, reference_endpoint)
            ]),
            "median_voltage_gain_vs_persistence_fraction": self._median([
                row["voltage_improvement_vs_persistence_fraction"] for row in endpoint
            ]),
            "median_STATE_gain_vs_persistence_fraction": self._median([
                row["state_improvement_vs_persistence_fraction"] for row in endpoint
            ]),
            "minimum_seed_STATE_gain_vs_persistence_fraction": min(
                row["state_improvement_vs_persistence_fraction"] for row in endpoint
            ),
            "median_alpha_mean": self._median([row["alpha_mean"] for row in endpoint]),
            "median_alpha_standard_deviation": self._median([
                row["alpha_standard_deviation"] for row in endpoint
            ]),
            "activity_gain_vs_persistence": {
                name: self._median([
                    row["activity_at_8ms"][name]["voltage_gain_vs_persistence_fraction"]
                    for row in rows
                ])
                for name in rows[0]["activity_at_8ms"]
            },
            "region_gain_vs_persistence": {
                name: self._median([
                    row["region_at_8ms"][name]["voltage_gain_vs_persistence_fraction"]
                    for row in rows
                ])
                for name in rows[0]["region_at_8ms"]
            },
            "physical_voltage_violation_count": int(sum(
                row["physical_voltage_violation_count"] for row in endpoint
            )),
        }

    def _main_effect(
        self,
        summaries: Mapping[str, Mapping[str, Any]],
        axis: int,
        positive: Any,
        negative: Any,
    ) -> float:
        contrasts = []
        specs = self._factor_specs()
        for spec in specs:
            if spec[axis] != positive:
                continue
            reference = list(spec)
            reference[axis] = negative
            left = summaries[self._factor_key(spec)]["median_gain_over_static_fraction"]
            right = summaries[self._factor_key(tuple(reference))]["median_gain_over_static_fraction"]
            contrasts.append(left - right)
        return self._median(contrasts)

    def finalize_structure_preserving_forensic(
        self,
        frozen_report: Mapping[str, Any],
        relaxation_report: Mapping[str, Any],
        evaluation: Mapping[str, Any],
    ) -> Dict[str, Any]:
        summaries = {
            self._factor_key(spec): self._factor_summary(self._factor_key(spec), evaluation)
            for spec in self._factor_specs()
        }
        low_prior, neutral_prior = min(self.config.factor_gate_priors), max(self.config.factor_gate_priors)
        effects = {
            "carry_first_over_neutral": self._main_effect(summaries, 0, low_prior, neutral_prior),
            "persistence_regret_over_standard": self._main_effect(summaries, 1, PERSISTENCE_REGRET, STANDARD_ROLLOUT),
            "endpoint_mix_over_pre_mix": self._main_effect(summaries, 2, ENDPOINT_MIXED_STATE, PRE_MIXED_STATE),
            "relaxation_over_generic_state": self._main_effect(summaries, 3, RELAXATION_STATE, GENERIC_STATE),
        }
        best_key = max(summaries, key=lambda key: summaries[key]["median_gain_over_static_fraction"])
        best = summaries[best_key]
        critical_region_gains = [
            value
            for name, value in best["region_gain_vs_persistence"].items()
            if any(token in name.lower() for token in ("soma", "axon", "ais"))
        ]
        selectable = bool(
            best["median_gain_over_static_fraction"] >= self.config.minimum_recursive_gain_over_static_fraction
            and best["minimum_seed_STATE_gain_vs_persistence_fraction"] > 0
            and best["activity_gain_vs_persistence"]["quiescent_lt_1mV"] >= 0
            and best["activity_gain_vs_persistence"]["moderate_1_to_5mV"] >= 0
            and best["activity_gain_vs_persistence"]["active_ge_5mV"] >= self.config.minimum_active_gain_fraction
            and best["region_gain_vs_persistence"]["soma"] >= 0
            and critical_region_gains
            and all(value >= 0 for value in critical_region_gains)
            and best["physical_voltage_violation_count"] == 0
        )
        material = [name for name, value in effects.items() if value >= self.config.minimum_material_factor_gain]
        if selectable:
            diagnosis = "STRUCTURE_PRESERVING_MIXTURE_REVISION_IDENTIFIED"
            next_step = "fresh_train_support_structure_preserving_confirmation"
        elif material:
            diagnosis = "MATERIAL_COMPONENT_EFFECTS_WITHOUT_ABSOLUTE_ROLLOUT_GATE"
            next_step = "focused_revision_from_registered_main_effects"
        else:
            diagnosis = "OBJECTIVE_COUPLING_AND_RELAXATION_DO_NOT_CLOSE_ROLLOUT_GAP"
            next_step = "revise_voltage_expert_family_or_state_contract"
        report = {
            "schema_version": "06b-n-final-report-v1",
            "valid": bool(
                frozen_report.get("valid")
                and relaxation_report.get("valid")
                and evaluation.get("valid")
            ),
            "component_playground_grade": True,
            "diagnosis": diagnosis,
            "selected_candidate": best_key if selectable else None,
            "best_observed_arm": best_key,
            "best_observed_arm_metrics": best,
            "factor_main_effects": effects,
            "material_positive_effects": material,
            "summaries": summaries,
            "frozen_counterfactual_valid": bool(frozen_report.get("valid")),
            "teacher_microtrace_upper_bound_selectable": False,
            "exact_cnexp_replay_claimed": False,
            "new_independent_confirmation_claimed": False,
            "validation_state_accessed": False,
            "test_state_accessed": False,
            "fresh_train_support_confirmation_authorized": selectable,
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
    "BoundedRelaxationStateUpdater",
    "ENDPOINT_MIXED_STATE",
    "EXPECTED_06BM_ARCHIVE_SHA256",
    "EXPECTED_06BM_FINAL_SHA256",
    "EXPECTED_06BM_INDEX_SHA256",
    "GENERIC_STATE",
    "PERSISTENCE_REGRET",
    "PRE_MIXED_STATE",
    "RELAXATION_STATE",
    "STANDARD_ROLLOUT",
    "StructurePreservingCouplingConfig",
    "StructurePreservingCouplingForensic",
    "audit_cnexp_teacher_contract",
    "verified_06bm_artifact_root",
]
