"""Paired closed-loop recurrence test authorized by the positive 05o audit."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import math
from pathlib import Path
import random
import time
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np

from src.hayflow_data.composite_flowmap import CompositeFlowmapBundle
from src.hayflow_data.flowmap_dataset import StateNormalizer

from .graph_state_contract_reassessment import sketch_normalized_state
from .hines_experiment import _write_json
from .hines_isolation_experiment import sha256_file
from .hines_netcon_semantic_repair import HinesNetConSemanticRepairConfig
from .hines_regenerative_confirmation import _verified_artifact_root
from .hines_synaptic_domain_repair import (
    BoundedSynapticStateEncoder,
    HinesSynapticDomainRepairConfig,
)
from .rollout_aware_architecture_canary import (
    CAUSAL_DRIVE_FEATURES,
    model_parameter_count,
    nn,
    torch,
    torch_functional,
)
from .topology_controlled_recurrence_expansion import (
    TopologyControlledRecurrenceConfig,
    TopologyControlledRecurrenceExpansion,
    expanded_train_episode_roles,
)


EXPECTED_05O_ARCHIVE_SHA256 = (
    "a51fbdab2b7d14c33c112c17c7cc2f50a26cb86e67f6aa2c655cc52d22e29477"
)
EXPECTED_05O_INDEX_SHA256 = (
    "e8f83a57ba51ef6db5a6fdd4a10c393a934fda493df071d898e0fe299d622ea3"
)
EXPECTED_05O_FINAL_SHA256 = (
    "80842fe8e56f25a6c8735da06f5cec860cb22d924f4798ca9a973ced058fa984"
)


def verified_nonlinear_observability_artifact_root(
    source: Path, cache_dir: Path
) -> Tuple[Path, Dict[str, Any], Dict[str, Any]]:
    return _verified_artifact_root(
        Path(source),
        Path(cache_dir),
        marker_name="nonlinear_observability_reassessment_config.json",
        archive_sha256=EXPECTED_05O_ARCHIVE_SHA256,
        index_sha256=EXPECTED_05O_INDEX_SHA256,
        final_sha256=EXPECTED_05O_FINAL_SHA256,
    )


@dataclass(frozen=True)
class AxialRichStateRecurrentCanaryConfig(TopologyControlledRecurrenceConfig):
    state_sketch_dim: int = 64
    state_clip: float = 8.0
    epochs: int = 30
    progress_interval: int = 10
    material_gain_fraction: float = 0.05
    regenerative_noninferiority_margin_fraction: float = 0.01
    minimum_paired_win_count: int = 2

    def validate(self) -> None:
        super().validate()
        if self.state_sketch_dim != 64:
            raise ValueError("05p must preserve the verified 64-wide 05o state sketch")
        if self.state_clip != 8.0:
            raise ValueError("05p must preserve the verified 05o state clip")
        if not 0 < self.material_gain_fraction < 1:
            raise ValueError("05p materiality gate is invalid")
        if not 0 <= self.regenerative_noninferiority_margin_fraction < 1:
            raise ValueError("05p regenerative non-inferiority gate is invalid")
        if self.minimum_paired_win_count not in {2, 3}:
            raise ValueError("05p paired seed gate is invalid")

    @classmethod
    def from_mapping(
        cls, values: Mapping[str, Any]
    ) -> "AxialRichStateRecurrentCanaryConfig":
        payload = dict(values)
        for name in ("horizons_ms", "seeds"):
            if name in payload:
                payload[name] = tuple(map(int, payload[name]))
        result = cls(**payload)
        result.validate()
        return result


if nn is not None:

    class AxialRichStateGraphGRU(nn.Module):
        """One fixed-capacity GraphGRU with causally masked contract blocks."""

        def __init__(
            self,
            segment_static: np.ndarray,
            parent_ids: np.ndarray,
            axial_conductance: np.ndarray,
            *,
            state_width: int,
            hidden_width: int,
            voltage_delta_limit_mv: float,
            use_axial: bool,
            use_rich_state: bool,
        ) -> None:
            super().__init__()
            static = np.asarray(segment_static, dtype=np.float32)
            parent = np.asarray(parent_ids, dtype=np.int64)
            conductance = np.maximum(
                np.asarray(axial_conductance, dtype=np.float32), 0.0
            )
            positive = conductance[conductance > 0]
            reference = float(np.median(positive)) if len(positive) else 1.0
            scaled = np.clip(conductance / max(reference, 1e-12), 0.0, 20.0)
            child_ids = np.flatnonzero(parent != np.arange(len(parent))).astype(
                np.int64
            )
            child_parent = parent[child_ids]
            child_count = np.bincount(child_parent, minlength=len(parent)).astype(
                np.float32
            )
            child_weight = np.bincount(
                child_parent,
                weights=scaled[child_ids],
                minlength=len(parent),
            ).astype(np.float32)
            self.segment_count = int(len(parent))
            self.state_width = int(state_width)
            self.hidden_width = int(hidden_width)
            self.voltage_delta_limit_mv = float(voltage_delta_limit_mv)
            self.use_axial = bool(use_axial)
            self.use_rich_state = bool(use_rich_state)
            self.register_buffer("segment_static", torch.as_tensor(static))
            self.register_buffer("parent_ids", torch.as_tensor(parent))
            self.register_buffer("scaled_conductance", torch.as_tensor(scaled))
            self.register_buffer("child_ids", torch.as_tensor(child_ids))
            self.register_buffer("child_parent_ids", torch.as_tensor(child_parent))
            self.register_buffer("child_count", torch.as_tensor(child_count))
            self.register_buffer("child_weight", torch.as_tensor(child_weight))
            input_width = (
                1
                + len(CAUSAL_DRIVE_FEATURES)
                + static.shape[1]
                + 4
                + self.state_width
            )
            self.input_encoder = nn.Sequential(
                nn.Linear(input_width, hidden_width),
                nn.SiLU(),
                nn.Linear(hidden_width, hidden_width),
            )
            self.initial_encoder = nn.Linear(input_width, hidden_width)
            self.mixer = nn.Linear(3 * hidden_width, hidden_width)
            self.gru = nn.GRUCell(hidden_width, hidden_width)
            self.voltage_head = nn.Sequential(
                nn.Linear(hidden_width, hidden_width),
                nn.SiLU(),
                nn.Linear(hidden_width, 1),
            )

        def _axial(self, voltage: Any) -> Any:
            parent_delta = voltage[:, self.parent_ids] - voltage
            child_delta = torch.zeros_like(voltage)
            child_axial = torch.zeros_like(voltage)
            if self.child_ids.numel():
                differences = (
                    voltage[:, self.child_ids]
                    - voltage[:, self.child_parent_ids]
                )
                child_delta.index_add_(1, self.child_parent_ids, differences)
                child_axial.index_add_(
                    1,
                    self.child_parent_ids,
                    differences * self.scaled_conductance[self.child_ids],
                )
            child_delta = child_delta / self.child_count.clamp_min(1.0)[None, :]
            child_axial = child_axial / self.child_weight.clamp_min(1.0)[None, :]
            return torch.stack(
                (
                    parent_delta / 100.0,
                    child_delta / 100.0,
                    self.scaled_conductance[None, :] * parent_delta / 100.0,
                    child_axial / 100.0,
                ),
                dim=-1,
            )

        def _input(
            self, voltage: Any, drive: Any, state: Any, *, initial: bool
        ) -> Any:
            static = self.segment_static.unsqueeze(0).expand(
                voltage.shape[0], -1, -1
            )
            axial = self._axial(voltage)
            if not self.use_axial:
                axial = torch.zeros_like(axial)
            state_values = state if initial and self.use_rich_state else torch.zeros_like(state)
            return torch.cat(
                ((voltage / 100.0).unsqueeze(-1), drive, static, axial, state_values),
                dim=-1,
            )

        def _mix(self, hidden: Any) -> Any:
            parent = hidden[:, self.parent_ids]
            child_sum = torch.zeros_like(hidden)
            if self.child_ids.numel():
                child_sum.index_add_(
                    1, self.child_parent_ids, hidden[:, self.child_ids]
                )
            child_mean = child_sum / self.child_count.clamp_min(1.0).view(1, -1, 1)
            return torch_functional.silu(
                self.mixer(torch.cat((hidden, parent, child_mean), dim=-1))
            )

        def forward(
            self, voltage_t: Any, causal_drive: Any, initial_state: Any
        ) -> Dict[str, Any]:
            zero_drive = voltage_t.new_zeros(
                voltage_t.shape[0], self.segment_count, len(CAUSAL_DRIVE_FEATURES)
            )
            hidden = torch.tanh(
                self.initial_encoder(
                    self._input(
                        voltage_t, zero_drive, initial_state, initial=True
                    )
                )
            )
            voltage = voltage_t
            values = []
            for step in range(causal_drive.shape[1]):
                encoded = self.input_encoder(
                    self._input(
                        voltage, causal_drive[:, step], initial_state, initial=False
                    )
                )
                mixed = self._mix(hidden)
                hidden = self.gru(
                    (encoded + mixed).reshape(-1, self.hidden_width),
                    hidden.reshape(-1, self.hidden_width),
                ).reshape_as(hidden)
                delta = self.voltage_delta_limit_mv * torch.tanh(
                    self.voltage_head(hidden).squeeze(-1)
                )
                voltage = voltage + delta
                values.append(voltage)
            return {"voltage": torch.stack(values, dim=1), "hidden": hidden}

else:  # pragma: no cover

    class AxialRichStateGraphGRU:
        def __init__(self, *_: Any, **__: Any) -> None:
            raise RuntimeError("05p requires PyTorch")


class AxialRichStateRecurrentCanary(TopologyControlledRecurrenceExpansion):
    CONTRACTS = {
        "graphgru_voltage_only": (False, False),
        "graphgru_axial": (True, False),
        "graphgru_rich_state": (False, True),
        "graphgru_axial_rich_state": (True, True),
    }

    def __init__(
        self,
        bundle: CompositeFlowmapBundle,
        output_dir: Path,
        config: AxialRichStateRecurrentCanaryConfig,
        artifact_05o_source: Path,
        *,
        code_revision: str,
    ) -> None:
        super().__init__(
            bundle,
            output_dir,
            config,
            artifact_05o_source,
            code_revision=code_revision,
        )
        self.artifact_05o_source = Path(artifact_05o_source).resolve()
        self.netcon_config = HinesNetConSemanticRepairConfig()
        self.domain_config = HinesSynapticDomainRepairConfig()
        self.semantic_encoder = BoundedSynapticStateEncoder(
            self.store.layout, self.netcon_config, self.domain_config
        )
        self.normalizer: Any = None
        self.projection: Any = None

    @property
    def recurrent_config(self) -> AxialRichStateRecurrentCanaryConfig:
        return self.config  # type: ignore[return-value]

    @staticmethod
    def _state_sha256(model: Any) -> str:
        digest = hashlib.sha256()
        for name, value in sorted(model.state_dict().items()):
            if name in {"use_axial", "use_rich_state"}:
                continue
            digest.update(name.encode("utf-8"))
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
        return digest.hexdigest()

    def _model(self, family: str, device: Any) -> Any:
        use_axial, use_state = self.CONTRACTS[family]
        conductance = np.asarray(
            [
                max(0.0, float(row["axial_conductance_to_parent_us"]))
                for row in self.store.layout.segments
            ],
            dtype=np.float32,
        )
        return AxialRichStateGraphGRU(
            self.store.layout.segment_static,
            self._parent_ids(),
            conductance,
            state_width=self.recurrent_config.state_sketch_dim,
            hidden_width=self.config.hidden_width,
            voltage_delta_limit_mv=self.config.voltage_delta_limit_mv,
            use_axial=use_axial,
            use_rich_state=use_state,
        ).to(device)

    def _materialize_role(
        self, windows: Sequence[np.ndarray]
    ) -> Dict[str, np.ndarray]:
        values = super()._materialize_role(windows)
        starts = np.asarray([int(window[0]) for window in windows], dtype=np.int64)
        raw = self.store.read_state(starts, "t")
        times = np.asarray(
            self.store.metadata["start_time_ms"][starts], dtype=np.float64
        )
        semantic = self.semantic_encoder.encode(raw, times)
        normalized = self.normalizer.normalize_state(semantic)
        values["initial_state"] = sketch_normalized_state(
            normalized,
            self.store.layout,
            self.projection,
            clip=self.recurrent_config.state_clip,
        )
        return values

    def prepare(self) -> Dict[str, Any]:
        root, final, contract = verified_nonlinear_observability_artifact_root(
            self.artifact_05o_source,
            self.output_dir.parent / ".05p_artifact_cache" / "05o",
        )
        blockers = []
        if not final.get("valid") or not final.get("decision_grade"):
            blockers.append("05o is not decision-grade")
        if final.get("diagnosis") != "NONLINEAR_JOINT_GRAPH_AND_STATE_SIGNAL":
            blockers.append("05o diagnosis does not authorize the joint canary")
        if final.get("next_step") != "05p_axial_graphgru_rich_state_micro_canary":
            blockers.append("05o next step is not 05p")
        if not final.get("graph_information_signal") or not final.get(
            "initial_state_information_signal"
        ):
            blockers.append("05o joint information signal is incomplete")
        self.artifact_contract = contract
        archive = np.load(root / "initial_state_normalizer_and_projection.npz")
        self.normalizer = StateNormalizer(self.store.layout)
        self.normalizer.transform_codes = archive["transform_codes"].copy()
        self.normalizer.state_center = archive["state_center"].copy()
        self.normalizer.state_scale = archive["state_scale"].copy()
        self.projection = archive["projection"].copy()
        expected_projection = (
            self.store.layout.state_width,
            self.recurrent_config.state_sketch_dim,
        )
        if self.projection.shape != expected_projection:
            blockers.append("05o state projection shape changed")
        mapping = self.semantic_encoder.mapping_report
        if int(mapping.get("unmapped_coordinate_count", -1)) != 0:
            blockers.append("05p semantic state mapping is incomplete")
        self.roles, self.expansion_report = expanded_train_episode_roles(
            self.store.episode_rows, config=self.recurrent_config
        )
        overlap = {
            "seed": self._role_overlap(self.roles, "seed"),
            "snapshot": self._role_overlap(self.roles, "snapshot_id"),
            "snapshot_source": self._role_overlap(self.roles, "snapshot_source"),
            "trajectory": self._role_overlap(self.roles, "trajectory_id"),
        }
        if any(overlap.values()):
            blockers.append("05p role isolation failed")
        self.windows = self._build_windows()
        self.materialized = {
            role: self._materialize_role(windows)
            for role, windows in self.windows.items()
        }
        regenerative_counts = {}
        for role, values in self.materialized.items():
            endpoint = values["target_voltage"][:, -1]
            regenerative = (
                np.abs(endpoint - values["initial_voltage"]) >= 5.0
            ) | (endpoint >= -20.0)
            regenerative_counts[role] = int(regenerative.sum())
            if not regenerative_counts[role]:
                blockers.append(f"05p role {role} has no regenerative endpoints")
        if torch is None:
            blockers.append("PyTorch is unavailable")
            counts: Dict[str, int] = {}
        else:
            torch.manual_seed(int(self.config.seeds[0]))
            models = {
                family: self._model(family, torch.device("cpu"))
                for family in self.CONTRACTS
            }
            counts = {
                family: model_parameter_count(model)
                for family, model in models.items()
            }
            if len(set(counts.values())) != 1:
                blockers.append("05p recurrent contracts are not parameter-identical")
        report = {
            "schema_version": "05p-preflight-v1",
            "valid": not blockers,
            "blockers": blockers,
            "artifact_05o": contract,
            "dataset_fingerprint": self.bundle.fingerprint,
            "source_split_used": "train_only",
            "validation_or_test_loaded": False,
            "sealed_fresh_test_loaded": False,
            "teacher_future_state_used_as_model_input": False,
            "initial_teacher_state_used_only_at_window_boundary": True,
            "role_episode_counts": {
                name: len(rows) for name, rows in self.roles.items()
            },
            "role_window_counts": {
                name: len(rows) for name, rows in self.windows.items()
            },
            "role_overlap": overlap,
            "role_regenerative_endpoint_coordinate_counts": regenerative_counts,
            "contracts": self.CONTRACTS,
            "parameter_counts": counts,
            "parameter_identical": len(set(counts.values())) == 1 if counts else False,
            "state_sketch_shape": list(expected_projection),
            "causal_drive_features": list(CAUSAL_DRIVE_FEATURES),
            "support_expansion": self.expansion_report,
        }
        self.prepare_report = report
        _write_json(self.output_dir / "preflight_report.json", report)
        _write_json(
            self.output_dir / "axial_rich_state_recurrent_canary_config.json",
            {
                "schema_version": "05p-config-v1",
                "config": asdict(self.recurrent_config),
                "artifact_05o": contract,
                "dataset_fingerprint": self.bundle.fingerprint,
                "code_revision": self.code_revision,
            },
        )
        if blockers:
            raise RuntimeError(f"05p preflight failed: {blockers}")
        return report

    def _evaluate_arrays(
        self, model: Any, values: Mapping[str, Any]
    ) -> Dict[str, Dict[str, float]]:
        model.eval()
        result = {}
        with torch.no_grad():
            for horizon in self.config.horizons_ms:
                prediction = model(
                    values["initial_voltage"],
                    values["causal_drive"][:, :horizon],
                    values["initial_state"],
                )["voltage"]
                target = values["target_voltage"][:, :horizon]
                endpoint_error = prediction[:, -1] - target[:, -1]
                persistence_error = values["initial_voltage"] - target[:, -1]
                regenerative = (
                    torch.abs(target[:, -1] - values["initial_voltage"]) >= 5.0
                ) | (target[:, -1] >= -20.0)
                regenerative_error = endpoint_error[regenerative]
                violations = (
                    (prediction < self.config.physical_voltage_floor_mv)
                    | (prediction > self.config.physical_voltage_ceiling_mv)
                )
                result[str(horizon)] = {
                    "endpoint_rmse_mv": float(
                        torch.sqrt(torch.mean(endpoint_error.square())).cpu()
                    ),
                    "regenerative_endpoint_rmse_mv": float(
                        torch.sqrt(torch.mean(regenerative_error.square())).cpu()
                    ) if regenerative_error.numel() else math.nan,
                    "regenerative_coordinate_count": int(regenerative.sum().cpu()),
                    "path_rmse_mv": float(
                        torch.sqrt(torch.mean((prediction - target).square())).cpu()
                    ),
                    "endpoint_mean_drift_mv": float(torch.mean(endpoint_error).cpu()),
                    "persistence_endpoint_rmse_mv": float(
                        torch.sqrt(torch.mean(persistence_error.square())).cpu()
                    ),
                    "physical_voltage_violation_count": int(violations.sum().cpu()),
                    "finite": bool(torch.isfinite(prediction).all().cpu()),
                }
        return result

    def _train_one(
        self, family: str, seed: int, device: Any
    ) -> Tuple[Any, Dict[str, Any]]:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        model = self._model(family, device)
        initial_sha256 = self._state_sha256(model)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        fit = self._tensor_role("fit", device)
        calibration = self._tensor_role("calibration", device)
        rng = np.random.default_rng(seed + 7001)
        best_score = math.inf
        best_state = None
        best_epoch = -1
        history: List[Dict[str, Any]] = []
        started = time.monotonic()
        window_count = int(fit["initial_voltage"].shape[0])
        order_digest = hashlib.sha256()
        for epoch in range(self.config.epochs):
            if epoch < self.config.epochs // 3:
                horizon = 2
            elif epoch < 2 * self.config.epochs // 3:
                horizon = 4
            else:
                horizon = 8
            order = rng.permutation(window_count)
            order_digest.update(order.astype("<i8", copy=False).tobytes())
            losses = []
            gradients = []
            model.train()
            for start in range(0, window_count, self.config.batch_size):
                positions = torch.as_tensor(
                    order[start : start + self.config.batch_size],
                    dtype=torch.long,
                    device=device,
                )
                optimizer.zero_grad(set_to_none=True)
                prediction = model(
                    fit["initial_voltage"].index_select(0, positions),
                    fit["causal_drive"].index_select(0, positions)[:, :horizon],
                    fit["initial_state"].index_select(0, positions),
                )["voltage"]
                target = fit["target_voltage"].index_select(0, positions)[:, :horizon]
                loss = self._loss(prediction, target)
                loss.backward()
                gradient = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), self.config.gradient_clip_norm
                )
                optimizer.step()
                losses.append(float(loss.detach().cpu()))
                gradients.append(float(gradient.detach().cpu()))
            evaluate = (
                epoch == 0
                or (epoch + 1) % self.config.evaluation_interval == 0
                or epoch + 1 == self.config.epochs
            )
            row: Dict[str, Any] = {
                "epoch": epoch + 1,
                "curriculum_horizon_ms": horizon,
                "fit_loss": float(np.mean(losses)),
                "gradient_norm_pre_clip": float(np.mean(gradients)),
            }
            if evaluate:
                metrics = self._evaluate_arrays(model, calibration)
                score = metrics["8"]["endpoint_rmse_mv"]
                row["calibration_endpoint_rmse_8ms_mv"] = score
                row["calibration_regenerative_rmse_8ms_mv"] = metrics["8"][
                    "regenerative_endpoint_rmse_mv"
                ]
                if score < best_score:
                    best_score = score
                    best_epoch = epoch + 1
                    best_state = {
                        name: value.detach().cpu().clone()
                        for name, value in model.state_dict().items()
                    }
            history.append(row)
            if (
                epoch == 0
                or (epoch + 1) % self.config.progress_interval == 0
                or epoch + 1 == self.config.epochs
            ):
                elapsed = time.monotonic() - started
                eta = elapsed / (epoch + 1) * (self.config.epochs - epoch - 1)
                print(
                    f"[HayFlow 05p][{family} seed={seed}] "
                    f"{epoch + 1}/{self.config.epochs} ETA {eta / 60:.1f} min "
                    f"loss={row['fit_loss']:.4g}",
                    flush=True,
                )
        if best_state is None:
            raise RuntimeError("05p produced no calibration checkpoint")
        model.load_state_dict(best_state)
        checkpoint = self.checkpoint_dir / f"{family}-seed{seed}.pt"
        torch.save(
            {
                "state_dict": best_state,
                "family": family,
                "seed": seed,
                "best_epoch": best_epoch,
                "dataset_fingerprint": self.bundle.fingerprint,
                "code_revision": self.code_revision,
            },
            checkpoint,
        )
        return model, {
            "family": family,
            "seed": seed,
            "parameter_count": model_parameter_count(model),
            "initial_state_sha256": initial_sha256,
            "training_order_sha256": order_digest.hexdigest(),
            "best_epoch": best_epoch,
            "calibration_selection_rmse_8ms_mv": best_score,
            "checkpoint": checkpoint.relative_to(self.output_dir).as_posix(),
            "checkpoint_sha256": sha256_file(checkpoint),
            "history": history,
        }

    def run(self) -> Dict[str, Any]:
        if not self.prepare_report:
            self.prepare()
        if torch is None:
            raise RuntimeError("05p requires PyTorch")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        development = self._tensor_role("development", device)
        runs: Dict[str, Dict[str, Any]] = {
            family: {} for family in self.CONTRACTS
        }
        for family in self.CONTRACTS:
            for seed in self.config.seeds:
                model, row = self._train_one(family, int(seed), device)
                metrics = self._evaluate_arrays(model, development)
                eight = metrics["8"]
                improvement = 1.0 - eight["endpoint_rmse_mv"] / max(
                    eight["persistence_endpoint_rmse_mv"], 1e-12
                )
                row.update(
                    development=metrics,
                    development_improvement_vs_persistence_fraction=improvement,
                    passed=bool(
                        all(value["finite"] for value in metrics.values())
                        and sum(
                            value["physical_voltage_violation_count"]
                            for value in metrics.values()
                        ) == 0
                        and improvement
                        >= self.config.minimum_improvement_vs_persistence_fraction
                    ),
                )
                runs[family][str(seed)] = row
        paired_initialization = {
            str(seed): len({
                runs[family][str(seed)]["initial_state_sha256"]
                for family in self.CONTRACTS
            }) == 1
            for seed in self.config.seeds
        }
        paired_order = {
            str(seed): len({
                runs[family][str(seed)]["training_order_sha256"]
                for family in self.CONTRACTS
            }) == 1
            for seed in self.config.seeds
        }
        comparisons = {
            "axial_given_voltage": ("graphgru_axial", "graphgru_voltage_only"),
            "axial_given_rich_state": (
                "graphgru_axial_rich_state", "graphgru_rich_state"
            ),
            "state_given_graphgru": (
                "graphgru_rich_state", "graphgru_voltage_only"
            ),
            "state_given_axial": (
                "graphgru_axial_rich_state", "graphgru_axial"
            ),
        }
        gains = {}
        for name, (candidate, baseline) in comparisons.items():
            by_seed = {}
            for seed in map(str, self.config.seeds):
                candidate_metrics = runs[candidate][seed]["development"]["8"]
                baseline_metrics = runs[baseline][seed]["development"]["8"]
                by_seed[seed] = {
                    "rmse_gain_fraction": 1.0
                    - candidate_metrics["endpoint_rmse_mv"]
                    / max(baseline_metrics["endpoint_rmse_mv"], 1e-12),
                    "regenerative_rmse_gain_fraction": 1.0
                    - candidate_metrics["regenerative_endpoint_rmse_mv"]
                    / max(
                        baseline_metrics["regenerative_endpoint_rmse_mv"], 1e-12
                    ),
                }
            gains[name] = {
                "by_seed": by_seed,
                "median_rmse_gain_fraction": float(np.median([
                    value["rmse_gain_fraction"] for value in by_seed.values()
                ])),
                "median_regenerative_gain_fraction": float(np.median([
                    value["regenerative_rmse_gain_fraction"]
                    for value in by_seed.values()
                ])),
                "positive_win_count": sum(
                    value["rmse_gain_fraction"] > 0
                    for value in by_seed.values()
                ),
            }

        def signal(names: Tuple[str, str]) -> bool:
            return all(
                gains[name]["median_rmse_gain_fraction"]
                >= self.recurrent_config.material_gain_fraction
                and gains[name]["median_regenerative_gain_fraction"]
                >= -self.recurrent_config.regenerative_noninferiority_margin_fraction
                and gains[name]["positive_win_count"]
                >= self.recurrent_config.minimum_paired_win_count
                for name in names
            )

        axial_signal = signal(("axial_given_voltage", "axial_given_rich_state"))
        state_signal = signal(("state_given_graphgru", "state_given_axial"))
        passing = {
            family: sum(bool(row["passed"]) for row in family_runs.values())
            for family, family_runs in runs.items()
        }
        joint_robust = (
            passing["graphgru_axial_rich_state"]
            >= self.config.minimum_passing_seed_count
        )
        if joint_robust and axial_signal and state_signal:
            diagnosis = "ROLLOUT_JOINT_AXIAL_GRAPH_AND_RICH_STATE_SIGNAL"
            next_step = "05q_joint_state_graph_rollout_expansion"
        elif joint_robust and state_signal:
            diagnosis = "ROLLOUT_RICH_INITIAL_STATE_SIGNAL_ONLY"
            next_step = "05q_rich_state_recurrent_expansion"
        elif joint_robust and axial_signal:
            diagnosis = "ROLLOUT_AXIAL_GRAPH_SIGNAL_ONLY"
            next_step = "05q_axial_graph_recurrent_expansion"
        else:
            diagnosis = "NONLINEAR_ONE_STEP_SIGNAL_DID_NOT_TRANSFER_TO_ROLLOUT"
            next_step = "05q_temporal_observability_contract_reassessment"
        valid = bool(
            all(
                row["development"]["8"]["finite"]
                for family in runs.values()
                for row in family.values()
            )
            and all(paired_initialization.values())
            and all(paired_order.values())
        )
        report = {
            "schema_version": "05p-final-report-v1",
            "valid": valid,
            "decision_grade": True,
            "diagnosis": diagnosis,
            "code_revision": self.code_revision,
            "device": str(device),
            "artifact_05o": self.artifact_contract,
            "dataset_fingerprint": self.bundle.fingerprint,
            "preflight": self.prepare_report,
            "training_contract": {
                "closed_loop_voltage_rollout": True,
                "initial_teacher_state_boundary_only": True,
                "teacher_state_after_initial_boundary": False,
                "teacher_forcing_inside_window": False,
                "causal_input": "U_realized",
                "fit_selects_gradients": True,
                "calibration_selects_checkpoint": True,
                "development_read_once_after_freeze": True,
                "validation_or_test_loaded": False,
                "sealed_fresh_test_loaded": False,
            },
            "runs": runs,
            "paired_incremental_gains": gains,
            "paired_initialization_verified_by_seed": paired_initialization,
            "paired_training_order_verified_by_seed": paired_order,
            "passing_seed_count": passing,
            "axial_graph_rollout_signal": axial_signal,
            "rich_initial_state_rollout_signal": state_signal,
            "joint_candidate_robust": joint_robust,
            "recurrent_expansion_authorized": bool(
                joint_robust and (axial_signal or state_signal)
            ),
            "full_training_authorized": False,
            "fresh_test_generation_authorized": False,
            "mass_dataset_generation_authorized": False,
            "next_step": next_step,
        }
        _write_json(self.output_dir / "final_report.json", report)
        records = []
        for path in sorted(self.output_dir.rglob("*")):
            if path.is_file() and path.name != "artifact_index.json":
                records.append({
                    "path": path.relative_to(self.output_dir).as_posix(),
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                })
        _write_json(
            self.output_dir / "artifact_index.json",
            {
                "schema_version": "05p-artifact-index-v1",
                "artifact_count": len(records),
                "artifacts": records,
            },
        )
        return report


__all__ = [
    "EXPECTED_05O_ARCHIVE_SHA256",
    "EXPECTED_05O_FINAL_SHA256",
    "EXPECTED_05O_INDEX_SHA256",
    "AxialRichStateGraphGRU",
    "AxialRichStateRecurrentCanary",
    "AxialRichStateRecurrentCanaryConfig",
    "verified_nonlinear_observability_artifact_root",
]
