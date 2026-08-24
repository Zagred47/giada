"""06b-c: distinguish voltage-bridge optimization from missing tree support.

The six mechanism-STATE updaters and the three 06b-b voltage bridges are
immutable.  Only a bounded continuation of the local bridge or a small
zero-initialized residual head is trained.  Authentic and relabelled topology
heads have identical parameters, initialization, samples, and optimizer
budgets.
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
from .causal_voltage_state_coupling_forensic import (
    CausalVoltageStateCouplingConfig,
    CausalVoltageStateCouplingForensic,
)
from .topology_controlled_recurrence_expansion import (
    topology_relabelled_parent_ids,
)


EXPECTED_06BB_ARCHIVE_SHA256 = (
    "d652c3fdf088569b212c6fc710185ab4f870857e5e84cc2947459b7f456bb349"
)
EXPECTED_06BB_INDEX_SHA256 = (
    "824cc0fdfb977c69fed7bbf3dfcea691f6c1346c84fadedd988aa20c12c56986"
)
EXPECTED_06BB_FINAL_SHA256 = (
    "41a1fc65c2bc1b812e06758f171983e8201b2627905fdd6a7b0813c442bacc2e"
)

REPRESENTATION_ARMS = (
    "frozen_local_bridge",
    "continued_local_bridge",
    "authentic_tree_residual",
    "relabelled_tree_residual",
)


def verified_06bb_artifact_root(
    source: Path, cache_dir: Path
) -> Tuple[Path, Dict[str, Any]]:
    source = Path(source).expanduser().resolve()
    cache_dir = Path(cache_dir).resolve()
    if not source.exists():
        raise FileNotFoundError(source)
    if source.is_file():
        if source.suffix.lower() != ".zip":
            raise RuntimeError("06b-b source must be a ZIP or extracted directory")
        archive_hash = atomic._sha256_file(source)
        if archive_hash != EXPECTED_06BB_ARCHIVE_SHA256:
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
        if atomic._sha256_file(path) == EXPECTED_06BB_INDEX_SHA256
    ]
    if len(matches) != 1:
        raise RuntimeError(f"expected one exact 06b-b artifact; found {len(matches)}")
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
        raise RuntimeError(f"06b-b indexed member verification failed: {failures}")
    final_path = root / "final_report.json"
    if atomic._sha256_file(final_path) != EXPECTED_06BB_FINAL_SHA256:
        raise RuntimeError("06b-b final report SHA-256 mismatch")
    final = json.loads(final_path.read_text(encoding="utf-8"))
    if (
        final.get("diagnosis") != "CAUSAL_VOLTAGE_BRIDGE_NOT_PREDICTIVE"
        or final.get("component_decision_grade") is not True
        or final.get("next_step") != "inspect_voltage_bridge_representation"
    ):
        raise RuntimeError("06b-b source does not authorize representation forensics")
    return root, {
        "valid": True,
        "source_kind": "zip" if source.is_file() else "directory",
        "archive_sha256": archive_hash,
        "artifact_index_sha256": EXPECTED_06BB_INDEX_SHA256,
        "final_report_sha256": EXPECTED_06BB_FINAL_SHA256,
        "indexed_member_count": len(index.get("artifacts", [])),
        "diagnosis": final["diagnosis"],
    }


@dataclass(frozen=True)
class CausalVoltageBridgeRepresentationConfig(CausalVoltageStateCouplingConfig):
    continuation_training_steps: int = 500
    residual_training_steps: int = 500
    representation_hidden_width: int = 32
    topology_relabel_seed_offset: int = 606300
    representation_evaluation_interval: int = 50
    maximum_residual_parameter_count: int = 4000
    minimum_authentic_gain_over_relabelled_fraction: float = 0.02
    minimum_continuation_gain_over_frozen_fraction: float = 0.02

    def validate(self) -> None:
        super().validate()
        integers = (
            self.continuation_training_steps,
            self.residual_training_steps,
            self.representation_hidden_width,
            self.representation_evaluation_interval,
            self.maximum_residual_parameter_count,
        )
        if any(int(value) <= 0 for value in integers):
            raise ValueError("06b-c integer configuration is invalid")
        thresholds = (
            self.minimum_authentic_gain_over_relabelled_fraction,
            self.minimum_continuation_gain_over_frozen_fraction,
        )
        if any(not 0 < value < 1 for value in thresholds):
            raise ValueError("06b-c thresholds must lie in (0, 1)")

    @classmethod
    def from_mapping(
        cls, values: Mapping[str, Any]
    ) -> "CausalVoltageBridgeRepresentationConfig":
        payload = dict(values)
        for name in ("rollout_horizons_ms", "voltage_path_sample_indices", "pilot_seeds"):
            if name in payload:
                payload[name] = tuple(map(int, payload[name]))
        result = cls(**payload)
        result.validate()
        return result


if atomic.nn is not None:

    class TreeVoltageResidual(atomic.nn.Module):
        """Small correction over a frozen local bridge prediction."""

        def __init__(self, hidden_width: int, normalized_limit: float) -> None:
            super().__init__()
            self.normalized_limit = float(normalized_limit)
            self.network = atomic.nn.Sequential(
                atomic.nn.Linear(9, hidden_width),
                atomic.nn.SiLU(),
                atomic.nn.Linear(hidden_width, hidden_width),
                atomic.nn.SiLU(),
                atomic.nn.Linear(hidden_width, 1),
            )
            atomic.nn.init.zeros_(self.network[-1].weight)
            atomic.nn.init.zeros_(self.network[-1].bias)

        def forward(self, features: Any) -> Any:
            return self.normalized_limit * atomic.torch.tanh(
                self.network(features).squeeze(-1)
            )

else:  # pragma: no cover

    class TreeVoltageResidual:
        def __init__(self, *_: Any, **__: Any) -> None:
            raise RuntimeError("06b-c requires PyTorch")


class CausalVoltageBridgeRepresentationForensic(CausalVoltageStateCouplingForensic):
    config: CausalVoltageBridgeRepresentationConfig

    def __init__(
        self,
        bundle: Any,
        output_dir: Path,
        config: CausalVoltageBridgeRepresentationConfig,
        artifact_05t_source: Path,
        artifact_06b_source: Path,
        artifact_06bb_source: Path,
        *,
        code_revision: str,
    ) -> None:
        super().__init__(
            bundle,
            output_dir,
            config,
            artifact_05t_source,
            artifact_06b_source,
            code_revision=code_revision,
        )
        self.artifact_06bb_source = Path(artifact_06bb_source)
        self.frozen_bridge_models: Dict[int, Any] = {}
        self.candidate_models: Dict[Tuple[str, int], Any] = {}
        self.graphs: Dict[Tuple[str, int], np.ndarray] = {}

    def _load_frozen_bridges(self, root: Path, device: Any) -> None:
        for seed in self.config.pilot_seeds:
            checkpoint = atomic.torch.load(
                root / f"causal_voltage_bridge_seed{seed}.pt",
                map_location=device,
                weights_only=False,
            )
            if int(checkpoint.get("seed", -1)) != seed:
                raise RuntimeError(f"06b-c bridge identity mismatch for seed {seed}")
            model = self._new_bridge(device)
            model.load_state_dict(checkpoint["state_dict"])
            model.eval()
            for parameter in model.parameters():
                parameter.requires_grad_(False)
            self.frozen_bridge_models[seed] = model

    def prepare_representation_forensic(self) -> Dict[str, Any]:
        base = super().prepare_coupling_forensic()
        root, source = verified_06bb_artifact_root(
            self.artifact_06bb_source,
            self.output_dir.parent / ".06bc_artifact_cache" / "06bb",
        )
        device = next(iter(self.frozen_state_models.values())).proposal.weight.device
        self._load_frozen_bridges(root, device)
        parameter_count = int(
            sum(
                value.numel()
                for value in TreeVoltageResidual(
                    self.config.representation_hidden_width,
                    self.config.bridge_delta_limit_mv / self.config.bridge_voltage_scale_mv,
                ).parameters()
            )
        )
        blockers = []
        if parameter_count > self.config.maximum_residual_parameter_count:
            blockers.append("tree residual exceeds registered parameter ceiling")
        report = {
            **base,
            "schema_version": "06b-c-representation-contract-v1",
            "valid": bool(base.get("valid")) and not blockers,
            "blockers": blockers,
            "experiment": "causal_voltage_bridge_representation_forensic",
            "source_06bb": source,
            "representation_arms": list(REPRESENTATION_ARMS),
            "frozen_voltage_bridge_count": len(self.frozen_bridge_models),
            "residual_parameter_count": parameter_count,
            "residual_parameter_ceiling": self.config.maximum_residual_parameter_count,
            "authentic_and_relabelled_parameter_matched": True,
            "authentic_and_relabelled_initialization_matched": True,
            "authentic_and_relabelled_sample_stream_matched": True,
            "local_continuation_is_optimizer_budget_control": True,
            "state_updater_retraining_performed": False,
            "validation_state_accessed": False,
            "test_state_accessed": False,
        }
        atomic._write_json(self.output_dir / "representation_contract.json", report)
        if blockers:
            raise RuntimeError(f"06b-c preflight failed: {blockers}")
        return report

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

    def _tree_features(
        self,
        base_delta: np.ndarray,
        voltage: np.ndarray,
        state: np.ndarray,
        context: np.ndarray,
        parent: np.ndarray,
    ) -> np.ndarray:
        children, mask = self._children(parent)
        child_base = (base_delta[:, children] * mask[None, :, :]).sum(axis=2)
        child_voltage_delta = (
            (voltage[:, children] - voltage[:, :, None]) * mask[None, :, :]
        ).sum(axis=2)
        denominator = np.maximum(mask.sum(axis=1), 1.0)[None, :]
        child_base /= denominator
        child_voltage_delta /= denominator
        state_mean = state.mean(axis=2)
        context_energy = np.mean(np.abs(context), axis=2)
        return np.stack(
            (
                base_delta / self.config.bridge_voltage_scale_mv,
                base_delta[:, parent] / self.config.bridge_voltage_scale_mv,
                child_base / self.config.bridge_voltage_scale_mv,
                voltage / 100.0,
                (voltage[:, parent] - voltage) / 100.0,
                child_voltage_delta / 100.0,
                state_mean,
                state_mean[:, parent] - state_mean,
                context_energy,
            ),
            axis=-1,
        ).astype(np.float32)

    def _new_residual(self, device: Any) -> Any:
        model = TreeVoltageResidual(
            self.config.representation_hidden_width,
            self.config.bridge_delta_limit_mv / self.config.bridge_voltage_scale_mv,
        ).to(device)
        count = int(sum(value.numel() for value in model.parameters()))
        if count > self.config.maximum_residual_parameter_count:
            raise RuntimeError("06b-c tree residual parameter ceiling exceeded")
        return model

    def _predict_residual_arm(
        self, arm: str, seed: int, role: str, device: Any
    ) -> np.ndarray:
        values = self.materialized[role]
        base = self._predict_bridge(
            self.frozen_bridge_models[seed],
            values["state"],
            values["voltage_t"],
            values["context"],
            device,
        )
        if arm == "frozen_local_bridge":
            return base
        if arm == "continued_local_bridge":
            return self._predict_bridge(
                self.candidate_models[(arm, seed)],
                values["state"],
                values["voltage_t"],
                values["context"],
                device,
            )
        parent = self.graphs[(arm, seed)]
        features = self._tree_features(
            base,
            values["voltage_t"],
            self.segment_state[role],
            values["context"],
            parent,
        )
        model = self.candidate_models[(arm, seed)]
        output = np.empty_like(base)
        model.eval()
        with atomic.torch.no_grad():
            for start in range(0, len(features), 16):
                batch = atomic.torch.as_tensor(
                    features[start : start + 16],
                    dtype=atomic.torch.float32,
                    device=device,
                )
                output[start : start + 16] = (
                    model(batch).cpu().numpy() * self.config.bridge_voltage_scale_mv
                )
        return base + output

    def _voltage_metrics(self, prediction: np.ndarray, role: str) -> Dict[str, Any]:
        values = self.materialized[role]
        target = values["voltage_t1"] - values["voltage_t"]
        persistence = float(np.sqrt(np.mean(target * target)))
        rmse = float(np.sqrt(np.mean((prediction - target) ** 2)))
        active = np.abs(target) >= self.config.bridge_active_delta_threshold_mv
        active_rmse = float(np.sqrt(np.mean((prediction[active] - target[active]) ** 2))) if np.any(active) else math.nan
        active_persistence = float(np.sqrt(np.mean(target[active] ** 2))) if np.any(active) else math.nan
        return {
            "voltage_delta_rmse_mv": rmse,
            "persistence_voltage_delta_rmse_mv": persistence,
            "improvement_vs_persistence_fraction": 1.0 - rmse / max(persistence, 1e-12),
            "active_improvement_vs_persistence_fraction": 1.0 - active_rmse / max(active_persistence, 1e-12),
            "nonfinite_count": int(np.sum(~np.isfinite(prediction))),
        }

    def _train_local_continuation(self, seed: int, device: Any) -> Dict[str, Any]:
        model = self._new_bridge(device)
        model.load_state_dict(copy.deepcopy(self.frozen_bridge_models[seed].state_dict()))
        for parameter in model.parameters():
            parameter.requires_grad_(True)
        optimizer = atomic.torch.optim.AdamW(
            model.parameters(), lr=self.config.bridge_learning_rate,
            weight_decay=self.config.bridge_weight_decay,
        )
        rng = np.random.default_rng(seed + 630000)
        best_score = math.inf
        best_state: Optional[Dict[str, Any]] = None
        progress = atomic._CompactProgress(
            f"06b-c continued local seed={seed}",
            self.config.continuation_training_steps,
            self.config.bridge_progress_interval,
        )
        fit = self.materialized["fit"]
        for step in range(1, self.config.continuation_training_steps + 1):
            rows = rng.integers(0, len(fit["indices"]), size=self.config.bridge_batch_transition_count)
            segments = rng.integers(0, self.layout.segment_count, size=(self.config.bridge_batch_transition_count, self.config.bridge_segments_per_transition))
            inputs, target = self._bridge_batch("fit", rows, segments, device)
            prediction = model(*inputs)
            weight = 1.0 + self.config.bridge_active_weight * (target.abs() >= self.config.bridge_active_delta_threshold_mv / self.config.bridge_voltage_scale_mv).float()
            loss = atomic.torch.mean(weight * atomic.torch_functional.smooth_l1_loss(prediction, target, reduction="none"))
            optimizer.zero_grad(set_to_none=True); loss.backward()
            atomic.torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.bridge_gradient_clip_norm); optimizer.step()
            if step == 1 or step % self.config.representation_evaluation_interval == 0 or step == self.config.continuation_training_steps:
                candidate = self._predict_bridge(model, self.materialized["calibration"]["state"], self.materialized["calibration"]["voltage_t"], self.materialized["calibration"]["context"], device)
                score = self._voltage_metrics(candidate, "calibration")["voltage_delta_rmse_mv"]
                if score < best_score:
                    best_score = score; best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            progress.update(step, f"loss={float(loss.detach().cpu()):.4g} calV={best_score:.4g}")
        if best_state is None:
            raise RuntimeError("06b-c local continuation created no checkpoint")
        selected = self._new_bridge(device); selected.load_state_dict(best_state); selected.eval()
        self.candidate_models[("continued_local_bridge", seed)] = selected
        checkpoint = self.output_dir / f"continued_local_bridge_seed{seed}.pt"
        atomic.torch.save({"state_dict": best_state, "arm": "continued_local_bridge", "seed": seed}, checkpoint)
        return {"best_calibration_rmse_mv": best_score, "parameter_count": int(sum(v.numel() for v in selected.parameters())), "checkpoint": checkpoint.name, "checkpoint_sha256": atomic._sha256_file(checkpoint)}

    def _train_tree_residual(self, arm: str, seed: int, device: Any) -> Dict[str, Any]:
        parent = self.layout.parent_ids.copy()
        if arm == "relabelled_tree_residual":
            parent = topology_relabelled_parent_ids(parent, seed=self.config.topology_relabel_seed_offset + seed)
        self.graphs[(arm, seed)] = parent
        atomic.torch.manual_seed(seed + 631000)
        model = self._new_residual(device)
        optimizer = atomic.torch.optim.AdamW(model.parameters(), lr=self.config.bridge_learning_rate, weight_decay=self.config.bridge_weight_decay)
        rng = np.random.default_rng(seed + 632000)
        fit = self.materialized["fit"]
        frozen = self._predict_bridge(self.frozen_bridge_models[seed], fit["state"], fit["voltage_t"], fit["context"], device)
        features = self._tree_features(frozen, fit["voltage_t"], self.segment_state["fit"], fit["context"], parent)
        target = fit["voltage_t1"] - fit["voltage_t"]
        best_score = math.inf
        best_state: Optional[Dict[str, Any]] = None
        progress = atomic._CompactProgress(f"06b-c {arm} seed={seed}", self.config.residual_training_steps, self.config.bridge_progress_interval)
        for step in range(1, self.config.residual_training_steps + 1):
            rows = rng.integers(0, len(fit["indices"]), size=self.config.bridge_batch_transition_count)
            segments = rng.integers(0, self.layout.segment_count, size=(self.config.bridge_batch_transition_count, self.config.bridge_segments_per_transition))
            batch_features = features[rows[:, None], segments]
            residual_target = (target[rows[:, None], segments] - frozen[rows[:, None], segments]) / self.config.bridge_voltage_scale_mv
            tensor_features = atomic.torch.as_tensor(batch_features, dtype=atomic.torch.float32, device=device)
            tensor_target = atomic.torch.as_tensor(residual_target, dtype=atomic.torch.float32, device=device)
            prediction = model(tensor_features)
            absolute_target = atomic.torch.as_tensor(target[rows[:, None], segments], dtype=atomic.torch.float32, device=device)
            weight = 1.0 + self.config.bridge_active_weight * (absolute_target.abs() >= self.config.bridge_active_delta_threshold_mv).float()
            loss = atomic.torch.mean(weight * atomic.torch_functional.smooth_l1_loss(prediction, tensor_target, reduction="none"))
            optimizer.zero_grad(set_to_none=True); loss.backward()
            atomic.torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.bridge_gradient_clip_norm); optimizer.step()
            if step == 1 or step % self.config.representation_evaluation_interval == 0 or step == self.config.residual_training_steps:
                self.candidate_models[(arm, seed)] = model
                candidate = self._predict_residual_arm(arm, seed, "calibration", device)
                score = self._voltage_metrics(candidate, "calibration")["voltage_delta_rmse_mv"]
                if score < best_score:
                    best_score = score; best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            progress.update(step, f"loss={float(loss.detach().cpu()):.4g} calV={best_score:.4g}")
        if best_state is None:
            raise RuntimeError(f"06b-c {arm} created no checkpoint")
        selected = self._new_residual(device); selected.load_state_dict(best_state); selected.eval()
        self.candidate_models[(arm, seed)] = selected
        checkpoint = self.output_dir / f"{arm}_seed{seed}.pt"
        atomic.torch.save({"state_dict": best_state, "arm": arm, "seed": seed, "topology_sha256": hashlib.sha256(parent.tobytes()).hexdigest()}, checkpoint)
        return {"best_calibration_rmse_mv": best_score, "parameter_count": int(sum(v.numel() for v in selected.parameters())), "topology_sha256": hashlib.sha256(parent.tobytes()).hexdigest(), "checkpoint": checkpoint.name, "checkpoint_sha256": atomic._sha256_file(checkpoint)}

    def train_representation_controls(self) -> Dict[str, Any]:
        device = next(iter(self.frozen_state_models.values())).proposal.weight.device
        runs: Dict[str, Any] = {}
        for seed in self.config.pilot_seeds:
            runs[str(seed)] = {
                "continued_local_bridge": self._train_local_continuation(seed, device),
                "authentic_tree_residual": self._train_tree_residual("authentic_tree_residual", seed, device),
                "relabelled_tree_residual": self._train_tree_residual("relabelled_tree_residual", seed, device),
            }
        valid = all(
            row["authentic_tree_residual"]["parameter_count"] == row["relabelled_tree_residual"]["parameter_count"]
            for row in runs.values()
        )
        report = {"schema_version": "06b-c-training-v1", "valid": valid, "device": str(device), "runs": runs, "state_updater_retraining_performed": False}
        atomic._write_json(self.output_dir / "representation_training.json", report)
        return report

    def evaluate_representation_controls(self) -> Dict[str, Any]:
        device = next(iter(self.frozen_state_models.values())).proposal.weight.device
        per_seed = {}
        for seed in self.config.pilot_seeds:
            rows = {}
            for arm in REPRESENTATION_ARMS:
                prediction = self._predict_residual_arm(arm, seed, "development", device)
                voltage = self._voltage_metrics(prediction, "development")
                state = self._state_metrics_from_path(self.frozen_state_models[("linear_endpoint_path", seed)], "development", prediction, device)
                rows[arm] = {"voltage": voltage, "state": state}
            per_seed[str(seed)] = rows
        report = {"schema_version": "06b-c-development-v1", "valid": all(row[arm]["voltage"]["nonfinite_count"] == 0 and row[arm]["state"]["nonfinite_count"] == 0 for row in per_seed.values() for arm in REPRESENTATION_ARMS), "validation_state_accessed": False, "test_state_accessed": False, "per_seed": per_seed}
        atomic._write_json(self.output_dir / "representation_development.json", report)
        return report

    @staticmethod
    def _median(values: Sequence[float]) -> float:
        return float(np.median(np.asarray(values, dtype=np.float64)))

    def finalize_representation_forensic(self, training: Mapping[str, Any], evaluation: Mapping[str, Any]) -> Dict[str, Any]:
        per_seed = {}
        for seed, rows in evaluation["per_seed"].items():
            gain = {arm: float(row["voltage"]["improvement_vs_persistence_fraction"]) for arm, row in rows.items()}
            state = {arm: float(row["state"]["improvement_vs_persistence_fraction"]) for arm, row in rows.items()}
            per_seed[seed] = {
                "voltage_gain": gain,
                "state_gain": state,
                "continuation_voltage_gain_over_frozen": gain["continued_local_bridge"] - gain["frozen_local_bridge"],
                "authentic_voltage_gain_over_relabelled": gain["authentic_tree_residual"] - gain["relabelled_tree_residual"],
                "authentic_state_gain_over_relabelled": state["authentic_tree_residual"] - state["relabelled_tree_residual"],
            }
        continuation = self._median([row["continuation_voltage_gain_over_frozen"] for row in per_seed.values()])
        topology_v = self._median([row["authentic_voltage_gain_over_relabelled"] for row in per_seed.values()])
        topology_state = self._median([row["authentic_state_gain_over_relabelled"] for row in per_seed.values()])
        continuation_identified = continuation >= self.config.minimum_continuation_gain_over_frozen_fraction
        topology_identified = topology_v >= self.config.minimum_authentic_gain_over_relabelled_fraction and topology_state > 0 and all(row["authentic_voltage_gain_over_relabelled"] > 0 for row in per_seed.values())
        if topology_identified and continuation_identified:
            diagnosis = "OPTIMIZATION_AND_AUTHENTIC_TOPOLOGY_BOTH_CONTRIBUTE"
        elif topology_identified:
            diagnosis = "AUTHENTIC_TOPOLOGY_INFORMATION_IDENTIFIED"
        elif continuation_identified:
            diagnosis = "LOCAL_BRIDGE_OPTIMIZATION_LIMIT_IDENTIFIED"
        else:
            diagnosis = "LOCAL_VOLTAGE_REPRESENTATION_STILL_INSUFFICIENT"
        report = {
            "schema_version": "06b-c-final-report-v1",
            "valid": bool(training.get("valid") and evaluation.get("valid")),
            "component_decision_grade": True,
            "diagnosis": diagnosis,
            "optimization_limit_identified": continuation_identified,
            "authentic_topology_information_identified": topology_identified,
            "median": {"continuation_voltage_gain_over_frozen": continuation, "authentic_voltage_gain_over_relabelled": topology_v, "authentic_state_gain_over_relabelled": topology_state},
            "per_seed": per_seed,
            "registered_thresholds": {"minimum_continuation_gain_over_frozen_fraction": self.config.minimum_continuation_gain_over_frozen_fraction, "minimum_authentic_gain_over_relabelled_fraction": self.config.minimum_authentic_gain_over_relabelled_fraction},
            "state_updater_retraining_performed": False,
            "validation_state_accessed": False,
            "test_state_accessed": False,
            "full_training_authorized": False,
            "fresh_test_generation_authorized": False,
            "mass_dataset_generation_authorized": False,
            "next_step": "choose_voltage_bridge_repair_from_paired_causal_result",
            "code_revision": self.code_revision,
        }
        atomic._write_json(self.output_dir / "final_report.json", report)
        self._write_artifact_index()
        return report


__all__ = [
    "EXPECTED_06BB_ARCHIVE_SHA256",
    "EXPECTED_06BB_INDEX_SHA256",
    "EXPECTED_06BB_FINAL_SHA256",
    "REPRESENTATION_ARMS",
    "CausalVoltageBridgeRepresentationConfig",
    "CausalVoltageBridgeRepresentationForensic",
    "TreeVoltageResidual",
    "verified_06bb_artifact_root",
]
