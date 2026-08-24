"""Small, preregistered Branch-ELM benchmark on the enriched HayFlow data.

This is a sidecar benchmark.  It never selects the primary HayFlow experiment
and never turns the published 0.6376 mV soma-only metric into a comparison with
HayFlow's all-segment voltage RMSE.  Both the original frozen checkpoint and
the exact 8,002-parameter architecture retrained on train-only enriched data
are evaluated.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from ..hayflow_data.composite_flowmap import CompositeShard, CompositeTransitionStore
from ..hayflow_data.flowmap_dataset import FlowmapBundle
from . import atomic_state_dynamics_playground as atomic
from .hines_frozen_candidate_micro_rollout import verified_fresh_test_artifact_root
from .hines_regenerative_confirmation import _IndependentBundle
from .hines_regenerative_fresh_test import _FreshTestTransitionStore


ELM_INPUT_VIEWS = ("U_scheduled", "U_realized")
ORIGINAL_ELM_PARAMETER_COUNT = 8002
ORIGINAL_ELM_TEST_SOMA_RMSE_MV = 0.6375671602714604
ORIGINAL_ELM_TEST_AUC = 0.9921568089858758
# Duplicated deliberately to keep this lightweight data adapter independent
# from NeuronIO's plotting-heavy utility module.  The preflight documents the
# original convention and tests keep these values exact.
DEFAULT_Y_TRAIN_SOMA_BIAS = -67.7
DEFAULT_Y_SOMA_THRESHOLD = -55.0
DEFAULT_Y_TRAIN_SOMA_SCALE = 0.1


@dataclass(frozen=True)
class BranchELMEnrichedBenchmarkConfig:
    seeds: Tuple[int, ...] = (17, 29, 43)
    input_views: Tuple[str, ...] = ELM_INPUT_VIEWS
    fit_episode_count: int = 28
    calibration_episode_count: int = 10
    development_episode_count: int = 10
    training_steps: int = 300
    evaluation_interval: int = 50
    progress_interval: int = 25
    learning_rate: float = 0.0005
    weight_decay: float = 0.0
    gradient_clip_norm: float = 1.0
    train_burn_in_ms: int = 8
    fresh_test_burn_in_ms: int = 4
    spike_loss_weight: float = 0.5
    soma_loss_weight: float = 0.5
    expected_dendritic_segment_count: int = 639
    expected_parameter_count: int = ORIGINAL_ELM_PARAMETER_COUNT
    primary_metric: str = "clipped_soma_rmse_mv"

    def validate(self) -> None:
        if tuple(self.seeds) != (17, 29, 43):
            raise ValueError("Branch-ELM benchmark requires registered paired seeds")
        if tuple(self.input_views) != ELM_INPUT_VIEWS:
            raise ValueError("Branch-ELM benchmark requires scheduled and realized views")
        positive = (
            self.fit_episode_count,
            self.calibration_episode_count,
            self.development_episode_count,
            self.training_steps,
            self.evaluation_interval,
            self.progress_interval,
            self.train_burn_in_ms,
            self.fresh_test_burn_in_ms,
        )
        if any(int(value) <= 0 for value in positive):
            raise ValueError("Branch-ELM integer configuration is invalid")
        if not 0 < self.learning_rate < 1 or self.weight_decay < 0:
            raise ValueError("Branch-ELM optimizer configuration is invalid")
        if self.spike_loss_weight + self.soma_loss_weight != 1.0:
            raise ValueError("Branch-ELM loss weights must sum to one")
        if self.expected_dendritic_segment_count != 639:
            raise ValueError("original Branch-ELM contract requires 639 dendritic sites")
        if self.expected_parameter_count != ORIGINAL_ELM_PARAMETER_COUNT:
            raise ValueError("original Branch-ELM contract requires 8,002 parameters")

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "BranchELMEnrichedBenchmarkConfig":
        payload = dict(values)
        for name in ("seeds", "input_views"):
            if name in payload:
                payload[name] = tuple(payload[name])
        result = cls(**payload)
        result.validate()
        return result


def _binary_auc(target: np.ndarray, score: np.ndarray) -> float:
    target = np.asarray(target, dtype=np.int8)
    score = np.asarray(score, dtype=np.float64)
    positive = int(np.sum(target == 1)); negative = int(np.sum(target == 0))
    if positive == 0 or negative == 0:
        return math.nan
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score), dtype=np.float64)
    start = 0
    while start < len(score):
        stop = start + 1
        while stop < len(score) and score[order[stop]] == score[order[start]]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + stop - 1) + 1.0
        start = stop
    return float((ranks[target == 1].sum() - positive * (positive + 1) / 2) / (positive * negative))


class BranchELMEnrichedBenchmark:
    """Exact 8,002-parameter Branch-ELM on train-only and sealed fresh data."""

    def __init__(
        self,
        bundle: Any,
        output_dir: Path,
        config: BranchELMEnrichedBenchmarkConfig,
        fresh_test_source: Path,
        repository_root: Path,
        *,
        code_revision: str,
    ) -> None:
        config.validate()
        self.bundle = bundle
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=False)
        self.config = config
        self.fresh_test_source = Path(fresh_test_source)
        self.repository_root = Path(repository_root)
        self.code_revision = str(code_revision)
        self.store = CompositeTransitionStore(bundle)
        self.fresh_store: Optional[_FreshTestTransitionStore] = None
        self.roles: Dict[str, List[str]] = {}
        self.cache: Dict[Tuple[int, str, str], Dict[str, np.ndarray]] = {}
        self.models: Dict[Tuple[str, int], Any] = {}

    def close(self) -> None:
        self.store.close()
        if self.fresh_store is not None:
            self.fresh_store.close()

    @staticmethod
    def _hash_score(value: str, tag: str) -> str:
        return hashlib.sha256(f"branch-elm|{tag}|{value}".encode()).hexdigest()

    def _make_fresh_store(self, root: Path) -> _FreshTestTransitionStore:
        manifest = json.loads((root / "dataset_manifest.json").read_text())
        state_schema = json.loads((root / "state_schema.json").read_text())
        teacher_manifest = json.loads((root / "manifest.json").read_text())
        teacher_report = json.loads((root / "teacher_fresh_test_report.json").read_text())
        transition = root / "transition_dataset.h5"
        layout_bundle = FlowmapBundle(
            root=root,
            transition_path=transition,
            manifest=manifest,
            state_schema=state_schema,
            teacher_manifest=teacher_manifest,
            validation_report=teacher_report,
            artifact_validation={"valid": True, "fresh_test": True},
        )
        shard = CompositeShard(
            shard_id="05jo_fresh_test",
            root=root,
            transition_path=transition,
            transition_count=768,
            transition_sha256=str(teacher_report["transition_store_sha256"]),
            dataset_manifest=manifest,
            validation_report=teacher_report,
            offset=0,
        )
        independent = _IndependentBundle(
            root / "dataset_manifest.json", manifest, shard, layout_bundle,
            str(teacher_report["transition_store_sha256"]),
        )
        return _FreshTestTransitionStore(independent)

    @staticmethod
    def _trajectory_has_somatic_current(store: Any, trajectory: str) -> bool:
        return any(
            action.get("kind") == "somatic_current"
            for index in store.trajectory_indices[trajectory]
            for action in store.actions(int(index), "U_scheduled")
        )

    @staticmethod
    def _trajectory_has_somatic_spike(store: Any, trajectory: str) -> bool:
        return any(
            event.get("kind") == "somatic_spike"
            for index in store.trajectory_indices[trajectory]
            for event in store.events(int(index))
        )

    def _select_roles(self) -> Dict[str, List[str]]:
        eligible = []
        excluded = []
        for trajectory, indices in self.store.trajectory_indices.items():
            split = str(self.store.metadata["split"][int(indices[0])])
            if split != "train":
                continue
            if len(indices) <= self.config.train_burn_in_ms:
                excluded.append((trajectory, "too_short")); continue
            if self._trajectory_has_somatic_current(self.store, trajectory):
                excluded.append((trajectory, "somatic_current_not_representable")); continue
            eligible.append((trajectory, self._trajectory_has_somatic_spike(self.store, trajectory)))
        positive = sorted((name for name, flag in eligible if flag), key=lambda value: self._hash_score(value, "positive"))
        negative = sorted((name for name, flag in eligible if not flag), key=lambda value: self._hash_score(value, "negative"))
        ordered = []
        for index in range(max(len(positive), len(negative))):
            if index < len(positive): ordered.append(positive[index])
            if index < len(negative): ordered.append(negative[index])
        counts = {
            "fit": self.config.fit_episode_count,
            "calibration": self.config.calibration_episode_count,
            "development": self.config.development_episode_count,
        }
        if len(ordered) < sum(counts.values()):
            raise RuntimeError(f"only {len(ordered)} Branch-ELM-compatible train episodes")
        roles: Dict[str, List[str]] = {}
        offset = 0
        for role, count in counts.items():
            roles[role] = ordered[offset : offset + count]; offset += count
        self.exclusion_report = {
            "eligible_train_episode_count": len(eligible),
            "excluded_train_episode_count": len(excluded),
            "excluded_by_reason": {reason: sum(item[1] == reason for item in excluded) for reason in sorted({item[1] for item in excluded})},
            "all_compatible_train_episodes_assigned": len(ordered) == sum(counts.values()),
        }
        return roles

    def _validate_morphology(self) -> Dict[int, int]:
        segments = self.store.layout.segments
        dendritic = [row for row in segments if str(row["region"]) not in {"soma", "ais", "axon"}]
        ids = [int(row["id"]) for row in dendritic]
        if len(ids) != self.config.expected_dendritic_segment_count or ids != list(range(1, 640)):
            raise RuntimeError("HayFlow dendritic sites do not preserve the original 639-site ordering")
        return {segment_id: segment_id - 1 for segment_id in ids}

    def prepare_benchmark(self) -> Dict[str, Any]:
        fresh_root, fresh_result, fresh_contract = verified_fresh_test_artifact_root(
            self.fresh_test_source,
            self.output_dir.parent / ".06bc_elm_artifact_cache" / "05jo",
        )
        self.fresh_store = self._make_fresh_store(fresh_root)
        self.segment_to_channel = self._validate_morphology()
        self.roles = self._select_roles()
        model_config = json.loads((self.repository_root / "models/num_memory_20/model_config.json").read_text())
        model_stats = json.loads((self.repository_root / "models/num_memory_20/model_stats.json").read_text())
        blockers = []
        if int(model_stats["model_param_count"]) != ORIGINAL_ELM_PARAMETER_COUNT:
            blockers.append("repository Branch-ELM is not the registered 8,002-parameter model")
        if model_config != {
            "input_to_synapse_routing": "neuronio_routing", "learn_memory_tau": False,
            "memory_tau_max": 150.0, "memory_tau_min": 1.0, "num_branch": 45,
            "num_input": 1278, "num_memory": 20, "num_output": 2,
            "num_synapse_per_branch": 100,
        }:
            blockers.append("repository Branch-ELM configuration differs from num_memory_20")
        fresh_compatible = [trajectory for trajectory in self.fresh_store.trajectory_indices if not self._trajectory_has_somatic_current(self.fresh_store, trajectory)]
        report = {
            "schema_version": "06b-c-elm-benchmark-contract-v1",
            "valid": not blockers,
            "blockers": blockers,
            "sidecar_only": True,
            "cannot_select_or_veto_primary_06b_c": True,
            "model_config": model_config,
            "trainable_parameter_count": int(model_stats["model_param_count"]),
            "input_views": list(self.config.input_views),
            "dendritic_segment_count": len(self.segment_to_channel),
            "roles": {key: len(value) for key, value in self.roles.items()},
            "role_overlap": {f"{a}-{b}": sorted(set(self.roles[a]) & set(self.roles[b])) for a, b in (("fit", "calibration"), ("fit", "development"), ("calibration", "development"))},
            "exclusions": self.exclusion_report,
            "fresh_test": fresh_contract,
            "fresh_test_episode_count": len(self.fresh_store.trajectory_indices),
            "fresh_test_compatible_episode_count": len(fresh_compatible),
            "fresh_test_excluded_somatic_current_count": len(self.fresh_store.trajectory_indices) - len(fresh_compatible),
            "recurrent_state_contract": {
                "reset": "zero at each independent teacher episode",
                "train_burn_in_ms": self.config.train_burn_in_ms,
                "fresh_test_burn_in_ms": self.config.fresh_test_burn_in_ms,
                "original_paper_burn_in_ms": 150,
                "limitation": "HayFlow diagnostic episodes are much shorter than the original 500 ms Branch-ELM windows, so this remains a bounded canary rather than an original-training-budget replication.",
            },
            "training_reads_splits": ["train"],
            "fresh_test_used_for_selection": False,
            "published_reference": {"dataset": "original NeuronIO", "soma_rmse_mv": ORIGINAL_ELM_TEST_SOMA_RMSE_MV, "auc": ORIGINAL_ELM_TEST_AUC},
            "metric_warning": "Published ELM RMSE is clipped soma-only; HayFlow ~0.40 mV is all-642-segment aggregate. Same fresh transitions do not make those target scopes identical.",
            "code_revision": self.code_revision,
        }
        atomic._write_json(self.output_dir / "benchmark_contract.json", report)
        if blockers:
            raise RuntimeError(f"Branch-ELM benchmark preflight failed: {blockers}")
        return report

    @staticmethod
    def _is_inhibitory(action: Mapping[str, Any]) -> bool:
        text = str(action.get("synapse_type", "")).lower()
        return "gaba" in text or "inh" in text

    def _episode(self, store: Any, trajectory: str, view: str) -> Dict[str, np.ndarray]:
        key = (id(store), trajectory, view)
        if key in self.cache:
            return self.cache[key]
        indices = np.asarray(store.trajectory_indices[trajectory], dtype=np.int64)
        x = np.zeros((len(indices), 1278), dtype=np.float32)
        spike = np.zeros(len(indices), dtype=np.float32)
        collision_count = 0
        ignored_non_dendritic = 0
        for time_index, logical_index in enumerate(indices):
            for action in store.actions(int(logical_index), view):
                if action.get("kind") != "synaptic_event":
                    continue
                if view == "U_realized" and not bool(action.get("release_success", False)):
                    continue
                segment_id = int(action["segment_id"])
                if segment_id not in self.segment_to_channel:
                    ignored_non_dendritic += 1; continue
                channel = self.segment_to_channel[segment_id]
                value = 1.0
                if self._is_inhibitory(action):
                    channel += 639; value = -1.0
                if x[time_index, channel] != 0:
                    collision_count += 1
                x[time_index, channel] = value
            spike[time_index] = float(any(event.get("kind") == "somatic_spike" for event in store.events(int(logical_index))))
        voltage_t = store.read_state(indices, "t", categories=("voltage",))[:, 0].astype(np.float32)
        voltage_t1 = store.read_state(indices, "t_plus_1", categories=("voltage",))[:, 0].astype(np.float32)
        result = {"x": x, "spike": spike, "voltage_t": voltage_t, "voltage_t1": voltage_t1, "collision_count": np.asarray(collision_count), "ignored_non_dendritic": np.asarray(ignored_non_dendritic)}
        self.cache[key] = result
        return result

    def _new_model(self, device: Any) -> Any:
        from ..expressive_leaky_memory_neuron import ELM

        values = json.loads((self.repository_root / "models/num_memory_20/model_config.json").read_text())
        model = ELM(**values).to(device)
        count = int(sum(value.numel() for value in model.parameters() if value.requires_grad))
        if count != ORIGINAL_ELM_PARAMETER_COUNT:
            raise RuntimeError(f"Branch-ELM parameter count is {count}, expected 8,002")
        return model

    def _loss(self, output: Any, episode: Mapping[str, np.ndarray], burn_in: int, device: Any) -> Any:
        target_spike = atomic.torch.as_tensor(episode["spike"][burn_in:], dtype=atomic.torch.float32, device=device)
        clipped = np.minimum(episode["voltage_t1"][burn_in:], DEFAULT_Y_SOMA_THRESHOLD)
        target_soma = atomic.torch.as_tensor((clipped - DEFAULT_Y_TRAIN_SOMA_BIAS) * DEFAULT_Y_TRAIN_SOMA_SCALE, dtype=atomic.torch.float32, device=device)
        prediction = output[0, burn_in:]
        spike_loss = atomic.torch_functional.binary_cross_entropy_with_logits(prediction[:, 0], target_spike)
        soma_loss = atomic.torch_functional.mse_loss(prediction[:, 1], target_soma)
        return self.config.spike_loss_weight * spike_loss + self.config.soma_loss_weight * soma_loss

    def _evaluate(self, model: Any, store: Any, trajectories: Sequence[str], view: str, burn_in: int, device: Any) -> Dict[str, Any]:
        target_voltage = []; raw_target_voltage = []; predicted_voltage = []; spike_target = []; spike_score = []
        collisions = ignored = 0
        model.eval()
        with atomic.torch.no_grad():
            for trajectory in trajectories:
                episode = self._episode(store, trajectory, view)
                x = atomic.torch.as_tensor(episode["x"][None], dtype=atomic.torch.float32, device=device)
                output = model(x)[0]
                probability = atomic.torch.sigmoid(output[:, 0]).cpu().numpy()
                voltage = output[:, 1].cpu().numpy() / DEFAULT_Y_TRAIN_SOMA_SCALE + DEFAULT_Y_TRAIN_SOMA_BIAS
                target_voltage.extend(np.minimum(episode["voltage_t1"][burn_in:], DEFAULT_Y_SOMA_THRESHOLD).tolist())
                raw_target_voltage.extend(episode["voltage_t1"][burn_in:].tolist())
                predicted_voltage.extend(voltage[burn_in:].tolist())
                spike_target.extend(episode["spike"][burn_in:].tolist())
                spike_score.extend(probability[burn_in:].tolist())
                collisions += int(episode["collision_count"]); ignored += int(episode["ignored_non_dendritic"])
        target = np.asarray(target_voltage); raw_target = np.asarray(raw_target_voltage); prediction = np.asarray(predicted_voltage)
        labels = np.asarray(spike_target, dtype=np.int8); scores = np.asarray(spike_score)
        binary = scores >= 0.5; tp = int(np.sum(binary & (labels == 1))); fp = int(np.sum(binary & (labels == 0))); fn = int(np.sum((~binary) & (labels == 1)))
        precision = tp / max(tp + fp, 1); recall = tp / max(tp + fn, 1)
        return {
            "clipped_soma_rmse_mv": float(np.sqrt(np.mean((prediction - target) ** 2))),
            "raw_soma_rmse_mv_secondary": float(np.sqrt(np.mean((prediction - raw_target) ** 2))),
            "soma_mae_mv": float(np.mean(np.abs(prediction - target))),
            "spike_auc": _binary_auc(labels, scores),
            "spike_f1_at_0_5": 2 * precision * recall / max(precision + recall, 1e-12),
            "spike_positive_count": int(labels.sum()),
            "evaluated_timestep_count": len(labels),
            "episode_count": len(trajectories),
            "burn_in_ms": burn_in,
            "input_collision_count": collisions,
            "ignored_non_dendritic_event_count": ignored,
        }

    def _train(self, view: str, seed: int, device: Any) -> Dict[str, Any]:
        atomic.torch.manual_seed(seed)
        model = self._new_model(device)
        optimizer = atomic.torch.optim.AdamW(model.parameters(), lr=self.config.learning_rate, weight_decay=self.config.weight_decay)
        rng = np.random.default_rng(seed + 640000)
        best_score = math.inf; best_state: Optional[Dict[str, Any]] = None
        curve = []
        progress = atomic._CompactProgress(f"06b-c Branch-ELM {view} seed={seed}", self.config.training_steps, self.config.progress_interval)
        for step in range(1, self.config.training_steps + 1):
            trajectory = self.roles["fit"][int(rng.integers(0, len(self.roles["fit"])))]
            episode = self._episode(self.store, trajectory, view)
            x = atomic.torch.as_tensor(episode["x"][None], dtype=atomic.torch.float32, device=device)
            output = model(x); loss = self._loss(output, episode, self.config.train_burn_in_ms, device)
            optimizer.zero_grad(set_to_none=True); loss.backward()
            atomic.torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.gradient_clip_norm); optimizer.step()
            if step == 1 or step % self.config.evaluation_interval == 0 or step == self.config.training_steps:
                metrics = self._evaluate(model, self.store, self.roles["calibration"], view, self.config.train_burn_in_ms, device)
                score = float(metrics[self.config.primary_metric]); curve.append({"step": step, "train_loss": float(loss.detach().cpu()), "calibration_clipped_soma_rmse_mv": score})
                if score < best_score:
                    best_score = score; best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            progress.update(step, f"loss={float(loss.detach().cpu()):.4g} calSoma={best_score:.4g}mV")
        if best_state is None:
            raise RuntimeError("Branch-ELM training created no checkpoint")
        selected = self._new_model(device); selected.load_state_dict(best_state); selected.eval()
        self.models[(view, seed)] = selected
        checkpoint = self.output_dir / f"branch_elm_8002_{view}_seed{seed}.pt"
        atomic.torch.save({"state_dict": best_state, "view": view, "seed": seed, "configuration": asdict(self.config)}, checkpoint)
        return {"seed": seed, "view": view, "best_calibration_clipped_soma_rmse_mv": best_score, "development": self._evaluate(selected, self.store, self.roles["development"], view, self.config.train_burn_in_ms, device), "learning_curve": curve, "checkpoint": checkpoint.name, "checkpoint_sha256": atomic._sha256_file(checkpoint)}

    def run_benchmark(self) -> Dict[str, Any]:
        if self.fresh_store is None:
            raise RuntimeError("prepare_benchmark() must run first")
        device = atomic.torch.device("cuda" if atomic.torch.cuda.is_available() else "cpu")
        compatible_fresh = [trajectory for trajectory in self.fresh_store.trajectory_indices if not self._trajectory_has_somatic_current(self.fresh_store, trajectory)]
        trained: Dict[str, Any] = {}
        for view in self.config.input_views:
            trained[view] = {}
            for seed in self.config.seeds:
                run = self._train(view, seed, device)
                trained[view][str(seed)] = run
        # Only after every retrained checkpoint has been selected from the
        # train-derived calibration role may fresh-test outcomes be read.
        frozen = self._new_model(device)
        frozen.load_state_dict(atomic.torch.load(self.repository_root / "models/num_memory_20/neuronio_best_model_state.pt", map_location=device, weights_only=False)); frozen.eval()
        zero_shot = {view: self._evaluate(frozen, self.fresh_store, compatible_fresh, view, self.config.fresh_test_burn_in_ms, device) for view in self.config.input_views}
        for view in self.config.input_views:
            for seed in self.config.seeds:
                trained[view][str(seed)]["fresh_test"] = self._evaluate(self.models[(view, seed)], self.fresh_store, compatible_fresh, view, self.config.fresh_test_burn_in_ms, device)
        report = {"schema_version": "06b-c-branch-elm-results-v1", "valid": True, "device": str(device), "zero_shot_original_checkpoint": zero_shot, "retrained_exact_architecture": trained, "fresh_test_used_for_selection": False, "trainable_parameter_count": ORIGINAL_ELM_PARAMETER_COUNT}
        atomic._write_json(self.output_dir / "benchmark_results.json", report)
        return report

    def finalize(self, contract: Mapping[str, Any], results: Mapping[str, Any]) -> Dict[str, Any]:
        medians = {}
        for view in self.config.input_views:
            values = [float(row["fresh_test"]["clipped_soma_rmse_mv"]) for row in results["retrained_exact_architecture"][view].values()]
            medians[view] = float(np.median(values))
        full_fresh_coverage = int(contract["fresh_test_compatible_episode_count"]) == int(contract["fresh_test_episode_count"])
        report = {
            "schema_version": "06b-c-branch-elm-final-v1",
            "valid": bool(contract.get("valid") and results.get("valid")),
            "sidecar_only": True,
            "primary_experiment_replaced": False,
            "architecture": "published Branch-ELM num_memory_20",
            "trainable_parameter_count": ORIGINAL_ELM_PARAMETER_COUNT,
            "fresh_test_median_retrained_clipped_soma_rmse_mv": medians,
            "zero_shot_original_checkpoint": results["zero_shot_original_checkpoint"],
            "published_original_dataset_reference": {"clipped_soma_rmse_mv": ORIGINAL_ELM_TEST_SOMA_RMSE_MV, "spike_auc": ORIGINAL_ELM_TEST_AUC},
            "comparability": {
                "same_fresh_transitions_as_hayflow_0_40": full_fresh_coverage,
                "same_fresh_dataset_but_compatible_subset_only": not full_fresh_coverage,
                "same_target_scope_as_hayflow_0_40": False,
                "reason": "Branch-ELM reports clipped soma-only RMSE; HayFlow 0.40 mV is an aggregate over all 642 segment voltages.",
                "claim_elm_beats_or_loses_to_hayflow_authorized": False,
            },
            "fresh_test_status": "retrospective_for_project_but_not_used_for_elm_selection",
            "scope": "bounded_enriched_data_canary_not_original_budget_replication",
            "fresh_test_used_for_selection": False,
            "full_training_authorized": False,
            "code_revision": self.code_revision,
        }
        atomic._write_json(self.output_dir / "final_report.json", report)
        self._write_artifact_index()
        return report

    def _write_artifact_index(self) -> None:
        artifacts = []
        for path in sorted(self.output_dir.rglob("*")):
            if not path.is_file() or path.name == "artifact_index.json":
                continue
            artifacts.append({"path": path.relative_to(self.output_dir).as_posix(), "size_bytes": path.stat().st_size, "sha256": atomic._sha256_file(path)})
        atomic._write_json(self.output_dir / "artifact_index.json", {"schema_version": "06b-c-branch-elm-index-v1", "artifacts": artifacts})


__all__ = [
    "ELM_INPUT_VIEWS", "ORIGINAL_ELM_PARAMETER_COUNT",
    "ORIGINAL_ELM_TEST_SOMA_RMSE_MV", "ORIGINAL_ELM_TEST_AUC",
    "BranchELMEnrichedBenchmarkConfig", "BranchELMEnrichedBenchmark",
]
