"""GPU training for the paper-scale information-matched soma comparison."""

from __future__ import annotations

import copy
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import numpy as np

from src.hayflow_model.branch_elm_information_matched_transition import (
    InformationMatchedBranchELM,
    InformationMatchedBridgeAdapter,
)
from src.hayflow_model.causal_voltage_state_coupling_forensic import CausalVoltageBridge


@dataclass(frozen=True)
class MatchedTrainingConfig:
    seeds: tuple[int, ...] = (61017, 61029, 61043)
    training_steps: int = 3000
    checkpoints: tuple[int, ...] = (100, 300, 1000, 3000)
    batch_size: int = 4096
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    gradient_clip_norm: float = 1.0
    voltage_scale_mv: float = 20.0
    delta_limit_mv: float = 100.0
    active_delta_threshold_mv: float = 5.0
    active_weight: float = 4.0
    normalization_sample_limit: int = 1_000_000
    evaluation_sample_limit: int = 1_000_000
    progress_interval: int = 50
    expected_input_width: int = 76
    expected_branch_elm_parameters: int = 8002
    expected_giada_parameters: int = 8985

    def validate(self) -> None:
        if not self.seeds or self.training_steps <= 0 or self.batch_size <= 0:
            raise ValueError("invalid matched training size")
        if not self.checkpoints or max(self.checkpoints) != self.training_steps:
            raise ValueError("final checkpoint must equal training_steps")
        if any(step <= 0 for step in self.checkpoints):
            raise ValueError("checkpoint steps must be positive")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("invalid optimizer configuration")
        if self.voltage_scale_mv <= 0 or self.delta_limit_mv <= 0:
            raise ValueError("invalid voltage scaling")

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "MatchedTrainingConfig":
        payload = dict(values)
        for key in ("seeds", "checkpoints"):
            if key in payload:
                payload[key] = tuple(map(int, payload[key]))
        result = cls(**payload)
        result.validate()
        return result


class LeanSomaCorpus:
    """Lazy reader for validated soma_paper shards."""

    def __init__(self, root: Path) -> None:
        try:
            import h5py
        except ImportError as error:  # pragma: no cover
            raise RuntimeError("paper-scale training requires h5py") from error
        self.h5py = h5py
        self.root = Path(root)
        self.paths = sorted((self.root / "shards").glob("shard-*.h5"))
        if not self.paths:
            raise FileNotFoundError(f"no paper-scale shards under {self.root}")
        self.rows: Dict[int, List[tuple[Path, np.ndarray]]] = {0: [], 1: []}
        self.metadata: Dict[str, Any] | None = None
        for path in self.paths:
            with h5py.File(path, "r") as handle:
                metadata = json.loads(handle.attrs["schema_metadata_json"])
                if metadata.get("storage_profile") != "soma_paper":
                    raise RuntimeError(f"{path.name} is not a soma_paper shard")
                if self.metadata is None:
                    self.metadata = metadata
                else:
                    stable = ("mechanism_group_names", "ion_names", "causal_drive_features", "segment_ids", "mechanism_presence", "segment_static", "region_names", "segment_region_ids")
                    if any(self.metadata[key] != metadata[key] for key in stable):
                        raise RuntimeError("paper-scale shard feature schemas differ")
                split = np.asarray(handle["split_code"][...], dtype=np.uint8)
                for code in (0, 1):
                    indices = np.flatnonzero(split == code)
                    if len(indices):
                        self.rows[code].append((path, indices))
        if not self.rows[0] or not self.rows[1]:
            raise RuntimeError("paper-scale corpus requires train and validation rows")
        assert self.metadata is not None
        self.train_count = sum(len(indices) for _, indices in self.rows[0])
        self.validation_count = sum(len(indices) for _, indices in self.rows[1])
        self._handles: Dict[Path, Any] = {}

    def _handle(self, path: Path) -> Any:
        if path not in self._handles:
            self._handles[path] = self.h5py.File(path, "r")
        return self._handles[path]

    def close(self) -> None:
        for handle in self._handles.values():
            handle.close()
        self._handles.clear()

    @staticmethod
    def _read_sorted(handle: Any, name: str, indices: np.ndarray) -> np.ndarray:
        # h5py requires strictly increasing fancy indices. Sampling with
        # replacement produces duplicates, so read each unique row once and
        # reconstruct the requested order afterward.
        unique, inverse = np.unique(indices, return_inverse=True)
        values = np.asarray(handle[name][unique])
        return values[inverse]

    def sample_raw(self, split_code: int, count: int, rng: np.random.Generator) -> Dict[str, np.ndarray]:
        groups = self.rows[int(split_code)]
        sizes = np.asarray([len(indices) for _, indices in groups], dtype=np.float64)
        selected_group = int(rng.choice(len(groups), p=sizes / sizes.sum()))
        path, available = groups[selected_group]
        positions = rng.integers(0, len(available), size=int(count))
        indices = available[positions]
        handle = self._handle(path)
        names = (
            "voltage_t_mv", "voltage_t_plus_1_mv", "parent_delta_t_mv",
            "mean_child_delta_t_mv", "mechanism_state_t", "ion_state_t",
            "causal_drive",
        )
        return {name: self._read_sorted(handle, name, indices)[:, 0] if handle[name].ndim == 2 else self._read_sorted(handle, name, indices)[:, 0, :] for name in names}

    def iter_raw(self, split_code: int, limit: int, chunk: int = 65536) -> Iterable[Dict[str, np.ndarray]]:
        remaining = int(limit)
        for path, available in self.rows[int(split_code)]:
            if remaining <= 0:
                break
            handle = self._handle(path)
            take = available[:remaining]
            for start in range(0, len(take), chunk):
                indices = take[start : start + chunk]
                names = (
                    "voltage_t_mv", "voltage_t_plus_1_mv", "parent_delta_t_mv",
                    "mean_child_delta_t_mv", "mechanism_state_t", "ion_state_t",
                    "causal_drive",
                )
                yield {name: np.asarray(handle[name][indices])[:, 0] if handle[name].ndim == 2 else np.asarray(handle[name][indices])[:, 0, :] for name in names}
            remaining -= len(take)


class FeatureTransform:
    def __init__(self, corpus: LeanSomaCorpus, config: MatchedTrainingConfig) -> None:
        self.corpus = corpus
        self.config = config
        metadata = corpus.metadata
        self.presence = np.asarray(metadata["mechanism_presence"][0], dtype=np.float32)
        self.static = np.asarray(metadata["segment_static"][0], dtype=np.float32)
        self.region_names = list(metadata["region_names"])
        self.region_id = int(metadata["segment_region_ids"][0])
        self.state_center = np.zeros(len(self.presence), dtype=np.float32)
        self.state_scale = np.ones(len(self.presence), dtype=np.float32)
        self.ion_center = np.zeros(len(metadata["ion_names"]), dtype=np.float32)
        self.ion_scale = np.ones(len(metadata["ion_names"]), dtype=np.float32)
        self.slices: Dict[str, slice] = {}

    @staticmethod
    def _robust(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        center = np.median(values, axis=0)
        q25, q75 = np.percentile(values, (25.0, 75.0), axis=0)
        scale = (q75 - q25) / 1.349
        std = np.std(values, axis=0)
        return center, np.where(scale > 1e-8, scale, np.where(std > 1e-8, std, 1.0))

    def fit(self) -> None:
        target = min(self.config.normalization_sample_limit, self.corpus.train_count)
        chunks = list(self.corpus.iter_raw(0, target))
        raw = {
            name: np.concatenate([chunk[name] for chunk in chunks], axis=0)
            for name in chunks[0]
        }
        state = np.log(
            np.clip(raw["mechanism_state_t"], 1e-6, 1.0 - 1e-6)
            / (1.0 - np.clip(raw["mechanism_state_t"], 1e-6, 1.0 - 1e-6))
        )
        active = self.presence.astype(bool)
        if np.any(active):
            self.state_center[active], self.state_scale[active] = self._robust(state[:, active])
        ions = np.log1p(np.maximum(raw["ion_state_t"], 0.0))
        if ions.shape[1]:
            self.ion_center, self.ion_scale = self._robust(ions)
        widths = {
            "axial_voltage": 3,
            "mechanism_state": len(self.presence),
            "mechanism_presence": len(self.presence),
            "causal_context": raw["causal_drive"].shape[1] + ions.shape[1],
            "segment_static": len(self.static),
            "region_one_hot": len(self.region_names),
        }
        start = 0
        for name, width in widths.items():
            self.slices[name] = slice(start, start + width)
            start += width
        if start % 2:
            self.slices["zero_padding"] = slice(start, start + 1)
            start += 1
        self.width = start
        if self.width != self.config.expected_input_width:
            raise RuntimeError(f"paper-scale common input width {self.width} != {self.config.expected_input_width}")

    def apply(self, raw: Mapping[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        state = np.log(
            np.clip(raw["mechanism_state_t"], 1e-6, 1.0 - 1e-6)
            / (1.0 - np.clip(raw["mechanism_state_t"], 1e-6, 1.0 - 1e-6))
        )
        state = (state - self.state_center) / self.state_scale
        state[:, ~self.presence.astype(bool)] = 0.0
        ions = (np.log1p(np.maximum(raw["ion_state_t"], 0.0)) - self.ion_center) / self.ion_scale
        region = np.zeros((len(state), len(self.region_names)), dtype=np.float32)
        region[:, self.region_id] = 1.0
        blocks = [
            np.stack((raw["voltage_t_mv"], raw["parent_delta_t_mv"], raw["mean_child_delta_t_mv"]), axis=-1) / 100.0,
            state,
            np.broadcast_to(self.presence, state.shape),
            np.concatenate((raw["causal_drive"], ions), axis=-1),
            np.broadcast_to(self.static, (len(state), len(self.static))),
            region,
        ]
        if "zero_padding" in self.slices:
            blocks.append(np.zeros((len(state), 1), dtype=np.float32))
        features = np.concatenate(blocks, axis=-1).astype(np.float32)
        target = ((raw["voltage_t_plus_1_mv"] - raw["voltage_t_mv"]) / self.config.voltage_scale_mv).astype(np.float32)
        return features, target

    def to_dict(self) -> Dict[str, Any]:
        return {
            "fit_split": "train",
            "state_transform": "logit_clip_1e-6",
            "ion_transform": "log1p_nonnegative",
            "state_center": self.state_center.tolist(),
            "state_scale": self.state_scale.tolist(),
            "ion_center": self.ion_center.tolist(),
            "ion_scale": self.ion_scale.tolist(),
            "feature_slices": {name: [value.start, value.stop] for name, value in self.slices.items()},
            "input_width": self.width,
        }


class PaperScaleMatchedTrainer:
    def __init__(self, corpus_root: Path, output_dir: Path, config: MatchedTrainingConfig, *, code_revision: str) -> None:
        try:
            import torch
        except ImportError as error:  # pragma: no cover
            raise RuntimeError("paper-scale training requires PyTorch") from error
        config.validate()
        self.torch = torch
        self.corpus = LeanSomaCorpus(corpus_root)
        self.output_dir = Path(output_dir)
        if self.output_dir.exists():
            raise FileExistsError(f"refusing to overwrite {self.output_dir}")
        self.output_dir.mkdir(parents=True)
        self.config = config
        self.code_revision = str(code_revision)
        self.transform = FeatureTransform(self.corpus, config)
        self.transform.fit()
        (self.output_dir / "normalization.json").write_text(
            json.dumps(self.transform.to_dict(), indent=2, sort_keys=True), encoding="utf-8"
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.device.type != "cuda":
            raise RuntimeError("paper-scale matched training requires a CUDA GPU pod")

    @staticmethod
    def _trainable_count(model: Any) -> int:
        return sum(value.numel() for value in model.parameters() if value.requires_grad)

    def _models(self) -> Dict[str, Any]:
        meta = self.corpus.metadata
        bridge = CausalVoltageBridge(
            state_width=len(meta["mechanism_group_names"]),
            presence_width=len(meta["mechanism_group_names"]),
            context_width=len(meta["causal_drive_features"]) + len(meta["ion_names"]),
            static_width=len(meta["segment_static"][0]),
            region_count=len(meta["region_names"]),
            region_embedding_width=8,
            hidden_width=64,
            normalized_delta_limit=self.config.delta_limit_mv / self.config.voltage_scale_mv,
        )
        models = {
            "branch_elm_core": InformationMatchedBranchELM(
                self.transform.width,
                self.config.delta_limit_mv / self.config.voltage_scale_mv,
            ),
            "giada_voltage_bridge": InformationMatchedBridgeAdapter(
                bridge, self.transform.slices
            ),
        }
        counts = {name: self._trainable_count(model) for name, model in models.items()}
        expected = {
            "branch_elm_core": self.config.expected_branch_elm_parameters,
            "giada_voltage_bridge": self.config.expected_giada_parameters,
        }
        if counts != expected:
            raise RuntimeError(f"matched parameter contract changed: {counts} != {expected}")
        return {name: model.to(self.device) for name, model in models.items()}

    def _evaluate(self, model: Any) -> Dict[str, float]:
        squared = 0.0
        persistence_squared = 0.0
        active_squared = 0.0
        active_count = 0
        count = 0
        model.eval()
        with self.torch.no_grad():
            for raw in self.corpus.iter_raw(1, self.config.evaluation_sample_limit):
                features, target = self.transform.apply(raw)
                prediction = model(self.torch.as_tensor(features, device=self.device)).cpu().numpy() * self.config.voltage_scale_mv
                target_mv = target * self.config.voltage_scale_mv
                error = prediction - target_mv
                squared += float(np.sum(error.astype(np.float64) ** 2))
                persistence_squared += float(np.sum(target_mv.astype(np.float64) ** 2))
                active = np.abs(target_mv) >= self.config.active_delta_threshold_mv
                active_squared += float(np.sum(error[active].astype(np.float64) ** 2))
                active_count += int(active.sum())
                count += len(error)
        rmse = math.sqrt(squared / count)
        persistence = math.sqrt(persistence_squared / count)
        return {
            "soma_rmse_mv": rmse,
            "persistence_soma_rmse_mv": persistence,
            "improvement_vs_persistence_fraction": 1.0 - rmse / max(persistence, 1e-12),
            "active_soma_rmse_mv": math.sqrt(active_squared / active_count) if active_count else 0.0,
            "active_count": active_count,
            "example_count": count,
        }

    def train(self) -> Dict[str, Any]:
        runs = []
        for seed in self.config.seeds:
            self.torch.manual_seed(seed)
            self.torch.cuda.manual_seed_all(seed)
            models = self._models()
            optimizers = {
                name: self.torch.optim.AdamW(model.parameters(), lr=self.config.learning_rate, weight_decay=self.config.weight_decay)
                for name, model in models.items()
            }
            best = {name: (math.inf, None) for name in models}
            rng = np.random.default_rng(seed)
            started = time.perf_counter()
            for step in range(1, self.config.training_steps + 1):
                raw = self.corpus.sample_raw(0, self.config.batch_size, rng)
                features, target = self.transform.apply(raw)
                x = self.torch.as_tensor(features, device=self.device)
                y = self.torch.as_tensor(target, device=self.device)
                active = self.torch.abs(y * self.config.voltage_scale_mv) >= self.config.active_delta_threshold_mv
                weight = self.torch.where(active, self.config.active_weight, 1.0)
                losses = {}
                for name, model in models.items():
                    model.train()
                    optimizers[name].zero_grad(set_to_none=True)
                    prediction = model(x)
                    loss = self.torch.mean(weight * (prediction - y) ** 2)
                    loss.backward()
                    self.torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.gradient_clip_norm)
                    optimizers[name].step()
                    losses[name] = float(loss.detach().cpu())
                if step in self.config.checkpoints:
                    for name, model in models.items():
                        metrics = self._evaluate(model)
                        runs.append({"seed": seed, "model": name, "step": step, **metrics})
                        if metrics["soma_rmse_mv"] < best[name][0]:
                            best[name] = (metrics["soma_rmse_mv"], copy.deepcopy(model.state_dict()))
                if step == 1 or step == self.config.training_steps or step % self.config.progress_interval == 0:
                    eta = (time.perf_counter() - started) / step * (self.config.training_steps - step)
                    compact = " ".join(f"{name}={value:.4g}" for name, value in losses.items())
                    print(f"[GIADA RunPod][matched seed={seed}] {step}/{self.config.training_steps} ETA {eta/60:.1f} min {compact}", flush=True)
            for name, model in models.items():
                model.load_state_dict(best[name][1])
                self.torch.save({"seed": seed, "model": name, "state_dict": model.state_dict(), "normalization": self.transform.to_dict()}, self.output_dir / f"{name}_seed{seed}.pt")
        final = [row for row in runs if row["step"] == self.config.training_steps]
        medians = {
            name: float(np.median([row["soma_rmse_mv"] for row in final if row["model"] == name]))
            for name in ("branch_elm_core", "giada_voltage_bridge")
        }
        report = {
            "schema_version": "giada-paper-scale-matched-training-v1",
            "valid": True,
            "code_revision": self.code_revision,
            "same_numeric_input": True,
            "same_target": "authentic_NEURON_one_ms_soma_voltage_transition",
            "same_sample_order": True,
            "same_optimizer_and_loss": True,
            "rollout_claimed": False,
            "train_transition_count": self.corpus.train_count,
            "validation_transition_count": self.corpus.validation_count,
            "configuration": asdict(self.config),
            "mini_scaling_law_runs": runs,
            "final_median_soma_rmse_mv": medians,
            "giada_relative_rmse_reduction_vs_branch_elm": 1.0 - medians["giada_voltage_bridge"] / medians["branch_elm_core"],
        }
        (self.output_dir / "final_report.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
        return report
