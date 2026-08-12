"""Notebook-05c causal isolation for the HayFlow-Hines boundary bottleneck."""

from __future__ import annotations

import hashlib
import io
import json
import math
import random
import time
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from ..hayflow_data.composite_flowmap import EVENT_KINDS, CompositeFlowmapBundle
from ..hayflow_eval.flowmap_metrics import write_parquet
from .hayflow_hines import HayFlowHines, model_parameter_count
from .hines_experiment import (
    HayFlowHinesExperiment,
    HinesPrototypeExperimentConfig,
    Progress,
    _write_json,
)
from .hines_layer import require_torch


EXPECTED_05B_ARCHIVE_SHA256 = (
    "f653bf8f0048134994727593237067ab1faf91d7f42f17d70150ff8dc339bca5"
)
EXPECTED_05B_CHECKPOINT_SHA256 = (
    "5fe46579aa05143e30a252ca78995e15d4fafe0c510967ac0c59b3e0344bd06f"
)
EXPECTED_05B_MEMBER_SHA256 = {
    "model_configurations.json": "f2b4687743696b02b791cbafda2a6397f2ea01b8ea67d693f4eb90a08fb2dd78",
    "canary_overfit_report.json": "0b7a9443f448a4dde4d241e5f603f99297d0c014d64103f65a623a68be77af3e",
    "hines_layer_tests.json": "65ed328887980e125de57bb9c8f79b71e89d3408003b539d4c793f4f561a9758",
    "composite_loader_report.json": "75a30398ed71e3d670c341e6ea4b8753269d4ddd42ffb8584d1ecd36c6c89842",
    "normalization_schema.json": "70ee2c3741d9273d5ce54a722b016a4964b84e7b8757730f724d3b149b771690",
    "checkpoints/canary_models.pt": EXPECTED_05B_CHECKPOINT_SHA256,
}
BOUNDARY_MODES = ("timed_masked", "untimed_masked", "no_event_jump", "direct_residual")


@dataclass(frozen=True)
class HinesIsolationConfig:
    subset_sizes: Tuple[int, ...] = (1, 8, 32, 76)
    subset_epochs: Tuple[int, ...] = (120, 100, 60, 40)
    modes: Tuple[str, ...] = ("timed_masked", "direct_residual")
    batch_size: int = 4
    learning_rate: float = 1e-3
    gradient_clip_norm: float = 10.0
    branch_epochs: int = 160
    seed: int = 1705
    tail_segment_count: int = 16
    diagnostic_one_transition_rmse_mv: float = 0.25
    diagnostic_one_transition_peak_error_mv: float = 1.0

    def validate(self) -> None:
        if not self.subset_sizes or len(self.subset_sizes) != len(self.subset_epochs):
            raise ValueError("subset_sizes and subset_epochs must be non-empty and aligned")
        if any(
            right <= left
            for left, right in zip(self.subset_sizes, self.subset_sizes[1:])
        ):
            raise ValueError("subset_sizes must be strictly increasing")
        if min(self.subset_sizes + self.subset_epochs) <= 0:
            raise ValueError("subset sizes and epochs must be positive")
        if not self.modes or len(set(self.modes)) != len(self.modes):
            raise ValueError("isolation modes must be non-empty and unique")
        if not set(self.modes).issubset(set(BOUNDARY_MODES)):
            raise ValueError(f"unsupported isolation mode in {self.modes}")
        if min(
            self.batch_size, self.learning_rate, self.gradient_clip_norm,
            self.branch_epochs, self.tail_segment_count,
        ) <= 0:
            raise ValueError("isolation optimization values must be positive")

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "HinesIsolationConfig":
        payload = dict(values)
        for name in ("subset_sizes", "subset_epochs", "modes"):
            if name in payload:
                payload[name] = tuple(payload[name])
        result = cls(**payload)
        result.validate()
        return result


def sha256_file(path: Path, block_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            block = handle.read(block_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


class HinesCausalIsolationExperiment(HayFlowHinesExperiment):
    """Small, fail-informative experiment; it never authorizes full training."""

    def __init__(
        self,
        bundle: CompositeFlowmapBundle,
        output_dir: Path,
        model_config: HinesPrototypeExperimentConfig,
        isolation_config: HinesIsolationConfig,
        checkpoint_source: Path,
    ) -> None:
        super().__init__(bundle, output_dir, model_config)
        isolation_config.validate()
        self.isolation = isolation_config
        self.checkpoint_source = Path(checkpoint_source).resolve()
        self.checkpoint_contract: Dict[str, Any] = {}
        self.canary_indices: Optional[np.ndarray] = None
        self.branch_pair: Optional[Tuple[int, int]] = None
        self.worst_transition: Optional[int] = None
        self.worst_segment: Optional[int] = None
        self.progressive_rows: List[Dict[str, Any]] = []
        self.branch_rows: List[Dict[str, Any]] = []

    def prepare_isolation(self) -> Dict[str, Any]:
        base = self.prepare()
        checkpoint_bytes, source_contract = self._read_05b_source()
        checkpoint_hash = hashlib.sha256(checkpoint_bytes).hexdigest()
        indices, pair = self._canary_indices()
        if len(indices) != max(self.isolation.subset_sizes):
            raise RuntimeError(
                f"05c expected {max(self.isolation.subset_sizes)} deterministic "
                f"canary transitions, observed {len(indices)}"
            )
        self.canary_indices = indices
        self.branch_pair = pair
        self._checkpoint_bytes = checkpoint_bytes
        self.checkpoint_contract = {
            **source_contract,
            "checkpoint_sha256": checkpoint_hash,
            "logical_indices_sha256": hashlib.sha256(indices.tobytes()).hexdigest(),
            "transition_count": int(len(indices)),
            "branch_pair": list(pair) if pair else None,
        }
        _write_json(self.output_dir / "isolation_config.json", {
            "schema_version": "05c-isolation-config-v1",
            "model": asdict(self.config),
            "isolation": asdict(self.isolation),
            "checkpoint": self.checkpoint_contract,
            "full_training_authorized": False,
        })
        return {**base, "checkpoint": self.checkpoint_contract}

    def _read_05b_source(self) -> Tuple[bytes, Dict[str, Any]]:
        """Read either the original ZIP or Kaggle's extracted artifact tree."""

        source = self.checkpoint_source
        if source.is_file():
            archive_hash = sha256_file(source)
            if archive_hash != EXPECTED_05B_ARCHIVE_SHA256:
                raise RuntimeError(
                    "05b archive SHA-256 mismatch; refusing checkpoint diagnosis"
                )
            with zipfile.ZipFile(source) as archive:
                members: Dict[str, bytes] = {}
                resolved_names: Dict[str, str] = {}
                for suffix in EXPECTED_05B_MEMBER_SHA256:
                    matches = [
                        name for name in archive.namelist()
                        if name.replace("\\", "/").endswith(suffix)
                    ]
                    if len(matches) != 1:
                        raise RuntimeError(
                            f"expected one 05b member ending in {suffix!r}, "
                            f"found {matches}"
                        )
                    resolved_names[suffix] = matches[0]
                    members[suffix] = archive.read(matches[0])
            source_contract: Dict[str, Any] = {
                "source_kind": "original_zip",
                "source_path": str(source),
                "archive_sha256": archive_hash,
                "checkpoint_member": resolved_names["checkpoints/canary_models.pt"],
            }
        elif source.is_dir():
            members = {}
            resolved_names = {}
            for suffix in EXPECTED_05B_MEMBER_SHA256:
                parts = tuple(Path(suffix).parts)
                matches = [
                    path for path in source.rglob(parts[-1])
                    if tuple(path.parts[-len(parts):]) == parts
                ]
                if len(matches) != 1:
                    raise RuntimeError(
                        f"expected one extracted 05b member ending in {suffix!r}, "
                        f"found {[str(path) for path in matches]}"
                    )
                resolved_names[suffix] = matches[0].relative_to(source).as_posix()
                members[suffix] = matches[0].read_bytes()
            source_contract = {
                "source_kind": "kaggle_extracted_directory",
                "source_path": str(source),
                "archive_sha256": None,
                "checkpoint_member": resolved_names["checkpoints/canary_models.pt"],
            }
        else:
            raise RuntimeError(f"05b artifact source does not exist: {source}")

        observed_hashes = {
            name: hashlib.sha256(payload).hexdigest()
            for name, payload in members.items()
        }
        mismatches = {
            name: {"expected": EXPECTED_05B_MEMBER_SHA256[name], "observed": value}
            for name, value in observed_hashes.items()
            if value != EXPECTED_05B_MEMBER_SHA256[name]
        }
        if mismatches:
            raise RuntimeError(f"05b artifact member SHA-256 mismatch: {mismatches}")
        source_contract["verified_member_sha256"] = observed_hashes
        return members["checkpoints/canary_models.pt"], source_contract

    def _load_h2_checkpoint(self, device: Any) -> Tuple[Any, Dict[str, Any]]:
        import torch

        model = self._fresh_h2(device)
        payload = torch.load(
            io.BytesIO(self._checkpoint_bytes), map_location=device,
            weights_only=True,
        )
        if "H2" not in payload:
            raise RuntimeError(f"H2 missing from checkpoint keys {sorted(payload)}")
        incompatible = model.load_state_dict(payload["H2"], strict=False)
        allowed_missing = {
            "direct_boundary_residual.weight", "direct_boundary_residual.bias"
        }
        if set(incompatible.missing_keys) != allowed_missing or incompatible.unexpected_keys:
            raise RuntimeError(
                f"unexpected checkpoint incompatibility: {incompatible}"
            )
        return model, {
            "missing_new_diagnostic_parameters": sorted(incompatible.missing_keys),
            "unexpected_parameters": sorted(incompatible.unexpected_keys),
        }

    def _fresh_h2(self, device: Any) -> Any:
        return HayFlowHines(
            self.config.model, self.layout.to_model_metadata(), self.arrays
        ).to(device)

    def _forward_numpy(
        self,
        model: Any,
        indices: Sequence[int],
        device: Any,
        boundary_mode: str,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        import torch

        raw = self._batch(indices, include_targets=True)
        batch = self._torch_batch(raw, device)
        model.eval()
        with torch.no_grad():
            output = model(
                batch, ablation="H2", decode_teacher=False,
                boundary_mode=boundary_mode,
            )
        wanted = (
            "voltage", "voltage_star", "continuous_residual", "event_jump",
            "direct_boundary_residual", "event_logits", "event_timing",
            "event_segment_logits", "event_local_gate",
            "event_boundary_raw_delta_mv", "event_boundary_delta_mv",
        )
        values = {
            name: output[name].detach().cpu().numpy()
            for name in wanted if name in output
        }
        return values, raw

    def diagnose_checkpoint(self) -> Dict[str, Any]:
        require_torch()
        import torch

        if self.canary_indices is None:
            raise RuntimeError("prepare_isolation() must run first")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model, compatibility = self._load_h2_checkpoint(device)
        transition_rows: List[Dict[str, Any]] = []
        predictions: Dict[str, Dict[str, np.ndarray]] = {}
        raw_reference: Optional[Dict[str, Any]] = None
        for mode in ("timed_masked", "untimed_masked", "no_event_jump"):
            output, raw = self._forward_numpy(
                model, self.canary_indices, device, mode
            )
            predictions[mode] = output
            raw_reference = raw
            error = output["voltage"] - raw["voltage_target"]
            for row, logical_index in enumerate(self.canary_indices):
                absolute = np.abs(error[row])
                segment = int(np.argmax(absolute))
                transition_rows.append({
                    "boundary_mode": mode,
                    "logical_index": int(logical_index),
                    "rmse_mv": float(np.sqrt(np.mean(error[row] ** 2))),
                    "maximum_absolute_error_mv": float(absolute[segment]),
                    "maximum_error_segment_id": segment,
                    "teacher_boundary_peak_mv": float(raw["voltage_target"][row].max()),
                    "predicted_boundary_peak_mv": float(output["voltage"][row].max()),
                    "boundary_peak_error_mv": float(abs(
                        output["voltage"][row].max()
                        - raw["voltage_target"][row].max()
                    )),
                    "event_kinds": ",".join(
                        EVENT_KINDS[col]
                        for col in np.flatnonzero(raw["event_presence"][row] > 0.5)
                    ),
                })
        write_parquet(
            self.output_dir / "checkpoint_transition_errors.parquet",
            transition_rows,
        )
        timed = [row for row in transition_rows if row["boundary_mode"] == "timed_masked"]
        worst = max(timed, key=lambda row: row["boundary_peak_error_mv"])
        self.worst_transition = int(worst["logical_index"])
        worst_position = int(np.flatnonzero(
            self.canary_indices == self.worst_transition
        )[0])
        teacher = raw_reference["voltage_target"][worst_position]
        predicted = predictions["timed_masked"]["voltage"][worst_position]
        self.worst_segment = int(np.argmax(np.abs(predicted - teacher)))
        segment_rows = []
        for segment_id, record in enumerate(self.layout.segments):
            row = {
                "logical_index": self.worst_transition,
                "segment_id": segment_id,
                "region": str(record.get("region", "unknown")),
                "voltage_t_mv": float(raw_reference["voltage_t"][worst_position, segment_id]),
                "teacher_voltage_t_plus_1_mv": float(teacher[segment_id]),
            }
            for mode in ("timed_masked", "untimed_masked", "no_event_jump"):
                values = predictions[mode]
                row[f"{mode}_voltage_star_mv"] = float(values["voltage_star"][worst_position, segment_id])
                row[f"{mode}_continuous_residual_mv"] = float(values["continuous_residual"][worst_position, segment_id])
                row[f"{mode}_event_jump_mv"] = float(values["event_jump"][worst_position, segment_id])
                row[f"{mode}_voltage_t_plus_1_mv"] = float(values["voltage"][worst_position, segment_id])
                row[f"{mode}_absolute_error_mv"] = float(abs(
                    values["voltage"][worst_position, segment_id] - teacher[segment_id]
                ))
            segment_rows.append(row)
        write_parquet(
            self.output_dir / "worst_transition_segment_decomposition.parquet",
            segment_rows,
        )
        event_rows = []
        output = predictions["timed_masked"]
        for event_index, kind in enumerate(EVENT_KINDS):
            segment_logits = output["event_segment_logits"][worst_position, event_index]
            event_rows.append({
                "logical_index": self.worst_transition,
                "event_kind": kind,
                "teacher_present": bool(raw_reference["event_presence"][worst_position, event_index]),
                "predicted_probability": float(1.0 / (1.0 + np.exp(-output["event_logits"][worst_position, event_index]))),
                "predicted_segment_id": int(np.argmax(segment_logits)),
                "predicted_onset_peak_offset_duration": output["event_timing"][worst_position, event_index].tolist(),
                "raw_boundary_delta_mv": float(output["event_boundary_raw_delta_mv"][worst_position, event_index]),
                "timed_boundary_delta_mv": float(output["event_boundary_delta_mv"][worst_position, event_index]),
                "maximum_local_gate": float(output["event_local_gate"][worst_position, :, event_index].max()),
            })
        _write_json(self.output_dir / "worst_transition_events.json", {
            "schema_version": "05c-worst-transition-events-v1",
            "logical_index": self.worst_transition,
            "maximum_error_segment_id": self.worst_segment,
            "teacher_events": self.store.events(self.worst_transition),
            "predicted_events": event_rows,
        })
        report = {
            "schema_version": "05c-checkpoint-forensics-v1",
            "checkpoint_compatibility": compatibility,
            "worst_transition": self.worst_transition,
            "worst_segment": self.worst_segment,
            "worst_timed_masked": worst,
            "mode_summary": {
                mode: {
                    "mean_rmse_mv": float(np.mean([
                        row["rmse_mv"] for row in transition_rows
                        if row["boundary_mode"] == mode
                    ])),
                    "maximum_peak_error_mv": float(max(
                        row["boundary_peak_error_mv"] for row in transition_rows
                        if row["boundary_mode"] == mode
                    )),
                }
                for mode in ("timed_masked", "untimed_masked", "no_event_jump")
            },
        }
        _write_json(self.output_dir / "checkpoint_forensics.json", report)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return report

    def _nested_subsets(self) -> Dict[int, np.ndarray]:
        if self.canary_indices is None or self.worst_transition is None:
            raise RuntimeError("checkpoint diagnosis must run first")
        raw = self._batch(self.canary_indices, include_targets=True)
        richness = raw["event_presence"].sum(1)
        voltage_delta = np.max(
            np.abs(raw["voltage_target"] - raw["voltage_t"]), axis=1
        )
        priority = np.lexsort((self.canary_indices, -voltage_delta, -richness))
        ordered = [self.worst_transition]
        if self.branch_pair:
            ordered.extend(self.branch_pair)
        ordered.extend(int(self.canary_indices[index]) for index in priority)
        unique = np.asarray(list(dict.fromkeys(ordered)), dtype=np.int64)
        return {
            int(size): unique[: int(size)].copy()
            for size in self.isolation.subset_sizes
        }

    def _primary_loss(
        self,
        output: Mapping[str, Any],
        batch: Mapping[str, Any],
        *,
        include_events: bool,
    ) -> Tuple[Any, Dict[str, float]]:
        import torch
        import torch.nn.functional as functional

        error = (output["voltage"] - batch["voltage_target"]) / 10.0
        mean = torch.mean(error ** 2)
        peak = torch.mean(
            ((output["voltage"].amax(1) - batch["voltage_target"].amax(1)) / 10.0) ** 2
        )
        k = min(self.isolation.tail_segment_count, error.shape[1])
        tail = torch.topk(torch.abs(error), k=k, dim=1).values.square().mean()
        event = mean.new_zeros(())
        if include_events:
            event = functional.binary_cross_entropy_with_logits(
                output["event_logits"], batch["event_presence"]
            )
        total = mean + 2.0 * peak + tail + 0.1 * event
        return total, {
            "mean_voltage_mse": float(mean.detach()),
            "peak_mse": float(peak.detach()),
            "tail_mse": float(tail.detach()),
            "event_bce": float(event.detach()),
        }

    def _train_micro_overfit(
        self,
        indices: np.ndarray,
        epochs: int,
        boundary_mode: str,
        device: Any,
        seed: int,
        label: str,
        *,
        branch_weight: float = 0.0,
    ) -> Tuple[Any, Dict[str, Any]]:
        import torch

        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        model = self._fresh_h2(device)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=self.isolation.learning_rate, weight_decay=0.0
        )
        raw = self._batch(indices, include_targets=True)
        progress = Progress(label, epochs)
        history = []
        rng = np.random.default_rng(seed)
        for epoch in range(epochs):
            order = np.arange(len(indices))
            rng.shuffle(order)
            epoch_losses = []
            gradient_norms = []
            component_rows: List[Dict[str, float]] = []
            model.train()
            for start in range(0, len(order), self.isolation.batch_size):
                positions = order[start:start + self.isolation.batch_size]
                selected = {}
                for key, value in raw.items():
                    if (
                        isinstance(value, np.ndarray) and value.ndim
                        and value.shape[0] == len(indices)
                        and key != "anchor_segment_ids"
                    ):
                        selected[key] = value[positions]
                    else:
                        selected[key] = value
                batch = self._torch_batch(selected, device)
                optimizer.zero_grad(set_to_none=True)
                output = model(
                    batch, ablation="H2", decode_teacher=False,
                    boundary_mode=boundary_mode,
                )
                loss, components = self._primary_loss(
                    output, batch, include_events=boundary_mode != "direct_residual"
                )
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), self.isolation.gradient_clip_norm
                )
                optimizer.step()
                epoch_losses.append(float(loss.detach()))
                gradient_norms.append(float(norm.detach()))
                component_rows.append(components)
            if branch_weight and self.branch_pair:
                optimizer.zero_grad(set_to_none=True)
                branch = branch_weight * self._branch_loss_mode(
                    model, self.branch_pair, device, boundary_mode
                )
                branch.backward()
                norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), self.isolation.gradient_clip_norm
                )
                optimizer.step()
                gradient_norms.append(float(norm.detach()))
                branch_value = float(branch.detach())
            else:
                branch_value = 0.0
            if epoch == 0 or (epoch + 1) % max(1, epochs // 10) == 0 or epoch + 1 == epochs:
                metrics = self._micro_metrics(model, indices, device, boundary_mode)
                progress.update(
                    epoch + 1,
                    f"mode={boundary_mode} V={metrics['voltage_rmse_mv']:.3g} "
                    f"peak={metrics['maximum_peak_error_mv']:.3g}",
                )
                history.append({
                    "epoch": epoch,
                    "loss": float(np.mean(epoch_losses)),
                    "gradient_norm_pre_clip": float(np.mean(gradient_norms)),
                    "branching_loss_weighted": branch_value,
                    **{
                        key: float(np.mean([row[key] for row in component_rows]))
                        for key in component_rows[0]
                    },
                    **metrics,
                })
        final = self._micro_metrics(model, indices, device, boundary_mode)
        final["history"] = history
        return model, final

    def _micro_metrics(
        self, model: Any, indices: Sequence[int], device: Any, boundary_mode: str
    ) -> Dict[str, float]:
        output, raw = self._forward_numpy(model, indices, device, boundary_mode)
        error = output["voltage"] - raw["voltage_target"]
        peak_error = np.abs(
            output["voltage"].max(1) - raw["voltage_target"].max(1)
        )
        segment_error = np.abs(error)
        return {
            "voltage_rmse_mv": float(np.sqrt(np.mean(error ** 2))),
            "maximum_peak_error_mv": float(peak_error.max()),
            "maximum_segment_error_mv": float(segment_error.max()),
        }

    def _branch_loss_mode(
        self, model: Any, pair: Sequence[int], device: Any, boundary_mode: str
    ) -> Any:
        import torch

        raw = self._batch(pair, include_targets=True)
        batch = self._torch_batch(raw, device)
        output = model(
            batch, ablation="H2", decode_teacher=False,
            boundary_mode=boundary_mode,
        )
        predicted = torch.sqrt(torch.mean(
            (output["voltage"][0] - output["voltage"][1]) ** 2
        ) + 1e-12)
        teacher = torch.sqrt(torch.mean(
            (batch["voltage_target"][0] - batch["voltage_target"][1]) ** 2
        ) + 1e-12)
        return torch.abs(predicted - teacher) / teacher.detach().clamp_min(1e-3)

    def _branch_metrics(
        self, model: Any, device: Any, boundary_mode: str
    ) -> Dict[str, float]:
        if not self.branch_pair:
            return {"teacher_distance_mv": math.nan, "predicted_distance_mv": math.nan, "retention": math.nan}
        output, raw = self._forward_numpy(
            model, self.branch_pair, device, boundary_mode
        )
        teacher = float(np.sqrt(np.mean(
            (raw["voltage_target"][0] - raw["voltage_target"][1]) ** 2
        )))
        predicted = float(np.sqrt(np.mean(
            (output["voltage"][0] - output["voltage"][1]) ** 2
        )))
        return {
            "teacher_distance_mv": teacher,
            "predicted_distance_mv": predicted,
            "retention": predicted / max(teacher, 1e-8),
        }

    def run_progressive_isolation(self) -> Dict[str, Any]:
        require_torch()
        import torch

        subsets = self._nested_subsets()
        _write_json(self.output_dir / "nested_subsets.json", {
            "schema_version": "05c-nested-subsets-v1",
            "worst_transition": self.worst_transition,
            "subsets": {str(size): values.tolist() for size, values in subsets.items()},
        })
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        total = len(self.isolation.modes) * len(self.isolation.subset_sizes)
        tracker = Progress("progressive isolation matrix", total)
        run = 0
        checkpoint_root = self.output_dir / "checkpoints"
        checkpoint_root.mkdir(exist_ok=True)
        for mode_index, mode in enumerate(self.isolation.modes):
            for size_index, (size, epochs) in enumerate(zip(
                self.isolation.subset_sizes, self.isolation.subset_epochs
            )):
                seed = self.isolation.seed + 100 * mode_index + size_index
                model, metrics = self._train_micro_overfit(
                    subsets[int(size)], int(epochs), mode, device, seed,
                    f"micro-overfit n={size}",
                )
                branch = self._branch_metrics(model, device, mode)
                row = {
                    "boundary_mode": mode,
                    "subset_size": int(size),
                    "epochs": int(epochs),
                    "seed": int(seed),
                    "parameter_count": model_parameter_count(model),
                    **{key: value for key, value in metrics.items() if key != "history"},
                    "branching_retention_without_branch_loss": branch["retention"],
                }
                self.progressive_rows.append(row)
                _write_json(
                    self.output_dir / f"history_{mode}_n{size}.json",
                    {"run": row, "history": metrics["history"]},
                )
                torch.save(
                    {"model": model.state_dict(), "run": row},
                    checkpoint_root / f"{mode}_n{size}.pt",
                )
                run += 1
                tracker.update(run, f"mode={mode} n={size}")
                del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        write_parquet(
            self.output_dir / "progressive_overfit_metrics.parquet",
            self.progressive_rows,
        )
        one = {
            row["boundary_mode"]: row
            for row in self.progressive_rows if row["subset_size"] == 1
        }
        report = {
            "schema_version": "05c-progressive-isolation-v1",
            "valid": len(self.progressive_rows) == total,
            "one_transition": one,
            "direct_one_transition_passed": bool(
                one.get("direct_residual", {}).get("voltage_rmse_mv", math.inf)
                < self.isolation.diagnostic_one_transition_rmse_mv
                and one.get("direct_residual", {}).get("maximum_peak_error_mv", math.inf)
                < self.isolation.diagnostic_one_transition_peak_error_mv
            ),
            "timed_one_transition_passed": bool(
                one.get("timed_masked", {}).get("voltage_rmse_mv", math.inf)
                < self.isolation.diagnostic_one_transition_rmse_mv
                and one.get("timed_masked", {}).get("maximum_peak_error_mv", math.inf)
                < self.isolation.diagnostic_one_transition_peak_error_mv
            ),
        }
        _write_json(self.output_dir / "progressive_isolation_report.json", report)
        return report

    def run_branch_isolation(self) -> Dict[str, Any]:
        require_torch()
        import torch

        if not self.branch_pair:
            raise RuntimeError("no train counterfactual pair available")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        indices = np.asarray(self.branch_pair, dtype=np.int64)
        checkpoint_root = self.output_dir / "checkpoints"
        for mode_index, mode in enumerate(self.isolation.modes):
            seed = self.isolation.seed + 900 + mode_index
            model, metrics = self._train_micro_overfit(
                indices, self.isolation.branch_epochs, mode, device, seed,
                "branch-pair isolation", branch_weight=1.0,
            )
            branch = self._branch_metrics(model, device, mode)
            row = {
                "boundary_mode": mode,
                "epochs": self.isolation.branch_epochs,
                "seed": seed,
                **{key: value for key, value in metrics.items() if key != "history"},
                **branch,
            }
            self.branch_rows.append(row)
            _write_json(
                self.output_dir / f"branch_history_{mode}.json",
                {"run": row, "history": metrics["history"]},
            )
            torch.save(
                {"model": model.state_dict(), "run": row},
                checkpoint_root / f"branch_{mode}.pt",
            )
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        write_parquet(
            self.output_dir / "branch_isolation_metrics.parquet", self.branch_rows
        )
        report = {
            "schema_version": "05c-branch-isolation-v1",
            "pair": list(self.branch_pair),
            "rows": self.branch_rows,
        }
        _write_json(self.output_dir / "branch_isolation_report.json", report)
        return report

    def finalize_isolation(
        self,
        checkpoint_report: Mapping[str, Any],
        progressive_report: Mapping[str, Any],
        branch_report: Mapping[str, Any],
    ) -> Dict[str, Any]:
        direct_pass = bool(progressive_report["direct_one_transition_passed"])
        timed_pass = bool(progressive_report["timed_one_transition_passed"])
        if direct_pass and not timed_pass:
            diagnosis = "EVENT_BOUNDARY_BOTTLENECK"
        elif not direct_pass and not timed_pass:
            diagnosis = "ENCODER_OR_OPTIMIZATION_BOTTLENECK"
        elif direct_pass and timed_pass:
            diagnosis = "SINGLE_TRANSITION_SOLVED_CHECK_SCALING"
        else:
            diagnosis = "EVENT_PATH_ADVANTAGE_UNEXPECTED_DIRECT_FAILURE"
        report = {
            "schema_version": "05c-final-report-v1",
            "valid": bool(progressive_report.get("valid")),
            "decision": "DIAGNOSTIC_ONLY_NO_FULL_TRAINING",
            "full_training_authorized": False,
            "diagnosis": diagnosis,
            "dataset_fingerprint": self.bundle.fingerprint,
            "checkpoint": self.checkpoint_contract,
            "worst_transition": checkpoint_report["worst_transition"],
            "worst_segment": checkpoint_report["worst_segment"],
            "checkpoint_mode_summary": checkpoint_report["mode_summary"],
            "progressive": progressive_report,
            "branching": branch_report,
            "next_decision": (
                "Choose the 05d architecture change only from the registered "
                "05c diagnosis; do not start the full curriculum."
            ),
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
        _write_json(self.output_dir / "artifact_index.json", {
            "schema_version": "05c-artifact-index-v1",
            "artifact_count": len(records),
            "artifacts": records,
        })
        return report
