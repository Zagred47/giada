"""Notebook-05d conditioning controls for the HayFlow-Hines boundary path."""

from __future__ import annotations

import hashlib
import json
import math
import random
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from ..hayflow_data.composite_flowmap import CompositeFlowmapBundle
from ..hayflow_eval.flowmap_metrics import write_parquet
from .hines_experiment import HayFlowHinesExperiment, Progress, _write_json
from .hines_isolation_experiment import (
    HinesCausalIsolationExperiment,
    HinesIsolationConfig,
    sha256_file,
)
from .hines_layer import require_torch

try:  # Keep data-only imports usable without a local PyTorch installation.
    import torch
    from torch import nn
except ImportError:  # pragma: no cover - exercised by CPU-light environments.
    torch = None
    nn = None


EXPECTED_05C_ARCHIVE_SHA256 = (
    "b6a2e222529fd293fd75602bdac3b0feca8729832f371fadd24c9aa3b96b0d70"
)
EXPECTED_05C_MEMBER_SHA256 = {
    "artifact_index.json": "cbde016c9a79b583caa8adf80fee3e2432c93f017239f38f5dc5b01e2decc579",
    "final_report.json": "f5287ee6f099350af4c79bb45234783d0348a81fd3b7b3e001014a027494a48b",
    "checkpoint_forensics.json": "d8cbdd8f5041dc329ba0e832c6b0dc16e281128aca34d8113cc60a862b429c65",
    "progressive_isolation_report.json": "63305dd6f68c42b2756f6117ede30d02476065ea72ba03cd227f2e992cbe76a4",
    "branch_isolation_report.json": "8d97bf31d11b10101afc9c7175ff674fffe0cd5075ce636a561c8e3d79ffefb5",
}
DECODER_PARAMETERIZATIONS = ("linear", "scaled_linear", "tanh")
UNFREEZING_STAGES = ("head_only", "local_features", "base_dynamics")


@dataclass(frozen=True)
class HinesConditioningConfig:
    free_residual_epochs: int = 5
    free_residual_learning_rate: float = 1.0
    decoder_epochs: int = 300
    decoder_parameterizations: Tuple[str, ...] = DECODER_PARAMETERIZATIONS
    decoder_learning_rates: Tuple[float, ...] = (1e-2, 1e-3, 1e-4)
    scaled_linear_factor_mv: float = 10.0
    tanh_limit_mv: float = 120.0
    unfreezing_stages: Tuple[str, ...] = UNFREEZING_STAGES
    one_transition_epochs: int = 400
    branch_pair_epochs: int = 600
    model_learning_rate: float = 1e-5
    branch_loss_weight: float = 1.0
    feature_epsilon: float = 1e-5
    evaluation_interval: int = 20
    early_stop_confirmations: int = 3
    seed: int = 2705
    one_transition_rmse_mv: float = 0.25
    one_transition_max_error_mv: float = 1.0
    pair_rmse_mv: float = 1.0
    pair_max_error_mv: float = 5.0
    pair_retention_minimum: float = 0.90
    pair_retention_maximum: float = 1.10
    free_rmse_mv: float = 0.05
    free_max_error_mv: float = 0.20

    def validate(self) -> None:
        positive = (
            self.free_residual_epochs,
            self.free_residual_learning_rate,
            self.decoder_epochs,
            self.scaled_linear_factor_mv,
            self.tanh_limit_mv,
            self.one_transition_epochs,
            self.branch_pair_epochs,
            self.model_learning_rate,
            self.feature_epsilon,
            self.evaluation_interval,
            self.early_stop_confirmations,
        )
        if min(positive) <= 0:
            raise ValueError("conditioning optimization values must be positive")
        if (
            not self.decoder_parameterizations
            or len(set(self.decoder_parameterizations))
            != len(self.decoder_parameterizations)
            or not set(self.decoder_parameterizations).issubset(
                set(DECODER_PARAMETERIZATIONS)
            )
        ):
            raise ValueError("invalid or repeated decoder parameterization")
        if (
            not self.decoder_learning_rates
            or min(self.decoder_learning_rates) <= 0
        ):
            raise ValueError("decoder learning rates must be positive")
        if tuple(self.unfreezing_stages) != UNFREEZING_STAGES:
            raise ValueError(
                f"unfreezing stages must preserve the preregistered order "
                f"{UNFREEZING_STAGES}"
            )
        if not 0 < self.pair_retention_minimum < self.pair_retention_maximum:
            raise ValueError("invalid pair-retention interval")

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "HinesConditioningConfig":
        payload = dict(values)
        for name in (
            "decoder_parameterizations", "decoder_learning_rates",
            "unfreezing_stages",
        ):
            if name in payload:
                payload[name] = tuple(payload[name])
        result = cls(**payload)
        result.validate()
        return result


if nn is not None:

    class ZeroInitializedBoundaryDecoder(nn.Module):
        """Shared per-segment decoder whose initial residual is exactly zero."""

        def __init__(
            self,
            hidden_width: int,
            parameterization: str,
            *,
            scaled_linear_factor_mv: float,
            tanh_limit_mv: float,
        ) -> None:
            super().__init__()
            if parameterization not in DECODER_PARAMETERIZATIONS:
                raise ValueError(f"unknown parameterization {parameterization!r}")
            self.parameterization = parameterization
            self.scaled_linear_factor_mv = float(scaled_linear_factor_mv)
            self.tanh_limit_mv = float(tanh_limit_mv)
            self.linear = nn.Linear(int(hidden_width), 1)
            nn.init.zeros_(self.linear.weight)
            nn.init.zeros_(self.linear.bias)

        def forward(self, features: Any) -> Tuple[Any, Any]:
            raw = self.linear(features).squeeze(-1)
            if self.parameterization == "linear":
                residual = raw
            elif self.parameterization == "scaled_linear":
                residual = self.scaled_linear_factor_mv * raw
            else:
                residual = self.tanh_limit_mv * torch.tanh(raw)
            return residual, raw

else:

    class ZeroInitializedBoundaryDecoder:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            require_torch()


class HinesResidualConditioningExperiment(HinesCausalIsolationExperiment):
    """Decision-grade conditioning ladder that never runs full training."""

    _STAGE_PREFIXES = {
        "head_only": (),
        "local_features": (
            "teacher_encoder", "region_embedding", "synaptic_encoder",
            "local_input", "local_blocks", "global_encoder",
        ),
        "base_dynamics": (
            "teacher_encoder", "region_embedding", "synaptic_encoder",
            "local_input", "local_blocks", "global_encoder",
            "effective_conductance", "source_current", "continuous_residual",
        ),
    }

    def __init__(
        self,
        bundle: CompositeFlowmapBundle,
        output_dir: Path,
        model_config: Any,
        isolation_config: HinesIsolationConfig,
        conditioning_config: HinesConditioningConfig,
        checkpoint_05b_source: Path,
        artifact_05c_source: Path,
        code_revision: Optional[str] = None,
    ) -> None:
        super().__init__(
            bundle, output_dir, model_config, isolation_config,
            checkpoint_05b_source,
        )
        conditioning_config.validate()
        self.conditioning = conditioning_config
        self.artifact_05c_source = Path(artifact_05c_source).resolve()
        self.code_revision = str(code_revision) if code_revision else None
        self.artifact_05c_contract: Dict[str, Any] = {}
        self.artifact_05c_report: Dict[str, Any] = {}
        self.free_rows: List[Dict[str, Any]] = []
        self.sweep_rows: List[Dict[str, Any]] = []
        self.ladder_rows: List[Dict[str, Any]] = []

    def prepare_conditioning(self) -> Dict[str, Any]:
        base = HayFlowHinesExperiment.prepare(self)
        checkpoint_bytes, checkpoint_source = self._read_05b_source()
        checkpoint_hash = hashlib.sha256(checkpoint_bytes).hexdigest()
        indices, pair = self._canary_indices()
        if len(indices) != 76 or pair != (9122, 5282):
            raise RuntimeError(
                f"unexpected 05b canary contract: count={len(indices)}, pair={pair}"
            )
        logical_hash = hashlib.sha256(indices.tobytes()).hexdigest()
        if logical_hash != "e79dad4ae40ab1a130026d0c45e7910073275475e4bd50c75c493a730080d4c9":
            raise RuntimeError("05b logical canary indices changed")
        report, artifact_contract = self._read_05c_source()
        if report.get("diagnosis") != "ENCODER_OR_OPTIMIZATION_BOTTLENECK":
            raise RuntimeError(f"unexpected 05c diagnosis: {report.get('diagnosis')}")
        if report.get("full_training_authorized") is not False:
            raise RuntimeError("05c provenance unexpectedly authorizes full training")
        if report.get("dataset_fingerprint") != self.bundle.fingerprint:
            raise RuntimeError("05c and mounted composite fingerprints disagree")
        self._checkpoint_bytes = checkpoint_bytes
        self.canary_indices = indices
        self.branch_pair = pair
        self.worst_transition = int(report["worst_transition"])
        self.worst_segment = int(report["worst_segment"])
        self.checkpoint_contract = {
            **checkpoint_source,
            "checkpoint_sha256": checkpoint_hash,
            "logical_indices_sha256": logical_hash,
            "transition_count": int(len(indices)),
            "branch_pair": list(pair),
        }
        self.artifact_05c_contract = artifact_contract
        self.artifact_05c_report = report
        payload = {
            "schema_version": "05d-conditioning-config-v1",
            "model": asdict(self.config),
            "conditioning": asdict(self.conditioning),
            "checkpoint_05b": self.checkpoint_contract,
            "artifact_05c": artifact_contract,
            "worst_transition": self.worst_transition,
            "worst_segment": self.worst_segment,
            "code_revision": self.code_revision,
            "full_training_authorized": False,
        }
        _write_json(self.output_dir / "conditioning_config.json", payload)
        return {**base, **payload}

    def _read_05c_source(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        source = self.artifact_05c_source
        if source.is_file():
            archive_hash = sha256_file(source)
            if archive_hash != EXPECTED_05C_ARCHIVE_SHA256:
                raise RuntimeError("05c archive SHA-256 mismatch")
            with zipfile.ZipFile(source) as archive:
                members: Dict[str, bytes] = {}
                resolved: Dict[str, str] = {}
                for suffix in EXPECTED_05C_MEMBER_SHA256:
                    matches = [
                        name for name in archive.namelist()
                        if name.replace("\\", "/").endswith(suffix)
                    ]
                    if len(matches) != 1:
                        raise RuntimeError(
                            f"expected one 05c member ending in {suffix!r}, "
                            f"found {matches}"
                        )
                    resolved[suffix] = matches[0]
                    members[suffix] = archive.read(matches[0])
            contract: Dict[str, Any] = {
                "source_kind": "original_zip",
                "source_path": str(source),
                "archive_sha256": archive_hash,
                "final_report_member": resolved["final_report.json"],
            }
        elif source.is_dir():
            members = {}
            resolved = {}
            for suffix in EXPECTED_05C_MEMBER_SHA256:
                parts = tuple(Path(suffix).parts)
                matches = [
                    path for path in source.rglob(parts[-1])
                    if tuple(path.parts[-len(parts):]) == parts
                ]
                if len(matches) != 1:
                    raise RuntimeError(
                        f"expected one extracted 05c member ending in {suffix!r}, "
                        f"found {[str(path) for path in matches]}"
                    )
                resolved[suffix] = matches[0].relative_to(source).as_posix()
                members[suffix] = matches[0].read_bytes()
            contract = {
                "source_kind": "kaggle_extracted_directory",
                "source_path": str(source),
                "archive_sha256": None,
                "final_report_member": resolved["final_report.json"],
            }
        else:
            raise RuntimeError(f"05c artifact source does not exist: {source}")
        observed = {
            name: hashlib.sha256(payload).hexdigest()
            for name, payload in members.items()
        }
        mismatches = {
            name: {"expected": EXPECTED_05C_MEMBER_SHA256[name], "observed": value}
            for name, value in observed.items()
            if value != EXPECTED_05C_MEMBER_SHA256[name]
        }
        if mismatches:
            raise RuntimeError(f"05c member SHA-256 mismatch: {mismatches}")
        contract["verified_member_sha256"] = observed
        return json.loads(members["final_report.json"]), contract

    @staticmethod
    def _set_seed(seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    @staticmethod
    def _voltage_metrics(predicted: Any, target: Any) -> Dict[str, float]:
        error = predicted - target
        result = {
            "voltage_rmse_mv": float(torch.sqrt(torch.mean(error.square())).detach()),
            "maximum_segment_error_mv": float(torch.max(torch.abs(error)).detach()),
            "maximum_peak_error_mv": float(torch.max(torch.abs(
                predicted.amax(1) - target.amax(1)
            )).detach()),
        }
        if len(predicted) == 2:
            teacher_distance = torch.sqrt(torch.mean(
                (target[0] - target[1]).square()
            ) + 1e-12)
            predicted_distance = torch.sqrt(torch.mean(
                (predicted[0] - predicted[1]).square()
            ) + 1e-12)
            result.update(
                teacher_distance_mv=float(teacher_distance.detach()),
                predicted_distance_mv=float(predicted_distance.detach()),
                branching_retention=float(
                    (predicted_distance / teacher_distance.clamp_min(1e-8)).detach()
                ),
            )
        return result

    def _fixed_base_and_features(
        self, indices: Sequence[int], device: Any
    ) -> Tuple[Any, Any, Any, Dict[str, Any]]:
        model, compatibility = self._load_h2_checkpoint(device)
        model.eval()
        raw = self._batch(indices, include_targets=True)
        batch = self._torch_batch(raw, device)
        with torch.no_grad():
            output = model(
                batch, ablation="H2", decode_teacher=False,
                boundary_mode="no_event_jump",
            )
        base = output["voltage"].detach()
        features = output["boundary_features"].detach()
        target = batch["voltage_target"].detach()
        del model
        return base, features, target, compatibility

    def _passes_free(self, metrics: Mapping[str, float]) -> bool:
        return bool(
            metrics["voltage_rmse_mv"] < self.conditioning.free_rmse_mv
            and metrics["maximum_segment_error_mv"]
            < self.conditioning.free_max_error_mv
        )

    def _passes_one(self, metrics: Mapping[str, float]) -> bool:
        return bool(
            metrics["voltage_rmse_mv"]
            < self.conditioning.one_transition_rmse_mv
            and metrics["maximum_segment_error_mv"]
            < self.conditioning.one_transition_max_error_mv
        )

    def _passes_pair(self, metrics: Mapping[str, float]) -> bool:
        return bool(
            metrics["voltage_rmse_mv"] < self.conditioning.pair_rmse_mv
            and metrics["maximum_segment_error_mv"]
            < self.conditioning.pair_max_error_mv
            and self.conditioning.pair_retention_minimum
            <= metrics.get("branching_retention", math.nan)
            <= self.conditioning.pair_retention_maximum
        )

    def run_free_residual_controls(self) -> Dict[str, Any]:
        require_torch()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint_root = self.output_dir / "checkpoints"
        checkpoint_root.mkdir(exist_ok=True)
        runs: Dict[str, Any] = {}
        for run_index, (name, indices) in enumerate((
            ("one_transition", [self.worst_transition]),
            ("branch_pair", list(self.branch_pair or ())),
        )):
            if not indices:
                raise RuntimeError(f"missing indices for {name}")
            self._set_seed(self.conditioning.seed + run_index)
            base, _, target, compatibility = self._fixed_base_and_features(
                indices, device
            )
            analytic = self._voltage_metrics(
                base + (target - base), target
            )
            residual = nn.Parameter(torch.zeros_like(target))
            # For L = 0.5 * sum((r - target)^2), SGD with lr=1 has the
            # exact one-step solution r <- target. This is deliberate: the
            # oracle checks plumbing, not Adam's scale-dependent convergence.
            optimizer = torch.optim.SGD(
                [residual], lr=self.conditioning.free_residual_learning_rate
            )
            progress = Progress(
                f"free residual {name}", self.conditioning.free_residual_epochs
            )
            history = []
            for epoch in range(self.conditioning.free_residual_epochs):
                optimizer.zero_grad(set_to_none=True)
                predicted = base + residual
                loss = 0.5 * torch.sum((predicted - target).square())
                loss.backward()
                gradient_norm = float(residual.grad.norm().detach())
                optimizer.step()
                if (
                    epoch == 0
                    or (epoch + 1) % self.conditioning.evaluation_interval == 0
                    or epoch + 1 == self.conditioning.free_residual_epochs
                ):
                    metrics = self._voltage_metrics(base + residual, target)
                    row = {
                        "run": name, "epoch": epoch + 1,
                        "loss": float(loss.detach()),
                        "gradient_norm": gradient_norm, **metrics,
                    }
                    history.append(row)
                    progress.update(
                        epoch + 1,
                        f"V={metrics['voltage_rmse_mv']:.4g} "
                        f"max={metrics['maximum_segment_error_mv']:.4g}",
                    )
            final = self._voltage_metrics(base + residual, target)
            final.update(
                run=name,
                transition_indices=[int(value) for value in indices],
                passed=self._passes_free(final),
                analytic_oracle=analytic,
                checkpoint_compatibility=compatibility,
            )
            runs[name] = final
            self.free_rows.extend(history)
            _write_json(
                self.output_dir / f"free_residual_{name}.json",
                {"run": final, "history": history},
            )
            torch.save(
                {"residual": residual.detach().cpu(), "run": final},
                checkpoint_root / f"free_residual_{name}.pt",
            )
        write_parquet(
            self.output_dir / "free_residual_history.parquet", self.free_rows
        )
        report = {
            "schema_version": "05d-free-residual-v1",
            "valid": True,
            "optimizer_contract": {
                "optimizer": "SGD",
                "objective": "0.5 * sum squared boundary error",
                "learning_rate": self.conditioning.free_residual_learning_rate,
                "expected_exact_solution_steps": 1,
            },
            "runs": runs,
            "passed": all(run["passed"] for run in runs.values()),
        }
        _write_json(self.output_dir / "free_residual_report.json", report)
        return report

    def _standardize_features(self, features: Any) -> Tuple[Any, Any, Any]:
        mean = features.mean(dim=(0, 1), keepdim=True)
        std = features.std(dim=(0, 1), keepdim=True, unbiased=False)
        std = std.clamp_min(self.conditioning.feature_epsilon)
        return (features - mean) / std, mean, std

    def _new_decoder(self, parameterization: str, device: Any) -> Any:
        return ZeroInitializedBoundaryDecoder(
            self.config.model.hidden_width,
            parameterization,
            scaled_linear_factor_mv=self.conditioning.scaled_linear_factor_mv,
            tanh_limit_mv=self.conditioning.tanh_limit_mv,
        ).to(device)

    @staticmethod
    def _gradient_norm(parameters: Sequence[Any]) -> float:
        squares = []
        for parameter in parameters:
            if parameter.grad is not None:
                squares.append(parameter.grad.detach().square().sum())
        if not squares:
            return 0.0
        return float(torch.sqrt(torch.stack(squares).sum()).detach())

    def _module_gradient_norms(
        self, model: Any, decoder: Any, stage: str
    ) -> Dict[str, float]:
        result = {
            "gradient_norm_decoder": self._gradient_norm(
                list(decoder.parameters())
            )
        }
        prefixes = self._STAGE_PREFIXES[stage]
        model_parameters = [
            parameter for parameter in model.parameters()
            if parameter.requires_grad
        ]
        result["gradient_norm_model_total"] = self._gradient_norm(
            model_parameters
        )
        named = list(model.named_parameters())
        for prefix in prefixes:
            result[f"gradient_norm_{prefix}"] = self._gradient_norm([
                parameter for name, parameter in named
                if parameter.requires_grad
                and (name == prefix or name.startswith(prefix + "."))
            ])
        return result

    @staticmethod
    def _saturation_fraction(raw: Any, parameterization: str) -> float:
        if parameterization != "tanh":
            return 0.0
        return float((torch.abs(raw) >= 2.0).float().mean().detach())

    def run_frozen_decoder_sweep(self) -> Dict[str, Any]:
        require_torch()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        indices = [self.worst_transition]
        base, features, target, _ = self._fixed_base_and_features(indices, device)
        standardized, mean, std = self._standardize_features(features)
        total = (
            len(self.conditioning.decoder_parameterizations)
            * len(self.conditioning.decoder_learning_rates)
        )
        progress = Progress("frozen decoder sweep", total)
        checkpoint_root = self.output_dir / "checkpoints"
        completed = 0
        for mode_index, parameterization in enumerate(
            self.conditioning.decoder_parameterizations
        ):
            for lr_index, learning_rate in enumerate(
                self.conditioning.decoder_learning_rates
            ):
                seed = self.conditioning.seed + 100 + 10 * mode_index + lr_index
                self._set_seed(seed)
                decoder = self._new_decoder(parameterization, device)
                optimizer = torch.optim.Adam(decoder.parameters(), lr=learning_rate)
                history = []
                nonfinite = False
                for epoch in range(self.conditioning.decoder_epochs):
                    optimizer.zero_grad(set_to_none=True)
                    residual, raw = decoder(standardized)
                    predicted = base + residual
                    loss = torch.mean((predicted - target).square())
                    if not bool(torch.isfinite(loss)):
                        nonfinite = True
                        break
                    loss.backward()
                    gradient_norm = self._gradient_norm(list(decoder.parameters()))
                    if not math.isfinite(gradient_norm):
                        nonfinite = True
                        break
                    optimizer.step()
                    if (
                        epoch == 0
                        or (epoch + 1) % self.conditioning.evaluation_interval == 0
                        or epoch + 1 == self.conditioning.decoder_epochs
                    ):
                        with torch.no_grad():
                            residual_eval, raw_eval = decoder(standardized)
                            metrics = self._voltage_metrics(
                                base + residual_eval, target
                            )
                        history.append({
                            "epoch": epoch + 1,
                            "loss": float(loss.detach()),
                            "gradient_norm": gradient_norm,
                            "saturation_fraction": self._saturation_fraction(
                                raw_eval, parameterization
                            ),
                            **metrics,
                        })
                with torch.no_grad():
                    residual, raw = decoder(standardized)
                    metrics = self._voltage_metrics(base + residual, target)
                row = {
                    "parameterization": parameterization,
                    "learning_rate": float(learning_rate),
                    "epochs": self.conditioning.decoder_epochs,
                    "seed": seed,
                    "nonfinite": nonfinite,
                    "gradient_clipping_applied": False,
                    "passed": bool(not nonfinite and self._passes_one(metrics)),
                    "saturation_fraction": self._saturation_fraction(
                        raw, parameterization
                    ),
                    **metrics,
                }
                self.sweep_rows.append(row)
                label = f"{parameterization}_lr{learning_rate:g}".replace(".", "p")
                _write_json(
                    self.output_dir / f"decoder_sweep_{label}.json",
                    {"run": row, "history": history},
                )
                torch.save(
                    {
                        "decoder": decoder.state_dict(), "run": row,
                        "feature_mean": mean.cpu(), "feature_std": std.cpu(),
                    },
                    checkpoint_root / f"decoder_sweep_{label}.pt",
                )
                completed += 1
                progress.update(
                    completed,
                    f"{parameterization} lr={learning_rate:g} "
                    f"V={metrics['voltage_rmse_mv']:.3g}",
                )
        write_parquet(
            self.output_dir / "frozen_decoder_sweep.parquet", self.sweep_rows
        )
        finite_rows = [
            row for row in self.sweep_rows
            if not row["nonfinite"]
            and math.isfinite(row["voltage_rmse_mv"])
            and math.isfinite(row["maximum_segment_error_mv"])
        ]
        best = min(
            finite_rows,
            key=lambda row: (
                row["voltage_rmse_mv"], row["maximum_segment_error_mv"],
                row["parameterization"], row["learning_rate"],
            ),
        ) if finite_rows else None
        report = {
            "schema_version": "05d-frozen-decoder-sweep-v1",
            "valid": len(self.sweep_rows) == total,
            "feature_standardization": {
                "minimum_std_before_floor": float(features.std(
                    dim=(0, 1), unbiased=False
                ).min().detach()),
                "maximum_std_before_floor": float(features.std(
                    dim=(0, 1), unbiased=False
                ).max().detach()),
                "epsilon": self.conditioning.feature_epsilon,
            },
            "best": best,
            "all_runs_nonfinite": not finite_rows,
            "any_passed": any(row["passed"] for row in self.sweep_rows),
            "rows": self.sweep_rows,
        }
        _write_json(self.output_dir / "frozen_decoder_sweep_report.json", report)
        return report

    def _configure_model_stage(self, model: Any, stage: str) -> List[Any]:
        prefixes = self._STAGE_PREFIXES[stage]
        trainable = []
        for name, parameter in model.named_parameters():
            enabled = any(
                name == prefix or name.startswith(prefix + ".")
                for prefix in prefixes
            )
            parameter.requires_grad_(enabled)
            if enabled:
                trainable.append(parameter)
        return trainable

    def _train_ladder_stage(
        self,
        *,
        sample_name: str,
        indices: Sequence[int],
        stage: str,
        parameterization: str,
        decoder_learning_rate: float,
        epochs: int,
        seed: int,
        device: Any,
    ) -> Dict[str, Any]:
        self._set_seed(seed)
        model, compatibility = self._load_h2_checkpoint(device)
        decoder = self._new_decoder(parameterization, device)
        model_parameters = self._configure_model_stage(model, stage)
        raw_batch = self._batch(indices, include_targets=True)
        batch = self._torch_batch(raw_batch, device)
        model.eval()
        with torch.no_grad():
            initial = model(
                batch, ablation="H2", decode_teacher=False,
                boundary_mode="no_event_jump",
            )
            feature_mean = initial["boundary_features"].mean(
                dim=(0, 1), keepdim=True
            )
            feature_std = initial["boundary_features"].std(
                dim=(0, 1), keepdim=True, unbiased=False
            ).clamp_min(self.conditioning.feature_epsilon)
        groups = [{
            "params": list(decoder.parameters()),
            "lr": float(decoder_learning_rate),
        }]
        if model_parameters:
            groups.append({
                "params": model_parameters,
                "lr": self.conditioning.model_learning_rate,
            })
        optimizer = torch.optim.Adam(groups)
        target = batch["voltage_target"]
        progress = Progress(f"conditioning {sample_name} {stage}", epochs)
        history = []
        confirmations = 0
        nonfinite = False
        for epoch in range(epochs):
            optimizer.zero_grad(set_to_none=True)
            output = model(
                batch, ablation="H2", decode_teacher=False,
                boundary_mode="no_event_jump",
            )
            standardized = (
                output["boundary_features"] - feature_mean
            ) / feature_std
            residual, raw = decoder(standardized)
            predicted = output["voltage"] + residual
            voltage_loss = torch.mean((predicted - target).square())
            branch_loss = voltage_loss.new_zeros(())
            if len(indices) == 2:
                predicted_distance = torch.sqrt(torch.mean(
                    (predicted[0] - predicted[1]).square()
                ) + 1e-12)
                teacher_distance = torch.sqrt(torch.mean(
                    (target[0] - target[1]).square()
                ) + 1e-12)
                branch_loss = torch.abs(
                    predicted_distance - teacher_distance
                ) / teacher_distance.detach().clamp_min(1e-8)
            loss = voltage_loss + self.conditioning.branch_loss_weight * branch_loss
            if not bool(torch.isfinite(loss)):
                nonfinite = True
                break
            loss.backward()
            all_parameters = list(decoder.parameters()) + model_parameters
            gradient_norm = self._gradient_norm(all_parameters)
            module_gradient_norms = self._module_gradient_norms(
                model, decoder, stage
            )
            if not math.isfinite(gradient_norm):
                nonfinite = True
                break
            optimizer.step()
            should_report = (
                epoch == 0
                or (epoch + 1) % self.conditioning.evaluation_interval == 0
                or epoch + 1 == epochs
            )
            if should_report:
                model.eval()
                with torch.no_grad():
                    check = model(
                        batch, ablation="H2", decode_teacher=False,
                        boundary_mode="no_event_jump",
                    )
                    check_features = (
                        check["boundary_features"] - feature_mean
                    ) / feature_std
                    check_residual, check_raw = decoder(check_features)
                    metrics = self._voltage_metrics(
                        check["voltage"] + check_residual, target
                    )
                passed = (
                    self._passes_one(metrics)
                    if len(indices) == 1 else self._passes_pair(metrics)
                )
                confirmations = confirmations + 1 if passed else 0
                row = {
                    "sample": sample_name, "stage": stage,
                    "epoch": epoch + 1,
                    "loss": float(loss.detach()),
                    "voltage_loss": float(voltage_loss.detach()),
                    "branch_loss": float(branch_loss.detach()),
                    "gradient_norm_pre_step": gradient_norm,
                    "gradient_clipping_applied": False,
                    "saturation_fraction": self._saturation_fraction(
                        check_raw, parameterization
                    ),
                    "passed": passed, **module_gradient_norms, **metrics,
                }
                history.append(row)
                progress.update(
                    epoch + 1,
                    f"V={metrics['voltage_rmse_mv']:.3g} "
                    f"max={metrics['maximum_segment_error_mv']:.3g}",
                )
                if confirmations >= self.conditioning.early_stop_confirmations:
                    break
                model.train()
        model.eval()
        with torch.no_grad():
            output = model(
                batch, ablation="H2", decode_teacher=False,
                boundary_mode="no_event_jump",
            )
            features = (output["boundary_features"] - feature_mean) / feature_std
            residual, raw = decoder(features)
            metrics = self._voltage_metrics(output["voltage"] + residual, target)
        final = {
            "sample": sample_name,
            "stage": stage,
            "parameterization": parameterization,
            "decoder_learning_rate": float(decoder_learning_rate),
            "model_learning_rate": self.conditioning.model_learning_rate,
            "epochs_budget": int(epochs),
            "epochs_completed": int(history[-1]["epoch"] if history else 0),
            "seed": int(seed),
            "trainable_model_parameter_count": int(sum(
                parameter.numel() for parameter in model_parameters
            )),
            "decoder_parameter_count": int(sum(
                parameter.numel() for parameter in decoder.parameters()
            )),
            "nonfinite": nonfinite,
            "saturation_fraction": self._saturation_fraction(raw, parameterization),
            "passed": bool(
                not nonfinite and (
                    self._passes_one(metrics)
                    if len(indices) == 1 else self._passes_pair(metrics)
                )
            ),
            "checkpoint_compatibility": compatibility,
            **metrics,
        }
        label = f"{sample_name}_{stage}"
        _write_json(
            self.output_dir / f"ladder_{label}.json",
            {"run": final, "history": history},
        )
        trained_model = {
            name: parameter.detach().cpu()
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }
        torch.save(
            {
                "model_trainable": trained_model,
                "decoder": decoder.state_dict(),
                "feature_mean": feature_mean.cpu(),
                "feature_std": feature_std.cpu(),
                "run": final,
            },
            self.output_dir / "checkpoints" / f"ladder_{label}.pt",
        )
        self.ladder_rows.extend(history)
        del model, decoder
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return final

    def run_unfreezing_ladder(
        self, sweep_report: Mapping[str, Any]
    ) -> Dict[str, Any]:
        require_torch()
        best = sweep_report.get("best")
        if not best:
            raise RuntimeError("no finite frozen-decoder run is available")
        parameterization = str(best["parameterization"])
        learning_rate = float(best["learning_rate"])
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        runs = []
        for sample_index, (sample_name, indices, epochs) in enumerate((
            (
                "one_transition", [self.worst_transition],
                self.conditioning.one_transition_epochs,
            ),
            (
                "branch_pair", list(self.branch_pair or ()),
                self.conditioning.branch_pair_epochs,
            ),
        )):
            if not indices:
                raise RuntimeError(f"missing indices for {sample_name}")
            for stage_index, stage in enumerate(self.conditioning.unfreezing_stages):
                runs.append(self._train_ladder_stage(
                    sample_name=sample_name,
                    indices=indices,
                    stage=stage,
                    parameterization=parameterization,
                    decoder_learning_rate=learning_rate,
                    epochs=epochs,
                    seed=self.conditioning.seed + 500 + 100 * sample_index + stage_index,
                    device=device,
                ))
        write_parquet(
            self.output_dir / "unfreezing_ladder_history.parquet",
            self.ladder_rows,
        )
        write_parquet(
            self.output_dir / "unfreezing_ladder_metrics.parquet", runs
        )
        report = {
            "schema_version": "05d-unfreezing-ladder-v1",
            "valid": len(runs) == 2 * len(self.conditioning.unfreezing_stages),
            "selected_decoder": {
                "parameterization": parameterization,
                "learning_rate": learning_rate,
            },
            "runs": runs,
        }
        _write_json(self.output_dir / "unfreezing_ladder_report.json", report)
        return report

    def finalize_conditioning(
        self,
        free_report: Mapping[str, Any],
        sweep_report: Optional[Mapping[str, Any]],
        ladder_report: Optional[Mapping[str, Any]],
    ) -> Dict[str, Any]:
        if not free_report.get("passed"):
            diagnosis = "OPTIMIZER_SANITY_CONTROL_FAILED"
        elif sweep_report and sweep_report.get("all_runs_nonfinite"):
            diagnosis = "FROZEN_DECODER_NONFINITE"
        elif not sweep_report or not ladder_report:
            diagnosis = "CONDITIONING_LADDER_INCOMPLETE"
        else:
            runs = {
                (row["sample"], row["stage"]): row
                for row in ladder_report["runs"]
            }
            head_both = all(
                runs[(sample, "head_only")]["passed"]
                for sample in ("one_transition", "branch_pair")
            )
            recovering_stages = [
                stage for stage in self.conditioning.unfreezing_stages
                if all(
                    runs[(sample, stage)]["passed"]
                    for sample in ("one_transition", "branch_pair")
                )
            ]
            any_one = any(
                runs[("one_transition", stage)]["passed"]
                for stage in self.conditioning.unfreezing_stages
            )
            if head_both:
                diagnosis = "FROZEN_FEATURES_SUFFICIENT_CONDITIONING_FIXED"
            elif recovering_stages:
                diagnosis = "PROGRESSIVE_UNFREEZING_RECOVERS"
            elif any_one:
                diagnosis = "BRANCH_IDENTIFIABILITY_FAILURE"
            else:
                diagnosis = "SHARED_REPRESENTATION_BOTTLENECK"
        report = {
            "schema_version": "05d-final-report-v1",
            "valid": bool(
                free_report.get("valid")
                and (sweep_report is None or sweep_report.get("valid"))
                and (ladder_report is None or ladder_report.get("valid"))
            ),
            "decision": "DIAGNOSTIC_ONLY_NO_FULL_TRAINING",
            "full_training_authorized": False,
            "diagnosis": diagnosis,
            "code_revision": self.code_revision,
            "dataset_fingerprint": self.bundle.fingerprint,
            "worst_transition": self.worst_transition,
            "worst_segment": self.worst_segment,
            "checkpoint_05b": self.checkpoint_contract,
            "artifact_05c": self.artifact_05c_contract,
            "free_residual": free_report,
            "frozen_decoder_sweep": sweep_report,
            "unfreezing_ladder": ladder_report,
            "next_decision": (
                "Choose 05e only from this conditioning diagnosis; full "
                "training remains forbidden."
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
            "schema_version": "05d-artifact-index-v1",
            "artifact_count": len(records),
            "artifacts": records,
        })
        return report
