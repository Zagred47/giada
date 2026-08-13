"""Notebook-05j-f region/mechanism expert revision."""

from __future__ import annotations

import hashlib
import io
import json
import math
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np

from ..hayflow_eval.flowmap_metrics import write_parquet
from .hines_architecture_reassessment import HinesArchitectureReassessment
from .hines_experiment import Progress, _write_json
from .hines_isolation_experiment import sha256_file
from .hines_layer import require_torch
from .hines_repaired_representation_revision import pair_gate_selection_score

try:
    import torch
    from torch import nn
except ImportError:  # pragma: no cover
    torch = None
    nn = object


EXPECTED_05JE_ARCHIVE_SHA256 = (
    "baa35dd2eb605c0b74a3f0b98bd2d80118ea81fa95ce6823b7b2b63df82bb82f"
)
EXPECTED_05JE_INDEX_SHA256 = (
    "c35c23a9920e27926a78dd0485b4f1624370ed172b2af2b00b4b28c268bc2115"
)
EXPECTED_05JE_FINAL_SHA256 = (
    "76b917bed38c88e7bab29c6b14b7737bae54c8a3bdfdd130e3780df806ec7a3f"
)


EXPERT_NAMES = (
    "general",
    "apical_trunk",
    "basal",
    "tuft",
    "soma_axon",
    "calcium_regenerative",
    "sodium_regenerative",
    "repolarization_and_h",
)


@dataclass(frozen=True)
class HinesRegionMechanismExpertConfig:
    families: Tuple[str, ...] = (
        "uniform_expert_control", "region_mechanism_experts",
    )
    expert_names: Tuple[str, ...] = EXPERT_NAMES
    seeds: Tuple[int, ...] = (17, 29, 43)
    expert_hidden_width: int = 32
    segment_embedding_dim: int = 16
    epochs: int = 1000
    evaluation_interval: int = 20
    patience: int = 240
    learning_rate: float = 8e-4
    weight_decay: float = 1e-5
    gradient_clip_norm: float = 5.0
    target_residual_limit_mv: float = 120.0
    error_scale_mv: float = 20.0
    branch_loss_weight: float = 1.0
    tail_loss_weight: float = 0.25
    tail_fraction: float = 0.05
    selection_max_error_weight: float = 0.05
    selection_branch_log_weight: float = 2.0
    minimum_passing_seeds: int = 2
    minimum_signal_seeds: int = 2
    within_seed_rmse_improvement_fraction: float = 0.15
    within_seed_max_error_improvement_fraction: float = 0.10
    expert_vs_uniform_improvement_fraction: float = 0.10
    checkpoint_reconstruction_metric_atol: float = 2e-4

    def validate(self) -> None:
        if tuple(self.families) != (
            "uniform_expert_control", "region_mechanism_experts",
        ):
            raise ValueError("05j-f requires the uniform and structured expert families")
        if tuple(self.expert_names) != EXPERT_NAMES:
            raise ValueError("05j-f expert definitions are preregistered")
        if tuple(self.seeds) != (17, 29, 43):
            raise ValueError("05j-f must reuse the registered three seeds")
        if not 1 <= self.minimum_passing_seeds <= len(self.seeds):
            raise ValueError("invalid robust pass seed count")
        if not 1 <= self.minimum_signal_seeds <= len(self.seeds):
            raise ValueError("invalid robust signal seed count")
        positive = (
            self.expert_hidden_width, self.segment_embedding_dim, self.epochs,
            self.evaluation_interval, self.patience, self.learning_rate,
            self.gradient_clip_norm, self.target_residual_limit_mv,
            self.error_scale_mv, self.branch_loss_weight,
            self.checkpoint_reconstruction_metric_atol,
        )
        if min(positive) <= 0:
            raise ValueError("05j-f configuration values must be positive")
        fractions = (
            self.tail_fraction, self.within_seed_rmse_improvement_fraction,
            self.within_seed_max_error_improvement_fraction,
            self.expert_vs_uniform_improvement_fraction,
        )
        if any(not 0 < value < 1 for value in fractions):
            raise ValueError("05j-f fractions must lie in (0, 1)")
        if self.tail_loss_weight < 0 or self.weight_decay < 0:
            raise ValueError("05j-f loss and decay weights cannot be negative")

    @classmethod
    def from_mapping(
        cls, values: Mapping[str, Any]
    ) -> "HinesRegionMechanismExpertConfig":
        payload = dict(values)
        for name in ("families", "expert_names", "seeds"):
            if name in payload:
                payload[name] = tuple(payload[name])
        result = cls(**payload)
        result.validate()
        return result


def region_mechanism_expert_gates(layout: Any) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Build target-independent expert memberships from teacher metadata."""

    segment_count = int(layout.segment_count)
    mechanisms: List[set[str]] = [set() for _ in range(segment_count)]
    for segment, record in zip(layout.core_segment_ids, layout.core_records):
        mechanisms[int(segment)].add(str(record.get("mechanism", "")).lower())
    regions = [str(row.get("region", "unknown")).lower() for row in layout.segments]
    binary = np.zeros((segment_count, len(EXPERT_NAMES)), dtype=np.float64)
    binary[:, 0] = 1.0
    for segment, (region, names) in enumerate(zip(regions, mechanisms)):
        joined = " ".join(sorted(names))
        binary[segment, 1] = region == "apical_trunk"
        binary[segment, 2] = region == "basal"
        binary[segment, 3] = region == "tuft"
        binary[segment, 4] = region in {"soma", "axon", "ais"}
        binary[segment, 5] = any(token in joined for token in (
            "ca_hva", "ca_lva", "cadynamics", "calcium",
        ))
        binary[segment, 6] = any(token in joined for token in (
            "nata", "nats", "nap_", "na_",
        ))
        binary[segment, 7] = any(token in joined for token in (
            "sk", "kv", "im", "ih", "k_pst", "k_tst",
        ))
    gates = binary / np.maximum(binary.sum(axis=1, keepdims=True), 1.0)
    counts = {
        name: int(np.sum(binary[:, index] > 0))
        for index, name in enumerate(EXPERT_NAMES)
    }
    signature_count = len({tuple(row.astype(int)) for row in binary})
    return gates.astype(np.float32), {
        "schema_version": "05j-f-expert-gates-v1",
        "expert_names": list(EXPERT_NAMES),
        "expert_segment_counts": counts,
        "mechanism_signature_count": signature_count,
        "all_segments_have_general_expert": bool(np.all(binary[:, 0] == 1)),
        "all_gate_rows_sum_to_one": bool(np.allclose(gates.sum(axis=1), 1.0)),
        "target_values_used": False,
        "development_values_used": False,
    }


if torch is not None:
    class GatedExpertCorrection(nn.Module):
        """Zero-initialized expert correction around a frozen 05j-d prediction."""

        def __init__(
            self, feature_width: int, segment_count: int, hidden_width: int,
            segment_embedding_dim: int, gates: np.ndarray,
            residual_limit_mv: float,
        ) -> None:
            super().__init__()
            self.residual_limit_mv = float(residual_limit_mv)
            gate_array = np.asarray(gates, dtype=np.float32)
            if gate_array.shape[0] != segment_count:
                raise ValueError("expert gate segment count mismatch")
            self.register_buffer("gates", torch.as_tensor(gate_array))
            self.segment_embedding = nn.Embedding(segment_count, segment_embedding_dim)
            input_width = feature_width + segment_embedding_dim
            self.experts = nn.ModuleList()
            for _ in range(gate_array.shape[1]):
                expert = nn.Sequential(
                    nn.Linear(input_width, hidden_width), nn.SiLU(),
                    nn.Linear(hidden_width, hidden_width), nn.SiLU(),
                    nn.Linear(hidden_width, 1),
                )
                nn.init.zeros_(expert[-1].weight)
                nn.init.zeros_(expert[-1].bias)
                self.experts.append(expert)

        def forward(self, features: Any, baseline_residual: Any) -> Any:
            samples, segments, _ = features.shape
            segment_ids = torch.arange(segments, device=features.device)
            embedding = self.segment_embedding(segment_ids)[None].expand(samples, -1, -1)
            values = torch.cat([features, embedding], dim=-1)
            corrections = torch.cat([expert(values) for expert in self.experts], dim=-1)
            correction = torch.sum(corrections * self.gates[None], dim=-1)
            bound = 1.0 - 1e-6
            baseline_logits = torch.atanh(torch.clamp(
                baseline_residual / self.residual_limit_mv, -bound, bound
            ))
            return self.residual_limit_mv * torch.tanh(baseline_logits + correction)
else:  # pragma: no cover
    class GatedExpertCorrection:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            require_torch()


def uniform_expert_gates(segment_count: int, expert_count: int) -> np.ndarray:
    return np.full((segment_count, expert_count), 1.0 / expert_count, dtype=np.float32)


class HinesRegionMechanismExpertRevision(HinesArchitectureReassessment):
    """Capacity-matched expert correction around frozen direct-tree checkpoints."""

    def __init__(
        self, *args: Any, expert_config: HinesRegionMechanismExpertConfig,
        artifact_05je_source: Path, **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        expert_config.validate()
        self.expert = expert_config
        self.artifact_05je_source = Path(artifact_05je_source).resolve()
        self.artifact_05je_contract: Dict[str, Any] = {}
        self.expert_gates: Dict[str, np.ndarray] = {}

    def _read_verified_05je(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        source = self.artifact_05je_source
        if source.is_file():
            if sha256_file(source) != EXPECTED_05JE_ARCHIVE_SHA256:
                raise RuntimeError("05j-e archive SHA-256 mismatch")
            with zipfile.ZipFile(source) as archive:
                names = archive.namelist()
                index_name = self._one_suffix(names, "artifact_index.json")
                root = index_name[:-len("artifact_index.json")]
                index_bytes = archive.read(index_name); index = json.loads(index_bytes)
                if hashlib.sha256(index_bytes).hexdigest() != EXPECTED_05JE_INDEX_SHA256:
                    raise RuntimeError("05j-e artifact index SHA-256 mismatch")
                for row in index["artifacts"]:
                    payload = archive.read(root + str(row["path"]).replace("\\", "/"))
                    if hashlib.sha256(payload).hexdigest() != row["sha256"] or len(payload) != int(row["size_bytes"]):
                        raise RuntimeError(f"05j-e indexed member mismatch: {row['path']}")
                final_bytes = archive.read(root + "final_report.json")
            kind, archive_hash = "original_zip", EXPECTED_05JE_ARCHIVE_SHA256
        elif source.is_dir():
            indices = [
                path for path in source.rglob("artifact_index.json")
                if (path.parent / "architecture_reassessment_config.json").is_file()
            ]
            if len(indices) != 1:
                raise RuntimeError("expected one extracted 05j-e artifact index")
            index_bytes = indices[0].read_bytes(); index = json.loads(index_bytes)
            if hashlib.sha256(index_bytes).hexdigest() != EXPECTED_05JE_INDEX_SHA256:
                raise RuntimeError("extracted 05j-e artifact index SHA-256 mismatch")
            root = indices[0].parent
            for row in index["artifacts"]:
                path = root / str(row["path"])
                if not path.is_file() or sha256_file(path) != row["sha256"] or path.stat().st_size != int(row["size_bytes"]):
                    raise RuntimeError(f"extracted 05j-e member mismatch: {row['path']}")
            final_bytes = (root / "final_report.json").read_bytes()
            kind, archive_hash = "kaggle_extracted_directory", None
        else:
            raise RuntimeError(f"05j-e source does not exist: {source}")
        if hashlib.sha256(final_bytes).hexdigest() != EXPECTED_05JE_FINAL_SHA256:
            raise RuntimeError("05j-e final report SHA-256 mismatch")
        return json.loads(final_bytes), {
            "source_kind": kind, "source_path": str(source),
            "archive_sha256": archive_hash,
            "artifact_index_sha256": EXPECTED_05JE_INDEX_SHA256,
            "final_report_sha256": EXPECTED_05JE_FINAL_SHA256,
            "verified_member_count": len(index["artifacts"]),
            "all_indexed_members_verified": len(index["artifacts"]) == int(index["artifact_count"]),
        }

    def prepare_region_mechanism_expert_revision(self) -> Dict[str, Any]:
        base = self.prepare_architecture_reassessment()
        report, contract = self._read_verified_05je()
        blockers = []
        if report.get("diagnosis") != "LOCALIZED_MORPHOLOGY_REGIME_ERROR_DOMINATES":
            blockers.append(f"unexpected 05j-e diagnosis: {report.get('diagnosis')}")
        if report.get("dataset_fingerprint") != self.bundle.fingerprint:
            blockers.append("05j-e dataset fingerprint mismatch")
        if report.get("micro_rollout_authorized") or report.get("full_training_authorized"):
            blockers.append("05j-e unexpectedly authorized downstream training")
        if report.get("methodology", {}).get("retraining_performed"):
            blockers.append("05j-e unexpectedly retrained a model")
        if report.get("methodology", {}).get("development_used_for_model_selection"):
            blockers.append("05j-e used development for model selection")
        if report.get("heldout_contract", {}).get("inputs_extracted"):
            blockers.append("05j-e heldout inputs were not sealed")
        if not contract["all_indexed_members_verified"]:
            blockers.append("05j-e artifact verification is incomplete")
        if blockers:
            raise RuntimeError(f"05j-f provenance blockers: {blockers}")
        self.artifact_05je_contract = contract
        payload = {
            "schema_version": "05j-f-region-mechanism-expert-config-v1",
            "region_mechanism_experts": asdict(self.expert),
            "artifact_05je": contract,
            "comparison": "capacity_matched_uniform_vs_metadata_gated_experts",
            "base_predictions": "frozen_direct_tree_per_seed",
            "expert_definition_roles": ["teacher_metadata_only"],
            "checkpoint_selection_roles": ["internal_calibration"],
            "development_used_for_model_selection": False,
            "heldout_inputs_extracted": False, "rollout_performed": False,
            "full_training_authorized": False, "code_revision": self.code_revision,
        }
        _write_json(self.output_dir / "region_mechanism_expert_config.json", payload)
        return {**base, **payload}

    def prepare_expert_gates(self) -> Dict[str, Any]:
        structured, report = region_mechanism_expert_gates(self.layout)
        empty = [name for name, count in report["expert_segment_counts"].items() if count == 0]
        if empty:
            raise RuntimeError(f"05j-f empty expert masks: {empty}")
        uniform = uniform_expert_gates(self.layout.segment_count, len(EXPERT_NAMES))
        self.expert_gates = {
            "uniform_expert_control": uniform,
            "region_mechanism_experts": structured,
        }
        report.update({
            "valid": bool(
                report["all_segments_have_general_expert"]
                and report["all_gate_rows_sum_to_one"] and not empty
            ),
            "uniform_control_shape": list(uniform.shape),
            "structured_gate_sha256": hashlib.sha256(structured.tobytes()).hexdigest(),
            "uniform_gate_sha256": hashlib.sha256(uniform.tobytes()).hexdigest(),
            "fit_targets_used": False, "calibration_targets_used": False,
            "development_targets_used": False,
        })
        _write_json(self.output_dir / "expert_gate_contract.json", report)
        if not report["valid"]:
            raise RuntimeError("05j-f expert gate contract failed")
        return report

    def _expert_loss(self, prediction: Any, target: Any) -> Any:
        scaled = (prediction - target) / self.expert.error_scale_mv
        point = torch.mean(scaled.square())
        paired = scaled.reshape(-1, 2, scaled.shape[-1])
        branch = torch.mean((paired[:, 0] - paired[:, 1]).square())
        k = max(1, int(math.ceil(scaled.shape[-1] * self.expert.tail_fraction)))
        tail = torch.topk(torch.abs(scaled), k, dim=1).values.square().mean()
        return point + self.expert.branch_loss_weight * branch + self.expert.tail_loss_weight * tail

    @staticmethod
    def _improvement(baseline: float, candidate: float) -> float:
        return float((baseline - candidate) / max(abs(baseline), 1e-12))

    def run_region_mechanism_expert_canary(self) -> Dict[str, Any]:
        require_torch()
        if not self.frozen_predictions:
            raise RuntimeError("reconstruct_frozen_checkpoints() must run first")
        if not self.expert_gates:
            raise RuntimeError("prepare_expert_gates() must run first")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tensors = {}
        for role in ("fit", "calibration", "development"):
            state = self.topology_roles[role]
            tensors[role] = {
                "features": torch.as_tensor(self.topology_designs[role], device=device),
                "target": torch.as_tensor(
                    np.asarray(state["target"]) - np.asarray(state["base"]),
                    dtype=torch.float32, device=device,
                ),
            }
        checkpoint_dir = self.output_dir / "checkpoints"; checkpoint_dir.mkdir(exist_ok=True)
        runs = []
        total = len(self.expert.families) * len(self.expert.seeds)
        overall = Progress("05j-f expert runs", total); completed = 0
        for family in self.expert.families:
            for seed in self.expert.seeds:
                torch.manual_seed(int(seed)); np.random.seed(int(seed))
                if torch.cuda.is_available(): torch.cuda.manual_seed_all(int(seed))
                if hasattr(torch.backends, "cudnn"):
                    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
                model = GatedExpertCorrection(
                    self.topology_designs["fit"].shape[-1], self.layout.segment_count,
                    self.expert.expert_hidden_width, self.expert.segment_embedding_dim,
                    self.expert_gates[family], self.expert.target_residual_limit_mv,
                ).to(device)
                optimizer = torch.optim.AdamW(
                    model.parameters(), lr=self.expert.learning_rate,
                    weight_decay=self.expert.weight_decay,
                )
                baseline = {
                    role: torch.as_tensor(
                        self.frozen_predictions["direct_tree"][int(seed)][role],
                        device=device,
                    ) for role in tensors
                }
                with torch.no_grad():
                    zero_error = float(torch.max(torch.abs(
                        model(tensors["fit"]["features"], baseline["fit"])
                        - baseline["fit"]
                    )).cpu())
                if zero_error > 2e-5:
                    raise RuntimeError(f"05j-f correction does not preserve baseline: {zero_error}")
                best_score, best_epoch, stale, best_state = math.inf, 0, 0, None
                history = []
                progress = Progress(f"05j-f {family} seed{seed}", self.expert.epochs)
                for epoch in range(1, self.expert.epochs + 1):
                    model.train(); optimizer.zero_grad(set_to_none=True)
                    prediction = model(tensors["fit"]["features"], baseline["fit"])
                    loss = self._expert_loss(prediction, tensors["fit"]["target"])
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.expert.gradient_clip_norm)
                    optimizer.step()
                    if epoch == 1 or epoch % self.expert.evaluation_interval == 0:
                        model.eval()
                        with torch.no_grad():
                            calibration = model(
                                tensors["calibration"]["features"], baseline["calibration"]
                            ).cpu().numpy()
                        metrics = self._role_metrics("calibration", calibration)
                        score = pair_gate_selection_score(
                            metrics, max_error_weight=self.expert.selection_max_error_weight,
                            branch_log_weight=self.expert.selection_branch_log_weight,
                        )
                        history.append({
                            "family": family, "seed": int(seed), "epoch": epoch,
                            "training_loss": float(loss.detach().cpu()),
                            "calibration_score": score,
                            "calibration_rmse_mv": metrics["aggregate_voltage_rmse_mv"],
                            "calibration_max_error_mv": metrics["maximum_segment_error_mv"],
                            "calibration_retention": metrics["median_branching_retention"],
                        })
                        if score < best_score - 1e-9:
                            best_score, best_epoch, stale = score, epoch, 0
                            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
                        else:
                            stale += self.expert.evaluation_interval
                        progress.update(epoch, f"loss={float(loss):.3g} cal={score:.3g}")
                        if stale >= self.expert.patience:
                            break
                if best_state is None:
                    raise RuntimeError("05j-f failed to select a checkpoint")
                model.load_state_dict(best_state); model.eval()
                role_metrics, baseline_metrics, improvements = {}, {}, {}
                with torch.no_grad():
                    for role in tensors:
                        candidate = model(tensors[role]["features"], baseline[role]).cpu().numpy()
                        role_metrics[role] = self._role_metrics(role, candidate)
                        baseline_metrics[role] = self._role_metrics(
                            role, baseline[role].cpu().numpy()
                        )
                        improvements[role] = {
                            "rmse_improvement_fraction": self._improvement(
                                baseline_metrics[role]["aggregate_voltage_rmse_mv"],
                                role_metrics[role]["aggregate_voltage_rmse_mv"],
                            ),
                            "maximum_error_improvement_fraction": self._improvement(
                                baseline_metrics[role]["maximum_segment_error_mv"],
                                role_metrics[role]["maximum_segment_error_mv"],
                            ),
                        }
                absolute_pass = all(row["passed"] for row in role_metrics.values())
                signal_pass = bool(
                    improvements["calibration"]["rmse_improvement_fraction"]
                    >= self.expert.within_seed_rmse_improvement_fraction
                    and improvements["development"]["rmse_improvement_fraction"]
                    >= self.expert.within_seed_rmse_improvement_fraction
                    and improvements["calibration"]["maximum_error_improvement_fraction"]
                    >= self.expert.within_seed_max_error_improvement_fraction
                    and improvements["development"]["maximum_error_improvement_fraction"]
                    >= self.expert.within_seed_max_error_improvement_fraction
                )
                checkpoint = checkpoint_dir / f"{family}_seed{seed}.pt"
                torch.save({
                    "state_dict": best_state, "family": family, "seed": int(seed),
                    "best_epoch": best_epoch, "dataset_fingerprint": self.bundle.fingerprint,
                    "base_checkpoint_family": "direct_tree",
                }, checkpoint)
                runs.append({
                    "family": family, "seed": int(seed), "best_epoch": best_epoch,
                    "best_calibration_score": best_score,
                    "zero_initialization_max_error_mv": zero_error,
                    "roles": role_metrics, "frozen_baseline_roles": baseline_metrics,
                    "improvement_vs_frozen_baseline": improvements,
                    "absolute_gate_passed": absolute_pass,
                    "within_seed_signal_passed": signal_pass,
                    "checkpoint": checkpoint.relative_to(self.output_dir).as_posix(),
                })
                write_parquet(self.output_dir / f"history_{family}_seed{seed}.parquet", history)
                completed += 1; overall.update(completed, f"{family} seed={seed} pass={absolute_pass}")
                del model
                if torch.cuda.is_available(): torch.cuda.empty_cache()
        summaries = []
        for family in self.expert.families:
            chosen = [row for row in runs if row["family"] == family]
            summaries.append({
                "family": family, "run_count": len(chosen),
                "absolute_passing_seed_count": int(sum(row["absolute_gate_passed"] for row in chosen)),
                "signal_passing_seed_count": int(sum(row["within_seed_signal_passed"] for row in chosen)),
                "robust_absolute_gate_passed": sum(row["absolute_gate_passed"] for row in chosen) >= self.expert.minimum_passing_seeds,
                "robust_within_seed_signal": sum(row["within_seed_signal_passed"] for row in chosen) >= self.expert.minimum_signal_seeds,
                **{
                    f"median_{role}_rmse_mv": float(np.median([
                        row["roles"][role]["aggregate_voltage_rmse_mv"] for row in chosen
                    ])) for role in ("fit", "calibration", "development")
                },
                **{
                    f"median_{role}_maximum_error_mv": float(np.median([
                        row["roles"][role]["maximum_segment_error_mv"] for row in chosen
                    ])) for role in ("calibration", "development")
                },
            })
        report = {
            "schema_version": "05j-f-region-mechanism-expert-canary-v1",
            "valid": True, "device": str(device), "runs": runs,
            "family_summary": summaries,
            "development_used_for_checkpoint_selection": False,
            "development_inference_after_checkpoint_freeze": True,
            "heldout_inputs_extracted": False, "rollout_performed": False,
        }
        _write_json(self.output_dir / "region_mechanism_expert_canary.json", report)
        write_parquet(self.output_dir / "region_mechanism_expert_run_summary.parquet", [{
            "family": row["family"], "seed": row["seed"],
            "best_epoch": row["best_epoch"],
            "absolute_gate_passed": row["absolute_gate_passed"],
            "within_seed_signal_passed": row["within_seed_signal_passed"],
            **{f"{role}_rmse_mv": row["roles"][role]["aggregate_voltage_rmse_mv"] for role in ("fit", "calibration", "development")},
            **{f"{role}_max_error_mv": row["roles"][role]["maximum_segment_error_mv"] for role in ("calibration", "development")},
        } for row in runs])
        return report

    @staticmethod
    def _family_summary(report: Mapping[str, Any], family: str) -> Mapping[str, Any]:
        matches = [row for row in report["family_summary"] if row["family"] == family]
        if len(matches) != 1:
            raise RuntimeError(f"expected one 05j-f family summary for {family}")
        return matches[0]

    def finalize_region_mechanism_expert_revision(
        self, gate_report: Mapping[str, Any], canary_report: Mapping[str, Any],
    ) -> Dict[str, Any]:
        uniform = self._family_summary(canary_report, "uniform_expert_control")
        expert = self._family_summary(canary_report, "region_mechanism_experts")
        cal_gain = self._improvement(
            uniform["median_calibration_rmse_mv"], expert["median_calibration_rmse_mv"]
        )
        dev_gain = self._improvement(
            uniform["median_development_rmse_mv"], expert["median_development_rmse_mv"]
        )
        specialization_specific = bool(
            cal_gain >= self.expert.expert_vs_uniform_improvement_fraction
            and dev_gain >= self.expert.expert_vs_uniform_improvement_fraction
        )
        passing_families = [
            row["family"] for row in (uniform, expert)
            if row["robust_absolute_gate_passed"]
        ]
        absolute = bool(passing_families)
        scoped_signal = bool(
            expert["robust_within_seed_signal"] and specialization_specific
        )
        if absolute and expert["robust_absolute_gate_passed"] and specialization_specific:
            diagnosis = "REGION_MECHANISM_EXPERTS_PASS_ROBUST_GATE"
            next_experiment = "05k_repaired_representation_micro_rollout"
        elif absolute:
            diagnosis = "ADDED_EXPERT_CAPACITY_PASSES_ROBUST_GATE"
            next_experiment = "05k_repaired_representation_micro_rollout"
        elif scoped_signal:
            diagnosis = "REGION_MECHANISM_SPECIALIZATION_HELPS_BUT_REMAINS_BELOW_GATE"
            next_experiment = "05j_g_regenerative_expert_refinement"
        elif expert["robust_within_seed_signal"]:
            diagnosis = "ADDED_CAPACITY_HELPS_WITHOUT_EXPERT_SPECIFICITY"
            next_experiment = "05j_g_regenerative_objective_and_capacity_revision"
        else:
            diagnosis = "REGION_MECHANISM_EXPERTS_DO_NOT_RESCUE_LOCALIZED_ERROR"
            next_experiment = "05j_g_regenerative_state_target_decomposition"
        report = {
            "schema_version": "05j-f-final-report-v1", "valid": True,
            "decision": "TRAIN_ONLY_REGION_MECHANISM_EXPERT_CANARY",
            "diagnosis": diagnosis, "code_revision": self.code_revision,
            "dataset_fingerprint": self.bundle.fingerprint,
            "artifact_05je": self.artifact_05je_contract,
            "expert_gate_contract": dict(gate_report),
            "expert_canary": dict(canary_report),
            "expert_vs_uniform": {
                "calibration_rmse_improvement_fraction": cal_gain,
                "development_rmse_improvement_fraction": dev_gain,
                "specialization_specific_material_improvement": specialization_specific,
            },
            "passing_families": passing_families,
            "region_mechanism_expert_revision_passed": bool(
                expert["robust_absolute_gate_passed"]
            ),
            "expert_canary_passed": absolute,
            "scoped_expert_signal": scoped_signal,
            "micro_rollout_authorized": absolute,
            "full_training_authorized": False,
            "heldout_contract": {
                "inputs_extracted": False, "boundary_targets_materialized": False,
                "event_targets_materialized": False, "candidate_inference_performed": False,
                "reveal_authorized": False,
            },
            "methodology": {
                "exact_05je_diagnosis_verified": True,
                "base_direct_tree_checkpoints_frozen": True,
                "expert_gates_from_teacher_metadata_only": True,
                "capacity_matched_uniform_control": True,
                "fit_calibration_split_unchanged": True,
                "calibration_used_for_checkpoint_selection": True,
                "development_used_for_checkpoint_selection": False,
                "development_inference_after_checkpoint_freeze": True,
                "heldout_inputs_extracted": False, "rollout_performed": False,
            },
            "next_step": {
                "requires_new_notebook": True, "experiment": next_experiment,
                "full_training_authorized": False,
            },
        }
        _write_json(self.output_dir / "final_report.json", report)
        records = []
        for path in sorted(self.output_dir.rglob("*")):
            if path.is_file() and path.name != "artifact_index.json":
                records.append({
                    "path": path.relative_to(self.output_dir).as_posix(),
                    "size_bytes": path.stat().st_size, "sha256": sha256_file(path),
                })
        _write_json(self.output_dir / "artifact_index.json", {
            "schema_version": "05j-f-artifact-index-v1",
            "artifact_count": len(records), "artifacts": records,
        })
        return report
