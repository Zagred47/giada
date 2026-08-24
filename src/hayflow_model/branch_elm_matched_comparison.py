"""Matched frozen-HayFlow completion of the Branch-ELM sidecar benchmark.

The original 05j-o number used 64 preregistered boundary transitions, whereas
the Branch-ELM fresh metric uses every post-burn-in transition in the same 64
episodes (512 transitions).  This module evaluates the already-frozen HayFlow
candidate on that exact 512-transition metric support without training,
selection, or architecture changes.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import shutil
import time
import zipfile
from dataclasses import asdict, fields
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from ..hayflow_data.hines_inputs import (
    canonical_anchor_segment_ids,
    encode_realized_synaptic_drive,
    explicit_teacher_views,
)
from ..hayflow_data.reconditioned_flowmap import (
    ReconditionedStateNormalizer,
    ReconditioningConfig,
)
from .hayflow_hines import HayFlowHines, hayflow_hines_arrays
from .hines_experiment import HinesPrototypeExperimentConfig, _write_json
from .hines_isolation_experiment import (
    EXPECTED_05B_ARCHIVE_SHA256,
    EXPECTED_05B_MEMBER_SHA256,
    sha256_file,
)
from .hines_netcon_semantic_repair import HinesNetConSemanticRepairConfig
from .hines_regenerative_confirmation import _verified_artifact_root
from .hines_regenerative_fresh_test import (
    EXPECTED_05JN_ARCHIVE_SHA256,
    EXPECTED_05JN_FINAL_SHA256,
    EXPECTED_05JN_INDEX_SHA256,
)
from .hines_spatial_support_revision import (
    HinesSpatialSupportRevisionConfig,
    apply_channel_pca,
    axial_tree_diffusion,
)
from .hines_state_normalization_repair import (
    HinesStateNormalizationRepairConfig,
    semantic_state_scale_repair,
)
from .hines_synaptic_domain_repair import (
    BoundedSynapticStateEncoder,
    HinesSynapticDomainRepairConfig,
)
from .hines_trainable_topology_canary import (
    HinesTrainableTopologyCanaryConfig,
    TrainableTopologyResidualHead,
)

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


EXPECTED_BRANCH_ELM_RESUME_ARCHIVE_SHA256 = (
    "721ba0bb5bffecaa11e46f9aa0fbec081a935142d97e13a13daff77acf0ab816"
)
EXPECTED_BRANCH_ELM_RESUME_INDEX_SHA256 = (
    "5c76a099b61a547d467b6894980a0a6ab0a1aa38752e97be75aa71877caf9dc0"
)
EXPECTED_BRANCH_ELM_RESUME_FINAL_SHA256 = (
    "8a3913d4630203fd35b6ce0d14be811d6b44bfea3f3f9661e6c25d7fc9a8a1b7"
)


def _one_suffix(names: Sequence[str], suffix: str) -> str:
    matches = [name for name in names if name.replace("\\", "/").endswith(suffix)]
    if len(matches) != 1:
        raise RuntimeError(f"expected one member ending in {suffix!r}, found {matches}")
    return matches[0]


def _verified_05b_checkpoint(source: Path) -> Tuple[bytes, Dict[str, Any]]:
    """Return the exact frozen H2 checkpoint bytes and provenance contract."""

    source = Path(source).expanduser().resolve()
    archive_hash: Optional[str] = None
    if source.is_file():
        archive_hash = sha256_file(source)
        if archive_hash != EXPECTED_05B_ARCHIVE_SHA256:
            raise RuntimeError("05b archive SHA-256 mismatch")
        with zipfile.ZipFile(source) as archive:
            payload = {
                suffix: archive.read(_one_suffix(archive.namelist(), suffix))
                for suffix in EXPECTED_05B_MEMBER_SHA256
            }
        kind = "original_zip"
    elif source.is_dir():
        payload = {}
        for suffix in EXPECTED_05B_MEMBER_SHA256:
            parts = tuple(Path(suffix).parts)
            matches = [
                path
                for path in source.rglob(parts[-1])
                if tuple(path.parts[-len(parts) :]) == parts
            ]
            if len(matches) != 1:
                raise RuntimeError(f"expected one extracted 05b {suffix}, found {matches}")
            payload[suffix] = matches[0].read_bytes()
        kind = "kaggle_extracted_directory"
    else:
        raise RuntimeError(f"05b source does not exist: {source}")
    observed = {name: hashlib.sha256(value).hexdigest() for name, value in payload.items()}
    mismatches = {
        name: {"expected": EXPECTED_05B_MEMBER_SHA256[name], "observed": digest}
        for name, digest in observed.items()
        if digest != EXPECTED_05B_MEMBER_SHA256[name]
    }
    if mismatches:
        raise RuntimeError(f"05b indexed member mismatch: {mismatches}")
    return payload["checkpoints/canary_models.pt"], {
        "source_kind": kind,
        "source_path": str(source),
        "archive_sha256": archive_hash,
        "verified_member_sha256": observed,
    }


def restore_registered_branch_elm_checkpoints(
    source: Path, output_dir: Path, cache_dir: Path
) -> Dict[str, Any]:
    """Restore the six completed ELM checkpoints without importing old reports."""

    root, report, contract = _verified_artifact_root(
        Path(source),
        Path(cache_dir),
        marker_name="benchmark_contract.json",
        archive_sha256=EXPECTED_BRANCH_ELM_RESUME_ARCHIVE_SHA256,
        index_sha256=EXPECTED_BRANCH_ELM_RESUME_INDEX_SHA256,
        final_sha256=EXPECTED_BRANCH_ELM_RESUME_FINAL_SHA256,
    )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoints = sorted(root.glob("branch_elm_8002_*.pt"))
    if len(checkpoints) != 6:
        raise RuntimeError(f"registered Branch-ELM artifact contains {len(checkpoints)} checkpoints")
    copied = []
    for source_path in checkpoints:
        destination = output_dir / source_path.name
        shutil.copy2(source_path, destination)
        copied.append({"path": destination.name, "sha256": sha256_file(destination)})
    return {
        "valid": bool(report.get("valid") and contract["all_indexed_members_verified"]),
        "checkpoint_count": len(copied),
        "checkpoints": copied,
        "source": contract,
        "reports_imported": False,
        "retraining_performed": False,
    }


def _dataclass_mapping(cls: Any, payload: Mapping[str, Any]) -> Dict[str, Any]:
    accepted = {field.name for field in fields(cls)}
    return {key: value for key, value in payload.items() if key in accepted}


class MatchedFrozenHayFlowComparison:
    """Evaluate the frozen HayFlow one-step candidate on Branch-ELM's support."""

    def __init__(
        self,
        branch_session: Any,
        h2_source: Path,
        refit_source: Path,
        *,
        batch_size: int = 32,
    ) -> None:
        if torch is None:
            raise RuntimeError("matched HayFlow evaluation requires PyTorch")
        if batch_size <= 0:
            raise ValueError("matched HayFlow batch size must be positive")
        self.session = branch_session
        self.h2_source = Path(h2_source)
        self.refit_source = Path(refit_source)
        self.batch_size = int(batch_size)
        self.output_dir = Path(branch_session.output_dir)
        self.store = branch_session.fresh_store
        self.root = Path(branch_session.fresh_root)
        if self.store is None or not self.root.is_dir():
            raise RuntimeError("Branch-ELM fresh store must be prepared first")
        self.layout = self.store.layout
        self.arrays = hayflow_hines_arrays(self.layout)
        self.anchors = canonical_anchor_segment_ids(self.layout)

    @staticmethod
    def _json(path: Path) -> Dict[str, Any]:
        return json.loads(Path(path).read_text(encoding="utf-8"))

    def _load_normalizer(self) -> Tuple[Any, Any, Dict[str, Any]]:
        normalization = self._json(self.root / "normalization_schema.json")
        state = normalization["state"]
        normalizer = ReconditionedStateNormalizer(
            self.layout, ReconditioningConfig(**state["config"])
        )
        array_specs = {
            "transform_codes": np.int8,
            "state_center": np.float64,
            "state_scale": np.float64,
            "delta_center": np.float64,
            "delta_scale": np.float64,
            "update_fraction": np.float64,
            "sparse_mask": bool,
            "activity_positive_weight": np.float64,
        }
        for name, dtype in array_specs.items():
            setattr(normalizer, name, np.asarray(state[name], dtype=dtype))
        repair_payload = self._json(self.root / "state_normalization_repair_config.json")
        repair = HinesStateNormalizationRepairConfig.from_mapping(repair_payload["repair"])
        domain_payload = self._json(self.root / "synaptic_domain_repair_config.json")
        domain = HinesSynapticDomainRepairConfig.from_mapping(
            _dataclass_mapping(HinesSynapticDomainRepairConfig, domain_payload["domain_repair"])
        )
        netcon = HinesNetConSemanticRepairConfig()
        encoder = BoundedSynapticStateEncoder(self.layout, netcon, domain)
        stored_codes = normalizer.transform_codes.copy()
        encoder.configure_transform_codes(normalizer)
        if not np.array_equal(stored_codes, normalizer.transform_codes):
            raise RuntimeError("embedded normalization transform codes disagree with bounded encoder")
        repaired, _ = semantic_state_scale_repair(
            encoder.records, normalizer.transform_codes, normalizer.state_scale, repair
        )
        repaired[encoder.recency_indices] = np.maximum(
            repaired[encoder.recency_indices], domain.bounded_recency_scale_floor
        )
        repaired[encoder.trace_indices] = np.maximum(
            repaired[encoder.trace_indices], domain.synaptic_trace_log1p_scale_floor
        )
        normalizer.state_scale = repaired
        verification = self._json(self.root / "verified_synaptic_domain_normalizer.json")
        digest = hashlib.sha256()
        digest.update(json.dumps(asdict(repair), sort_keys=True).encode())
        for values in (
            normalizer.transform_codes,
            normalizer.state_center,
            normalizer.state_scale,
        ):
            digest.update(np.ascontiguousarray(values).tobytes())
        observed = digest.hexdigest()
        expected = str(verification["repaired_normalizer_fingerprint"])
        if observed != expected:
            raise RuntimeError(
                f"reconstructed frozen normalizer fingerprint mismatch: {observed} != {expected}"
            )
        return normalizer, encoder, {
            "repair_fingerprint": observed,
            "verified_repair_fingerprint": expected,
            "runtime_normalizer_fingerprint": normalizer.fingerprint(),
            "state_width": int(len(normalizer.state_scale)),
        }

    def _load_transform(self, refit_root: Path) -> Dict[str, Any]:
        with np.load(refit_root / "refit_feature_transform.npz") as archive:
            return {
                "surfaces": {
                    surface: {
                        name: archive[f"{prefix}_{name}"]
                        for name in ("mean", "scale", "channel_mean", "components")
                    }
                    for surface, prefix in (("h2_raw", "h2"), ("causal_raw", "causal"))
                },
                "raw_mean": archive["raw_mean"],
                "raw_scale": archive["raw_scale"],
            }

    def _topology_design(
        self,
        role: Mapping[str, np.ndarray],
        transform: Mapping[str, Any],
        spatial: Any,
    ) -> np.ndarray:
        denominator = np.arcsinh(spatial.input_asinh_reference_z)
        sketches = []
        for name in ("h2_raw", "causal_raw"):
            row = transform["surfaces"][name]
            values = np.asarray(role[name], dtype=np.float64)
            transformed = np.arcsinh((values - row["mean"]) / row["scale"]) / denominator
            sketches.append(
                apply_channel_pca(transformed, row["channel_mean"], row["components"])
            )
        local = np.concatenate(sketches, axis=-1)
        tree = axial_tree_diffusion(
            local,
            np.asarray(self.arrays["parent_ids"], dtype=np.int64),
            np.asarray(self.arrays["axial_conductance_to_parent_us"], dtype=np.float64),
            spatial.diffusion_scales,
            spatial.diffusion_self_weight,
        )
        raw = np.concatenate(
            [role["voltage_t"][..., None] / 100.0, role["base"][..., None] / 100.0, tree],
            axis=-1,
        )
        return (
            np.arcsinh((raw - transform["raw_mean"]) / transform["raw_scale"])
            / denominator
        ).astype(np.float32)

    def _batch_role(
        self,
        h2: Any,
        indices: np.ndarray,
        normalizer: Any,
        encoder: Any,
        model_config: Any,
        device: Any,
    ) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
        raw_t = self.store.read_state(indices, "t")
        times = np.asarray(self.store.metadata["start_time_ms"][indices], dtype=np.float64)
        semantic_t = encoder.encode(raw_t, times)
        normalized_t = normalizer.normalize_state(semantic_t).astype(np.float32)
        voltage_t = raw_t[:, : self.layout.segment_count].astype(np.float32)
        calcium_t, synapse_t = explicit_teacher_views(
            normalized_t,
            self.arrays,
            segment_count=self.layout.segment_count,
            calcium_dim=model_config.calcium_state_dim,
            synapse_dim=model_config.synapse_state_dim,
        )
        raw_batch: Dict[str, Any] = {
            "indices": indices,
            "teacher_state_t": normalized_t,
            "voltage_t": voltage_t,
            "calcium_t": calcium_t,
            "synapse_state_t": synapse_t,
            "anchor_voltage_t": voltage_t[:, self.anchors],
            "anchor_segment_ids": self.anchors,
        }
        raw_batch.update(
            encode_realized_synaptic_drive(
                self.store,
                indices,
                voltage_t,
                dt_ms=model_config.dt_ms,
                raw_state_t=raw_t,
            )
        )
        batch = {
            key: torch.as_tensor(
                value,
                dtype=(torch.long if key in {"indices", "anchor_segment_ids"} else torch.float32),
                device=device,
            )
            if isinstance(value, np.ndarray)
            else value
            for key, value in raw_batch.items()
        }
        with torch.no_grad():
            output = h2(
                batch,
                ablation="H2",
                decode_teacher=False,
                boundary_mode="no_event_jump",
            )
        causal = torch.cat(
            [
                batch["synaptic_features"],
                batch["synaptic_conductance_us"].unsqueeze(-1),
                batch["synaptic_source_na"].unsqueeze(-1),
                batch["somatic_current_na"].unsqueeze(-1),
            ],
            dim=-1,
        )
        target = self.store.read_state(indices, "t_plus_1", categories=("voltage",))
        return {
            "base": output["voltage"].detach().cpu().double().numpy(),
            "h2_raw": output["boundary_features"].detach().cpu().double().numpy(),
            "causal_raw": causal.detach().cpu().double().numpy(),
            "voltage_t": voltage_t.astype(np.float64),
        }, np.asarray(target, dtype=np.float64)

    @staticmethod
    def _soma_metrics(prediction: np.ndarray, target: np.ndarray) -> Dict[str, float]:
        clipped = np.minimum(target, -55.0)
        return {
            "clipped_soma_rmse_mv": float(np.sqrt(np.mean((prediction - clipped) ** 2))),
            "raw_soma_rmse_mv_secondary": float(np.sqrt(np.mean((prediction - target) ** 2))),
            "soma_mae_mv": float(np.mean(np.abs(prediction - clipped))),
        }

    def evaluate(self, elm_results: Mapping[str, Any]) -> Dict[str, Any]:
        h2_bytes, h2_contract = _verified_05b_checkpoint(self.h2_source)
        refit_root, refit_report, refit_contract = _verified_artifact_root(
            self.refit_source,
            self.output_dir.parent / ".06bc_matched_artifact_cache" / "05jn",
            marker_name="regenerative_decoder_refit_config.json",
            archive_sha256=EXPECTED_05JN_ARCHIVE_SHA256,
            index_sha256=EXPECTED_05JN_INDEX_SHA256,
            final_sha256=EXPECTED_05JN_FINAL_SHA256,
        )
        normalizer, encoder, normalizer_contract = self._load_normalizer()
        model_payload = self._json(self.root / "model_configurations.json")
        experiment = HinesPrototypeExperimentConfig.from_mapping(model_payload["experiment"])
        spatial_payload = self._json(self.root / "spatial_support_revision_config.json")
        spatial = HinesSpatialSupportRevisionConfig.from_mapping(
            spatial_payload["spatial_support"]
        )
        topology_payload = self._json(self.root / "trainable_topology_canary_config.json")
        topology = HinesTrainableTopologyCanaryConfig.from_mapping(
            topology_payload["trainable_topology_canary"]
        )
        transform = self._load_transform(refit_root)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        h2 = HayFlowHines(experiment.model, self.layout.to_model_metadata(), self.arrays).to(device)
        checkpoint = torch.load(io.BytesIO(h2_bytes), map_location=device, weights_only=True)
        incompatible = h2.load_state_dict(checkpoint["H2"], strict=False)
        allowed_missing = {"direct_boundary_residual.weight", "direct_boundary_residual.bias"}
        if set(incompatible.missing_keys) != allowed_missing or incompatible.unexpected_keys:
            raise RuntimeError(f"unexpected H2 checkpoint incompatibility: {incompatible}")
        h2.eval()
        rows_by_seed = {
            int(row["seed"]): row for row in refit_report["decoder_refit"]["runs"]
        }
        heads = {}
        for seed in (17, 29, 43):
            row = rows_by_seed[seed]
            payload = torch.load(
                refit_root / str(row["checkpoint"]), map_location=device, weights_only=False
            )
            if payload["family"] != "direct_tree_refit" or int(payload["seed"]) != seed:
                raise RuntimeError(f"refit checkpoint identity mismatch for seed {seed}")
            feature_width = int(transform["raw_mean"].shape[-1])
            head = TrainableTopologyResidualHead(
                feature_width,
                self.layout.segment_count,
                topology.hidden_width,
                topology.segment_embedding_dim,
                topology.target_residual_limit_mv,
            ).to(device)
            head.load_state_dict(payload["state_dict"])
            head.eval()
            heads[seed] = head
        compatible = [
            trajectory
            for trajectory in self.store.trajectory_indices
            if not self.session._trajectory_has_somatic_current(self.store, trajectory)
        ]
        logical_indices = np.sort(np.concatenate(
            [
                np.asarray(self.store.trajectory_indices[trajectory], dtype=np.int64)[
                    self.session.config.fresh_test_burn_in_ms :
                ]
                for trajectory in compatible
            ]
        ))
        expected_count = len(compatible) * (
            12 - self.session.config.fresh_test_burn_in_ms
        )
        if len(compatible) != 64 or len(logical_indices) != expected_count or expected_count != 512:
            raise RuntimeError(
                f"matched fresh support is {len(compatible)} episodes/{len(logical_indices)} transitions"
            )
        predictions = {seed: [] for seed in heads}
        targets = []
        started = time.monotonic()
        chunk_count = math.ceil(len(logical_indices) / self.batch_size)
        for chunk_index, start in enumerate(range(0, len(logical_indices), self.batch_size), 1):
            indices = logical_indices[start : start + self.batch_size]
            role, target = self._batch_role(
                h2, indices, normalizer, encoder, experiment.model, device
            )
            design = self._topology_design(role, transform, spatial)
            features = torch.as_tensor(design, dtype=torch.float32, device=device)
            with torch.no_grad():
                for seed, head in heads.items():
                    residual = head(features).detach().cpu().numpy()
                    predictions[seed].append((role["base"] + residual)[:, 0])
            targets.append(target[:, 0])
            elapsed = time.monotonic() - started
            eta = (chunk_count - chunk_index) * elapsed / max(chunk_index, 1)
            print(
                f"[HayFlow ELM][matched frozen HayFlow] {chunk_index}/{chunk_count} "
                f"({100 * chunk_index / chunk_count:.1f}%) ETA {eta / 60:.1f} min",
                flush=True,
            )
        target_soma = np.concatenate(targets)
        joined = {seed: np.concatenate(values) for seed, values in predictions.items()}
        runs = {
            str(seed): self._soma_metrics(prediction, target_soma)
            for seed, prediction in joined.items()
        }
        ensemble = self._soma_metrics(
            np.mean(np.stack(list(joined.values())), axis=0), target_soma
        )
        hayflow_values = [runs[str(seed)]["clipped_soma_rmse_mv"] for seed in (17, 29, 43)]
        elm_values = {
            seed: float(
                elm_results["retrained_exact_architecture"]["U_realized"][str(seed)][
                    "fresh_test"
                ]["clipped_soma_rmse_mv"]
            )
            for seed in (17, 29, 43)
        }
        paired = {
            str(seed): {
                "hayflow_clipped_soma_rmse_mv": runs[str(seed)]["clipped_soma_rmse_mv"],
                "branch_elm_U_realized_clipped_soma_rmse_mv": elm_values[seed],
                "hayflow_error_reduction_vs_branch_elm_fraction": 1.0
                - runs[str(seed)]["clipped_soma_rmse_mv"] / elm_values[seed],
            }
            for seed in (17, 29, 43)
        }
        h2_parameters = int(sum(value.numel() for value in h2.parameters()))
        head_parameters = int(sum(value.numel() for value in heads[17].parameters()))
        report = {
            "schema_version": "06b-c-matched-hayflow-v1",
            "valid": True,
            "comparison_complete_for_voltage": True,
            "comparison_complete_for_spikes": False,
            "spike_comparison_reason": "The shared 512-transition support contains zero positive somatic spikes.",
            "metric_contract": {
                "dataset": "same 05j-o fresh dataset",
                "episode_count": len(compatible),
                "evaluated_transition_count": len(logical_indices),
                "burn_in_ms_per_episode": self.session.config.fresh_test_burn_in_ms,
                "target": "segment 0 soma voltage at t_plus_1 clipped above -55 mV",
                "rmse_pooling": "pooled over the same 512 transitions",
                "same_target": True,
                "same_transitions": True,
                "same_burn_in": True,
                "same_metric": True,
                "same_input_contract": False,
                "input_contract_warning": "Branch-ELM consumes event history only; frozen HayFlow consumes the complete teacher boundary state S_t plus U_realized.",
            },
            "frozen_hayflow": {
                "runs": runs,
                "median_clipped_soma_rmse_mv": float(np.median(hayflow_values)),
                "ensemble_mean": ensemble,
                "h2_parameter_count": h2_parameters,
                "refit_head_parameter_count": head_parameters,
                "combined_parameter_count": h2_parameters + head_parameters,
                "checkpoint_selection_performed": False,
                "retraining_performed": False,
            },
            "branch_elm_U_realized": {
                "runs": {
                    str(seed): {
                        "clipped_soma_rmse_mv": elm_values[seed],
                        "trainable_parameter_count": 8002,
                    }
                    for seed in (17, 29, 43)
                },
                "median_clipped_soma_rmse_mv": float(np.median(list(elm_values.values()))),
            },
            "paired_seed_comparison": paired,
            "median_hayflow_error_reduction_vs_branch_elm_fraction": 1.0
            - float(np.median(hayflow_values)) / float(np.median(list(elm_values.values()))),
            "provenance": {
                "h2": h2_contract,
                "refit": refit_contract,
                "normalizer": normalizer_contract,
                "original_05jo_selected_boundary_rmse_mv": {
                    str(row["seed"]): row["metrics"]["aggregate_voltage_rmse_mv"]
                    for row in self._json(self.root / "frozen_fresh_test_evaluation.json")["runs"]
                },
            },
            "methodology": {
                "frozen_hayflow_checkpoints": True,
                "fresh_outcomes_used_for_selection": False,
                "retraining_performed": False,
                "architecture_search_performed": False,
                "elm_primary_view_for_matched_comparison": "U_realized",
                "U_scheduled_retained_as_original_end_to_end_elm_secondary_view": True,
            },
        }
        _write_json(self.output_dir / "matched_hayflow_comparison.json", report)
        return report


__all__ = [
    "EXPECTED_BRANCH_ELM_RESUME_ARCHIVE_SHA256",
    "EXPECTED_BRANCH_ELM_RESUME_INDEX_SHA256",
    "EXPECTED_BRANCH_ELM_RESUME_FINAL_SHA256",
    "MatchedFrozenHayFlowComparison",
    "restore_registered_branch_elm_checkpoints",
]
