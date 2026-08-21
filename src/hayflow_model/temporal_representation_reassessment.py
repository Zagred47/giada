"""Frozen category and time-shift audit of the verified initial-state signal."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from pathlib import Path
from typing import Any, Dict, Mapping, Tuple

import numpy as np

from src.hayflow_data.composite_flowmap import CompositeFlowmapBundle

from .hines_experiment import _write_json
from .hines_isolation_experiment import sha256_file
from .hines_regenerative_confirmation import _verified_artifact_root
from .rollout_aware_architecture_canary import torch
from .temporal_observability_reassessment import (
    TemporalObservabilityContractReassessment,
    TemporalObservabilityReassessmentConfig,
)


EXPECTED_05Q_ARCHIVE_SHA256 = (
    "73ed6fc3e244f49fe5172e5e917b253519e0bd7b614ab88584f1f8a06b41e0ec"
)
EXPECTED_05Q_INDEX_SHA256 = (
    "26ed1c29cded7a53f2d2dfce0519b19f6dfa5b9f4945a2d0867f046a51ac5d9b"
)
EXPECTED_05Q_FINAL_SHA256 = (
    "51a29770690b9fd52306d5946bb5611649da03a702f06c72be27b392166cd94e"
)


def verified_temporal_observability_artifact_root(
    source: Path, cache_dir: Path
) -> Tuple[Path, Dict[str, Any], Dict[str, Any]]:
    return _verified_artifact_root(
        Path(source),
        Path(cache_dir),
        marker_name="temporal_observability_reassessment_config.json",
        archive_sha256=EXPECTED_05Q_ARCHIVE_SHA256,
        index_sha256=EXPECTED_05Q_INDEX_SHA256,
        final_sha256=EXPECTED_05Q_FINAL_SHA256,
    )


@dataclass(frozen=True)
class TemporalRepresentationReassessmentConfig(
    TemporalObservabilityReassessmentConfig
):
    category_ablation_materiality_fraction: float = 0.02
    temporal_shift_materiality_fraction: float = 0.02
    sketch_reconstruction_atol: float = 1.0e-5

    def validate(self) -> None:
        super().validate()
        if not 0 < self.category_ablation_materiality_fraction < 1:
            raise ValueError("05r category materiality threshold is invalid")
        if not 0 < self.temporal_shift_materiality_fraction < 1:
            raise ValueError("05r shift materiality threshold is invalid")
        if not 0 < self.sketch_reconstruction_atol <= 1.0e-4:
            raise ValueError("05r sketch reconstruction tolerance is invalid")

    @classmethod
    def from_mapping(
        cls, values: Mapping[str, Any]
    ) -> "TemporalRepresentationReassessmentConfig":
        payload = dict(values)
        for name in ("horizons_ms", "seeds"):
            if name in payload:
                payload[name] = tuple(map(int, payload[name]))
        result = cls(**payload)
        result.validate()
        return result


class TemporalRepresentationReassessment(
    TemporalObservabilityContractReassessment
):
    def __init__(
        self,
        bundle: CompositeFlowmapBundle,
        output_dir: Path,
        config: TemporalRepresentationReassessmentConfig,
        artifact_05q_source: Path,
        artifact_05p_source: Path,
        artifact_05o_source: Path,
        *,
        code_revision: str,
    ) -> None:
        super().__init__(
            bundle,
            output_dir,
            config,
            artifact_05p_source,
            artifact_05o_source,
            code_revision=code_revision,
        )
        self.artifact_05q_source = Path(artifact_05q_source).resolve()
        self.artifact_05q_report: Dict[str, Any] = {}
        self.artifact_05q_contract: Dict[str, Any] = {}

    @property
    def representation_config(self) -> TemporalRepresentationReassessmentConfig:
        return self.config  # type: ignore[return-value]

    def prepare(self) -> Dict[str, Any]:
        support = super().prepare()
        _, final, contract = verified_temporal_observability_artifact_root(
            self.artifact_05q_source,
            self.output_dir.parent / ".05r_artifact_cache" / "05q",
        )
        blockers = []
        if not final.get("valid") or not final.get("decision_grade"):
            blockers.append("05q is not decision-grade")
        if final.get("diagnosis") != "FROZEN_COUNTERFACTUALS_DO_NOT_SUPPORT_TEMPORAL_SYNERGY":
            blockers.append("05q diagnosis does not authorize 05r")
        if final.get("next_step") != "05r_temporal_representation_reassessment":
            blockers.append("05q next step is not 05r")
        if not final.get("joint_state_reliance") or not final.get(
            "joint_state_identity_signal"
        ):
            blockers.append("05q did not verify state identity dependence")
        if final.get("joint_axial_reliance"):
            blockers.append("05q unexpectedly verified material axial reliance")
        if final.get("bounded_rollout_expansion_authorized"):
            blockers.append("05q unexpectedly authorized rollout expansion")
        self.artifact_05q_report = final
        self.artifact_05q_contract = contract
        categories = sorted(
            {
                str(row["category"])
                for index, row in enumerate(self.store.layout.core_records)
                if str(row["category"]) != "voltage"
                and np.any(self.projection[index] != 0)
            }
        )
        if not categories:
            blockers.append("05r found no active non-voltage state categories")
        report = {
            "schema_version": "05r-preflight-v1",
            "valid": not blockers,
            "blockers": blockers,
            "artifact_05q": contract,
            "artifact_05p": support["artifact_05p"],
            "artifact_05o": support["artifact_05o"],
            "dataset_fingerprint": self.bundle.fingerprint,
            "active_state_categories": categories,
            "frozen_joint_checkpoint_only": True,
            "retraining_performed": False,
            "new_model_selection_performed": False,
            "development_role_only_for_counterfactuals": True,
            "future_teacher_state_used_only_for_registered_time_shift_diagnostic": True,
            "validation_or_test_loaded": False,
            "sealed_fresh_test_loaded": False,
            "support_reconstruction": support,
        }
        self.prepare_report = report
        _write_json(self.output_dir / "preflight_report.json", report)
        _write_json(
            self.output_dir / "temporal_representation_reassessment_config.json",
            {
                "schema_version": "05r-config-v1",
                "config": asdict(self.representation_config),
                "artifact_05q": contract,
                "artifact_05p": support["artifact_05p"],
                "artifact_05o": support["artifact_05o"],
                "dataset_fingerprint": self.bundle.fingerprint,
                "code_revision": self.code_revision,
            },
        )
        if blockers:
            raise RuntimeError(f"05r preflight failed: {blockers}")
        return report

    def _normalized_state(self, indices: np.ndarray) -> np.ndarray:
        raw = self.store.read_state(indices, "t")
        times = np.asarray(
            self.store.metadata["start_time_ms"][indices], dtype=np.float64
        )
        semantic = self.semantic_encoder.encode(raw, times)
        return self.normalizer.normalize_state(semantic)

    def _group_sketch(
        self, normalized: np.ndarray, category: str | None
    ) -> np.ndarray:
        values = np.clip(
            np.asarray(normalized),
            -self.config.state_clip,
            self.config.state_clip,
        ).astype(np.float32)
        segment_ids = np.asarray(
            self.store.layout.core_segment_ids, dtype=np.int64
        )
        categories = np.asarray([
            str(row["category"]) for row in self.store.layout.core_records
        ])
        output = np.zeros(
            (
                len(values),
                self.store.layout.segment_count,
                self.config.state_sketch_dim,
            ),
            dtype=np.float32,
        )
        for segment in range(self.store.layout.segment_count):
            indices = np.flatnonzero(segment_ids == segment)
            if not len(indices):
                continue
            active = np.any(self.projection[indices] != 0, axis=1)
            denominator = math.sqrt(max(1, int(active.sum())))
            selected = (
                indices
                if category is None
                else indices[categories[indices] == category]
            )
            if len(selected):
                output[:, segment] = (
                    values[:, selected] @ self.projection[selected]
                    / denominator
                )
        return output

    def _evaluate_variant(
        self, model: Any, values: Mapping[str, Any], state: Any, permutation: Any
    ) -> Dict[str, Any]:
        changed = dict(values)
        changed["initial_state"] = state
        prediction = self._closed_prediction(
            model,
            changed,
            state_mode="authentic",
            axial_enabled=True,
            permutation=permutation,
        )
        return self._evaluate_prediction(prediction, changed)

    def run(self) -> Dict[str, Any]:
        if not self.prepare_report:
            self.prepare()
        if torch is None or self.artifact_05p_root is None:
            raise RuntimeError("05r requires PyTorch and verified source artifacts")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        values = self._tensor_role("development", device)
        starts = np.asarray(
            [int(window[0]) for window in self.windows["development"]],
            dtype=np.int64,
        )
        shifted_one = np.asarray(
            [int(window[1]) for window in self.windows["development"]],
            dtype=np.int64,
        )
        shifted_four = np.asarray(
            [int(window[4]) for window in self.windows["development"]],
            dtype=np.int64,
        )
        normalized = self._normalized_state(starts)
        full = self._group_sketch(normalized, None)
        categories = tuple(self.prepare_report["active_state_categories"])
        contributions = {
            category: self._group_sketch(normalized, category)
            for category in categories
        }
        reconstructed = sum(contributions.values(), np.zeros_like(full))
        reconstruction_error = float(np.max(np.abs(full - reconstructed)))
        stored_error = float(np.max(np.abs(
            full - self.materialized["development"]["initial_state"]
        )))
        if max(reconstruction_error, stored_error) > self.config.sketch_reconstruction_atol:
            raise RuntimeError("05r additive state sketches do not reproduce 05p")
        state_variants = {
            "full": full,
            "zero": np.zeros_like(full),
            "shift_plus_1ms": self._group_sketch(
                self._normalized_state(shifted_one), None
            ),
            "shift_plus_4ms": self._group_sketch(
                self._normalized_state(shifted_four), None
            ),
        }
        for category, contribution in contributions.items():
            state_variants[f"ablate_{category}"] = full - contribution
            state_variants[f"only_{category}"] = contribution
        permutation = torch.arange(
            len(starts), dtype=torch.long, device=device
        )
        results: Dict[str, Dict[str, Any]] = {
            name: {} for name in state_variants
        }
        joint = "graphgru_axial_rich_state"
        reproduction_errors = []
        with torch.no_grad():
            for seed in self.config.seeds:
                seed_key = str(seed)
                model = self._model(joint, device)
                source = self.artifact_05p_report["runs"][joint][seed_key]
                checkpoint = torch.load(
                    self.artifact_05p_root / source["checkpoint"],
                    map_location=device,
                    weights_only=False,
                )
                model.load_state_dict(checkpoint["state_dict"])
                model.eval()
                for name, state in state_variants.items():
                    tensor = torch.as_tensor(
                        state, dtype=torch.float32, device=device
                    )
                    results[name][seed_key] = self._evaluate_variant(
                        model, values, tensor, permutation
                    )
                stored = source["development"]["8"]["endpoint_rmse_mv"]
                reproduced = results["full"][seed_key]["8"]["endpoint_rmse_mv"]
                reproduction_errors.append(abs(stored - reproduced))
                print(
                    f"[HayFlow 05r][frozen state] seed={seed} "
                    f"full8={reproduced:.3f} zero8="
                    f"{results['zero'][seed_key]['8']['endpoint_rmse_mv']:.3f}",
                    flush=True,
                )
        maximum_checkpoint_error = max(reproduction_errors, default=math.inf)
        if maximum_checkpoint_error > self.config.checkpoint_reproduction_atol_mv:
            raise RuntimeError("05r checkpoint reproduction disagrees with 05p")

        def degradation(variant: str) -> Dict[str, Any]:
            return self._comparison(
                {"full": results["full"], "variant": results[variant]},
                "full",
                "variant",
                self.config.seeds,
            )

        category_ablation = {
            category: degradation(f"ablate_{category}")
            for category in categories
        }
        category_only_vs_zero = {
            category: self._comparison(
                {
                    "only": results[f"only_{category}"],
                    "zero": results["zero"],
                },
                "only",
                "zero",
                self.config.seeds,
            )
            for category in categories
        }
        temporal_shift = {
            "plus_1ms": degradation("shift_plus_1ms"),
            "plus_4ms": degradation("shift_plus_4ms"),
        }
        all_predictions_finite = all(
            metrics["finite"]
            for by_seed in results.values()
            for by_horizon in by_seed.values()
            for metrics in by_horizon.values()
        )
        if not all_predictions_finite:
            raise RuntimeError("05r produced a non-finite frozen prediction")

        def material(row: Mapping[str, Any], threshold: float) -> bool:
            return bool(
                row["median_rmse_gain_fraction"] >= threshold
                and row["median_regenerative_gain_fraction"]
                >= -self.config.regenerative_noninferiority_margin_fraction
                and row["positive_win_count"]
                >= self.config.minimum_counterfactual_seed_count
            )

        material_categories = [
            category
            for category, row in category_ablation.items()
            if material(
                row,
                self.representation_config.category_ablation_materiality_fraction,
            )
        ]
        shift_signals = {
            name: material(
                row,
                self.representation_config.temporal_shift_materiality_fraction,
            )
            for name, row in temporal_shift.items()
        }
        if len(material_categories) == 1:
            diagnosis = "TEMPORAL_STATE_SIGNAL_LOCALIZED_TO_ONE_CATEGORY"
            next_step = f"05s_{material_categories[0]}_state_encoder_canary"
        elif len(material_categories) > 1:
            diagnosis = "TEMPORAL_STATE_SIGNAL_DISTRIBUTED_ACROSS_CATEGORIES"
            next_step = "05s_structured_multigroup_state_encoder_canary"
        elif any(shift_signals.values()):
            diagnosis = "TEMPORAL_PRECISION_SIGNAL_WITHOUT_STABLE_CATEGORY_LOCALIZATION"
            next_step = "05s_time_aware_state_encoder_canary"
        else:
            diagnosis = "FROZEN_STATE_SIGNAL_NOT_STABLE_UNDER_REPRESENTATION_AUDIT"
            next_step = "05s_state_projection_stability_reassessment"
        report = {
            "schema_version": "05r-final-report-v1",
            "valid": all_predictions_finite,
            "decision_grade": True,
            "diagnosis": diagnosis,
            "code_revision": self.code_revision,
            "device": str(device),
            "artifact_05q": self.artifact_05q_contract,
            "artifact_05p": self.prepare_report["artifact_05p"],
            "artifact_05o": self.prepare_report["artifact_05o"],
            "dataset_fingerprint": self.bundle.fingerprint,
            "maximum_checkpoint_reproduction_error_mv": maximum_checkpoint_error,
            "maximum_full_sketch_reconstruction_error": reconstruction_error,
            "maximum_stored_sketch_reproduction_error": stored_error,
            "all_predictions_finite": all_predictions_finite,
            "state_category_ablation_degradations": category_ablation,
            "state_category_only_gains_over_zero": category_only_vs_zero,
            "teacher_state_time_shift_degradations": temporal_shift,
            "material_state_categories": material_categories,
            "teacher_state_time_shift_signals": shift_signals,
            "counterfactual_results": results,
            "retraining_performed": False,
            "model_or_training_authorized": False,
            "future_teacher_state_used_only_for_diagnostic_counterfactual": True,
            "validation_or_test_loaded": False,
            "sealed_fresh_test_loaded": False,
            "bounded_encoder_canary_authorized": bool(material_categories),
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
                "schema_version": "05r-artifact-index-v1",
                "artifact_count": len(records),
                "artifacts": records,
            },
        )
        return report


__all__ = [
    "EXPECTED_05Q_ARCHIVE_SHA256",
    "EXPECTED_05Q_FINAL_SHA256",
    "EXPECTED_05Q_INDEX_SHA256",
    "TemporalRepresentationReassessment",
    "TemporalRepresentationReassessmentConfig",
    "verified_temporal_observability_artifact_root",
]
