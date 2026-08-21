"""Bounded paired canary for a mechanism-aligned initial-state encoder."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import math
from pathlib import Path
import random
import time
from typing import Any, Dict, List, Mapping, Tuple

import numpy as np

from src.hayflow_data.composite_flowmap import CompositeFlowmapBundle

from .axial_rich_state_recurrent_canary import AxialRichStateGraphGRU
from .graph_state_contract_reassessment import sketch_normalized_state
from .hines_experiment import _write_json
from .hines_isolation_experiment import sha256_file
from .hines_regenerative_confirmation import _verified_artifact_root
from .rollout_aware_architecture_canary import model_parameter_count, torch
from .temporal_representation_reassessment import (
    TemporalRepresentationReassessment,
    TemporalRepresentationReassessmentConfig,
)


EXPECTED_05R_ARCHIVE_SHA256 = (
    "7126ab8540934a60483bb4bcf3d64a9e5e6cc90364322a4df0b3ad70b84dff64"
)
EXPECTED_05R_INDEX_SHA256 = (
    "2b2e731006373ba1ec53a0cbd8099e9a269b1e095e75039d35c043bf8de88912"
)
EXPECTED_05R_FINAL_SHA256 = (
    "5c876547680eaee2bd8c12e73f63bf381294c3a9cf7e181fcc34468182c019c5"
)


def verified_temporal_representation_artifact_root(
    source: Path, cache_dir: Path
) -> Tuple[Path, Dict[str, Any], Dict[str, Any]]:
    return _verified_artifact_root(
        Path(source),
        Path(cache_dir),
        marker_name="temporal_representation_reassessment_config.json",
        archive_sha256=EXPECTED_05R_ARCHIVE_SHA256,
        index_sha256=EXPECTED_05R_INDEX_SHA256,
        final_sha256=EXPECTED_05R_FINAL_SHA256,
    )


@dataclass(frozen=True)
class MechanismStateEncoderCanaryConfig(
    TemporalRepresentationReassessmentConfig
):
    epochs: int = 30
    progress_interval: int = 15
    representation_seed: int = 510091
    candidate_material_gain_fraction: float = 0.02
    minimum_paired_win_count: int = 2

    def validate(self) -> None:
        super().validate()
        if self.epochs != 30:
            raise ValueError("05s uses the registered 30-epoch micro-canary budget")
        if self.progress_interval != 15:
            raise ValueError("05s progress must remain compact")
        if self.representation_seed <= 0:
            raise ValueError("05s representation seed is invalid")
        if not 0 < self.candidate_material_gain_fraction < 1:
            raise ValueError("05s candidate materiality threshold is invalid")
        if self.minimum_paired_win_count not in {2, 3}:
            raise ValueError("05s paired seed gate is invalid")

    @classmethod
    def from_mapping(
        cls, values: Mapping[str, Any]
    ) -> "MechanismStateEncoderCanaryConfig":
        payload = dict(values)
        for name in ("horizons_ms", "seeds"):
            if name in payload:
                payload[name] = tuple(map(int, payload[name]))
        result = cls(**payload)
        result.validate()
        return result


def shared_semantic_state_projection(
    layout: Any,
    *,
    dimension: int,
    seed: int,
    category: str | None,
) -> np.ndarray:
    """Map equal mechanism-variable identities to equal signed vectors."""

    projection = np.zeros((layout.state_width, int(dimension)), dtype=np.float32)
    cache: Dict[str, np.ndarray] = {}
    scale = math.sqrt(float(dimension))
    for index, record in enumerate(layout.core_records):
        record_category = str(record["category"])
        if record_category == "voltage":
            continue
        if category is not None and record_category != category:
            continue
        key = "|".join(
            (
                record_category,
                str(record["mechanism"]),
                str(record["variable"]),
                str(record["kind"]),
            )
        )
        if key not in cache:
            payload = hashlib.shake_256(
                f"{int(seed)}|{key}".encode("utf-8")
            ).digest(int(dimension))
            cache[key] = np.asarray(
                [1.0 if value & 1 else -1.0 for value in payload],
                dtype=np.float32,
            ) / scale
        projection[index] = cache[key]
    return projection


class MechanismStateEncoderCanary(TemporalRepresentationReassessment):
    REPRESENTATIONS = (
        "legacy_full_signed",
        "semantic_full",
        "semantic_mechanism_states",
    )

    def __init__(
        self,
        bundle: CompositeFlowmapBundle,
        output_dir: Path,
        config: MechanismStateEncoderCanaryConfig,
        artifact_05r_source: Path,
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
            artifact_05q_source,
            artifact_05p_source,
            artifact_05o_source,
            code_revision=code_revision,
        )
        self.artifact_05r_source = Path(artifact_05r_source).resolve()
        self.artifact_05r_report: Dict[str, Any] = {}
        self.artifact_05r_contract: Dict[str, Any] = {}
        self.representation_states: Dict[str, Dict[str, np.ndarray]] = {}

    @property
    def encoder_config(self) -> MechanismStateEncoderCanaryConfig:
        return self.config  # type: ignore[return-value]

    def _model_for_representation(self, device: Any) -> Any:
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
            state_width=self.encoder_config.state_sketch_dim,
            hidden_width=self.config.hidden_width,
            voltage_delta_limit_mv=self.config.voltage_delta_limit_mv,
            use_axial=True,
            use_rich_state=True,
        ).to(device)

    def prepare(self) -> Dict[str, Any]:
        support = super().prepare()
        _, final, contract = verified_temporal_representation_artifact_root(
            self.artifact_05r_source,
            self.output_dir.parent / ".05s_artifact_cache" / "05r",
        )
        blockers = []
        if not final.get("valid") or not final.get("decision_grade"):
            blockers.append("05r is not decision-grade")
        if final.get("diagnosis") != "TEMPORAL_STATE_SIGNAL_LOCALIZED_TO_ONE_CATEGORY":
            blockers.append("05r did not localize a single state category")
        if final.get("material_state_categories") != ["mechanism_states"]:
            blockers.append("05r did not localize the signal to mechanism states")
        if final.get("next_step") != "05s_mechanism_states_state_encoder_canary":
            blockers.append("05r next step is not 05s")
        if not final.get("bounded_encoder_canary_authorized"):
            blockers.append("05r did not authorize a bounded encoder canary")
        self.artifact_05r_report = final
        self.artifact_05r_contract = contract
        semantic_full = shared_semantic_state_projection(
            self.store.layout,
            dimension=self.encoder_config.state_sketch_dim,
            seed=self.encoder_config.representation_seed,
            category=None,
        )
        semantic_mechanism = shared_semantic_state_projection(
            self.store.layout,
            dimension=self.encoder_config.state_sketch_dim,
            seed=self.encoder_config.representation_seed,
            category="mechanism_states",
        )
        if not np.any(semantic_mechanism):
            blockers.append("05s mechanism-state projection is empty")
        for role, windows in self.windows.items():
            starts = np.asarray(
                [int(window[0]) for window in windows], dtype=np.int64
            )
            normalized = self._normalized_state(starts)
            legacy = np.asarray(
                self.materialized[role]["initial_state"], dtype=np.float32
            )
            self.representation_states[role] = {
                "legacy_full_signed": legacy,
                "semantic_full": sketch_normalized_state(
                    normalized,
                    self.store.layout,
                    semantic_full,
                    clip=self.encoder_config.state_clip,
                ),
                "semantic_mechanism_states": sketch_normalized_state(
                    normalized,
                    self.store.layout,
                    semantic_mechanism,
                    clip=self.encoder_config.state_clip,
                ),
            }
            for name, state in self.representation_states[role].items():
                if state.shape != legacy.shape or not np.isfinite(state).all():
                    blockers.append(f"05s {role}/{name} state is invalid")
        if torch is None:
            blockers.append("PyTorch is unavailable")
            parameter_count = 0
        else:
            model = self._model_for_representation(torch.device("cpu"))
            parameter_count = model_parameter_count(model)
        report = {
            "schema_version": "05s-preflight-v1",
            "valid": not blockers,
            "blockers": blockers,
            "artifact_05r": contract,
            "artifact_05q": support["artifact_05q"],
            "artifact_05p": support["artifact_05p"],
            "artifact_05o": support["artifact_05o"],
            "dataset_fingerprint": self.bundle.fingerprint,
            "representations": list(self.REPRESENTATIONS),
            "paired_parameter_count": parameter_count,
            "parameter_identical": True,
            "shared_semantic_identity_across_segments": True,
            "mechanism_candidate_excludes_other_state_categories": True,
            "source_split_used": "train_only",
            "fit_selects_gradients": True,
            "calibration_selects_checkpoint": True,
            "development_read_once_after_freeze": True,
            "validation_or_test_loaded": False,
            "sealed_fresh_test_loaded": False,
            "fresh_test_generated": False,
            "teacher_future_state_used_as_model_input": False,
            "support_reconstruction": support,
        }
        self.prepare_report = report
        _write_json(self.output_dir / "preflight_report.json", report)
        _write_json(
            self.output_dir / "mechanism_state_encoder_canary_config.json",
            {
                "schema_version": "05s-config-v1",
                "config": asdict(self.encoder_config),
                "artifact_05r": contract,
                "dataset_fingerprint": self.bundle.fingerprint,
                "code_revision": self.code_revision,
            },
        )
        if blockers:
            raise RuntimeError(f"05s preflight failed: {blockers}")
        return report

    def _tensor_representation_role(
        self, role: str, representation: str, device: Any
    ) -> Dict[str, Any]:
        values = self._tensor_role(role, device)
        values["initial_state"] = torch.as_tensor(
            self.representation_states[role][representation],
            dtype=torch.float32,
            device=device,
        )
        return values

    def _train_one(
        self, representation: str, seed: int, device: Any
    ) -> Tuple[Any, Dict[str, Any]]:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        model = self._model_for_representation(device)
        initial_sha256 = self._state_sha256(model)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        fit = self._tensor_representation_role("fit", representation, device)
        calibration = self._tensor_representation_role(
            "calibration", representation, device
        )
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
                target = fit["target_voltage"].index_select(0, positions)[
                    :, :horizon
                ]
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
                    f"[HayFlow 05s][{representation} seed={seed}] "
                    f"{epoch + 1}/{self.config.epochs} ETA {eta / 60:.1f} min "
                    f"loss={row['fit_loss']:.4g}",
                    flush=True,
                )
        if best_state is None:
            raise RuntimeError("05s produced no calibration checkpoint")
        model.load_state_dict(best_state)
        checkpoint = self.checkpoint_dir / f"{representation}-seed{seed}.pt"
        torch.save(
            {
                "state_dict": best_state,
                "representation": representation,
                "seed": seed,
                "best_epoch": best_epoch,
                "dataset_fingerprint": self.bundle.fingerprint,
                "code_revision": self.code_revision,
            },
            checkpoint,
        )
        return model, {
            "representation": representation,
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

    def _paired_gain(
        self,
        runs: Mapping[str, Any],
        candidate: str,
        baseline: str,
    ) -> Dict[str, Any]:
        by_seed = {}
        for seed in map(str, self.config.seeds):
            left = runs[candidate][seed]["development"]["8"]
            right = runs[baseline][seed]["development"]["8"]
            by_seed[seed] = {
                "rmse_gain_fraction": 1.0
                - left["endpoint_rmse_mv"]
                / max(right["endpoint_rmse_mv"], 1e-12),
                "regenerative_rmse_gain_fraction": 1.0
                - left["regenerative_endpoint_rmse_mv"]
                / max(right["regenerative_endpoint_rmse_mv"], 1e-12),
            }
        return {
            "candidate": candidate,
            "baseline": baseline,
            "by_seed": by_seed,
            "median_rmse_gain_fraction": float(np.median([
                row["rmse_gain_fraction"] for row in by_seed.values()
            ])),
            "median_regenerative_gain_fraction": float(np.median([
                row["regenerative_rmse_gain_fraction"]
                for row in by_seed.values()
            ])),
            "positive_win_count": sum(
                row["rmse_gain_fraction"] > 0 for row in by_seed.values()
            ),
        }

    def _material(self, comparison: Mapping[str, Any]) -> bool:
        return bool(
            comparison["median_rmse_gain_fraction"]
            >= self.encoder_config.candidate_material_gain_fraction
            and comparison["median_regenerative_gain_fraction"]
            >= -self.config.regenerative_noninferiority_margin_fraction
            and comparison["positive_win_count"]
            >= self.encoder_config.minimum_paired_win_count
        )

    def run(self) -> Dict[str, Any]:
        if not self.prepare_report:
            self.prepare()
        if torch is None:
            raise RuntimeError("05s requires PyTorch")
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        runs: Dict[str, Dict[str, Any]] = {
            name: {} for name in self.REPRESENTATIONS
        }
        for representation in self.REPRESENTATIONS:
            for seed in self.config.seeds:
                model, row = self._train_one(representation, int(seed), device)
                development = self._tensor_representation_role(
                    "development", representation, device
                )
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
                        )
                        == 0
                        and improvement
                        >= self.config.minimum_improvement_vs_persistence_fraction
                    ),
                )
                runs[representation][str(seed)] = row
        paired_initialization = {
            str(seed): len({
                runs[name][str(seed)]["initial_state_sha256"]
                for name in self.REPRESENTATIONS
            })
            == 1
            for seed in self.config.seeds
        }
        paired_order = {
            str(seed): len({
                runs[name][str(seed)]["training_order_sha256"]
                for name in self.REPRESENTATIONS
            })
            == 1
            for seed in self.config.seeds
        }
        comparisons = {
            "semantic_alignment_given_all_state": self._paired_gain(
                runs, "semantic_full", "legacy_full_signed"
            ),
            "mechanism_localization_given_semantic_alignment": self._paired_gain(
                runs, "semantic_mechanism_states", "semantic_full"
            ),
            "mechanism_encoder_vs_legacy": self._paired_gain(
                runs, "semantic_mechanism_states", "legacy_full_signed"
            ),
        }
        passing = {
            name: sum(bool(row["passed"]) for row in values.values())
            for name, values in runs.items()
        }
        mechanism_material = self._material(
            comparisons["mechanism_encoder_vs_legacy"]
        )
        semantic_full_material = self._material(
            comparisons["semantic_alignment_given_all_state"]
        )
        mechanism_robust = (
            passing["semantic_mechanism_states"]
            >= self.config.minimum_passing_seed_count
        )
        semantic_full_robust = (
            passing["semantic_full"] >= self.config.minimum_passing_seed_count
        )
        if mechanism_material and mechanism_robust:
            diagnosis = "MECHANISM_STATE_ENCODER_CANARY_PASSED"
            selected = "semantic_mechanism_states"
            next_step = "05t_consolidated_autoregressive_go_no_go"
        elif semantic_full_material and semantic_full_robust:
            diagnosis = "SEMANTIC_ALIGNMENT_SIGNAL_WITHOUT_MECHANISM_LOCALIZATION"
            selected = "semantic_full"
            next_step = "05t_consolidated_autoregressive_go_no_go"
        else:
            diagnosis = "MECHANISM_STATE_ENCODER_CANARY_DID_NOT_PASS"
            selected = None
            next_step = "stop_current_state_encoder_branch"
        valid = bool(
            all(paired_initialization.values())
            and all(paired_order.values())
            and all(
                metric["finite"]
                for representation in runs.values()
                for row in representation.values()
                for metric in row["development"].values()
            )
        )
        report = {
            "schema_version": "05s-final-report-v1",
            "valid": valid,
            "decision_grade": True,
            "diagnosis": diagnosis,
            "code_revision": self.code_revision,
            "device": str(device),
            "artifact_05r": self.artifact_05r_contract,
            "dataset_fingerprint": self.bundle.fingerprint,
            "runs": runs,
            "paired_representation_gains": comparisons,
            "paired_initialization_verified_by_seed": paired_initialization,
            "paired_training_order_verified_by_seed": paired_order,
            "passing_seed_count": passing,
            "mechanism_encoder_material": mechanism_material,
            "semantic_full_encoder_material": semantic_full_material,
            "selected_representation": selected,
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
            "bounded_autoregressive_go_no_go_authorized": bool(selected),
            "full_training_authorized": False,
            "fresh_test_generation_authorized": False,
            "mass_dataset_generation_authorized": False,
            "next_step": next_step,
        }
        _write_json(self.output_dir / "final_report.json", report)
        records = []
        for path in sorted(self.output_dir.rglob("*")):
            if path.is_file() and path.name != "artifact_index.json":
                records.append(
                    {
                        "path": path.relative_to(self.output_dir).as_posix(),
                        "size_bytes": path.stat().st_size,
                        "sha256": sha256_file(path),
                    }
                )
        _write_json(
            self.output_dir / "artifact_index.json",
            {
                "schema_version": "05s-artifact-index-v1",
                "artifact_count": len(records),
                "artifacts": records,
            },
        )
        return report


__all__ = [
    "EXPECTED_05R_ARCHIVE_SHA256",
    "EXPECTED_05R_FINAL_SHA256",
    "EXPECTED_05R_INDEX_SHA256",
    "MechanismStateEncoderCanary",
    "MechanismStateEncoderCanaryConfig",
    "shared_semantic_state_projection",
    "verified_temporal_representation_artifact_root",
]
