"""Parameter-matched nonlinear probes after the negative 05n linear audit."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import math
from pathlib import Path
import random
import time
from typing import Any, Dict, Mapping, Tuple

import numpy as np

from src.hayflow_data.composite_flowmap import CompositeFlowmapBundle

from .graph_state_contract_reassessment import (
    GraphStateContractReassessment,
    GraphStateContractReassessmentConfig,
    _episode_transition_starts,
    _metrics,
)
from .hines_experiment import _write_json
from .hines_isolation_experiment import sha256_file
from .hines_regenerative_confirmation import _verified_artifact_root
from .rollout_aware_architecture_canary import CAUSAL_DRIVE_FEATURES
from .topology_controlled_recurrence_expansion import (
    TopologyControlledRecurrenceConfig,
    expanded_train_episode_roles,
)


EXPECTED_05N_ARCHIVE_SHA256 = (
    "495c4419d2205c9a3e4c58fb0cf67da534d9ada38e9e28b393ba8dbc97b80316"
)
EXPECTED_05N_INDEX_SHA256 = (
    "975c8b0741d8639d19095447d1b06c06795020e478b21a036e5e3aadc60bd627"
)
EXPECTED_05N_FINAL_SHA256 = (
    "2460f61579d972464159dfe60f7d73e02e9ce0c9245138dc750253655db9ab30"
)


def verified_graph_state_reassessment_artifact_root(
    source: Path, cache_dir: Path
) -> Tuple[Path, Dict[str, Any], Dict[str, Any]]:
    return _verified_artifact_root(
        Path(source),
        Path(cache_dir),
        marker_name="graph_state_contract_reassessment_config.json",
        archive_sha256=EXPECTED_05N_ARCHIVE_SHA256,
        index_sha256=EXPECTED_05N_INDEX_SHA256,
        final_sha256=EXPECTED_05N_FINAL_SHA256,
    )


@dataclass(frozen=True)
class NonlinearObservabilityReassessmentConfig(
    GraphStateContractReassessmentConfig
):
    state_sketch_dim: int = 64
    seeds: Tuple[int, ...] = (17, 29, 43)
    hidden_width: int = 64
    residual_blocks: int = 2
    epochs: int = 50
    batch_size: int = 4096
    learning_rate: float = 1.0e-3
    weight_decay: float = 1.0e-5
    gradient_clip_norm: float = 1.0
    evaluation_interval: int = 5
    progress_interval: int = 10
    regenerative_training_weight: float = 3.0
    nonregenerative_per_regenerative: int = 3
    maximum_training_rows: int = 150000
    minimum_passing_seed_count: int = 2

    def validate(self) -> None:
        super().validate()
        if self.seeds != (17, 29, 43):
            raise ValueError("05o uses the three registered probe seeds")
        if min(
            self.hidden_width,
            self.residual_blocks,
            self.epochs,
            self.batch_size,
            self.evaluation_interval,
            self.progress_interval,
            self.nonregenerative_per_regenerative,
            self.maximum_training_rows,
        ) <= 0:
            raise ValueError("05o positive integer configuration is invalid")
        if not 0 < self.learning_rate < 1 or self.weight_decay < 0:
            raise ValueError("05o optimizer configuration is invalid")
        if self.regenerative_training_weight < 1:
            raise ValueError("05o regenerative weight cannot be below one")
        if self.minimum_passing_seed_count not in {2, 3}:
            raise ValueError("05o robust seed gate is invalid")

    @classmethod
    def from_mapping(
        cls, values: Mapping[str, Any]
    ) -> "NonlinearObservabilityReassessmentConfig":
        payload = dict(values)
        for name in ("ridge_lambdas", "seeds"):
            if name in payload:
                element = float if name == "ridge_lambdas" else int
                payload[name] = tuple(map(element, payload[name]))
        result = cls(**payload)
        result.validate()
        return result


try:  # Optional in local data-contract environments.
    import torch
    from torch import nn
    import torch.nn.functional as torch_functional
except ImportError:  # pragma: no cover
    torch = None
    nn = None
    torch_functional = None


if nn is not None:

    class FixedWidthObservabilityProbe(nn.Module):
        def __init__(
            self, input_width: int, hidden_width: int, residual_blocks: int
        ) -> None:
            super().__init__()
            self.input = nn.Sequential(
                nn.Linear(input_width, hidden_width),
                nn.SiLU(),
                nn.LayerNorm(hidden_width),
            )
            self.blocks = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(hidden_width, 2 * hidden_width),
                        nn.SiLU(),
                        nn.Linear(2 * hidden_width, hidden_width),
                    )
                    for _ in range(residual_blocks)
                ]
            )
            self.output = nn.Linear(hidden_width, 1)

        def forward(self, values: Any) -> Any:
            hidden = self.input(values)
            for block in self.blocks:
                hidden = hidden + block(hidden)
            return self.output(hidden).squeeze(-1)

else:  # pragma: no cover

    class FixedWidthObservabilityProbe:
        def __init__(self, *_: Any, **__: Any) -> None:
            raise RuntimeError("05o requires PyTorch")


class NonlinearObservabilityReassessment(GraphStateContractReassessment):
    def __init__(
        self,
        bundle: CompositeFlowmapBundle,
        output_dir: Path,
        config: NonlinearObservabilityReassessmentConfig,
        artifact_05n_source: Path,
        *,
        code_revision: str,
    ) -> None:
        super().__init__(
            bundle,
            output_dir,
            config,
            artifact_05n_source,
            code_revision=code_revision,
        )
        self.artifact_05n_source = Path(artifact_05n_source).resolve()

    @property
    def nonlinear_config(self) -> NonlinearObservabilityReassessmentConfig:
        return self.config  # type: ignore[return-value]

    def prepare(self) -> Dict[str, Any]:
        _, final, contract = verified_graph_state_reassessment_artifact_root(
            self.artifact_05n_source,
            self.output_dir.parent / ".05o_artifact_cache" / "05n",
        )
        blockers = []
        if not final.get("valid") or not final.get("decision_grade"):
            blockers.append("05n artifact is not decision-grade")
        if final.get("diagnosis") != "NO_LINEAR_GRAPH_OR_STATE_CONTRACT_SIGNAL":
            blockers.append("05n diagnosis does not authorize 05o")
        if final.get("next_step") != "05o_nonlinear_observability_reassessment":
            blockers.append("05n next step is not 05o")
        if final.get("graph_information_signal") or final.get(
            "initial_state_information_signal"
        ):
            blockers.append("05n unexpectedly reported a linear signal")
        mapping = self.semantic_encoder.mapping_report
        if int(mapping.get("unmapped_coordinate_count", -1)) != 0:
            blockers.append("05o NetCon semantic mapping is incomplete")
        if (
            len(self.semantic_encoder.recency_indices)
            != self.domain_config.expected_tsyn_coordinate_count
        ):
            blockers.append("05o bounded recency coordinate count changed")
        if (
            len(self.semantic_encoder.trace_indices)
            != self.domain_config.expected_dynamic_trace_coordinate_count
        ):
            blockers.append("05o synaptic trace coordinate count changed")
        self.artifact_contract = contract
        role_config = TopologyControlledRecurrenceConfig()
        self.roles, expansion = expanded_train_episode_roles(
            self.store.episode_rows, config=role_config
        )
        self.indices = _episode_transition_starts(self.store, self.roles)
        role_sets = {name: set(map(int, values)) for name, values in self.indices.items()}
        overlap = {
            f"{left}__{right}": sorted(role_sets[left] & role_sets[right])
            for position, left in enumerate(sorted(role_sets))
            for right in sorted(role_sets)[position + 1 :]
            if role_sets[left] & role_sets[right]
        }
        if overlap:
            blockers.append("05o transition roles overlap")
        base_width = (
            1
            + len(CAUSAL_DRIVE_FEATURES)
            + int(self.store.layout.segment_static.shape[1])
        )
        input_width = base_width + 4 + self.config.state_sketch_dim
        if torch is None:
            blockers.append("PyTorch is unavailable")
            parameter_count = 0
        else:
            probe = FixedWidthObservabilityProbe(
                input_width,
                self.nonlinear_config.hidden_width,
                self.nonlinear_config.residual_blocks,
            )
            parameter_count = int(sum(value.numel() for value in probe.parameters()))
        report = {
            "schema_version": "05o-preflight-v1",
            "valid": not blockers,
            "blockers": blockers,
            "artifact_05n": contract,
            "dataset_fingerprint": self.bundle.fingerprint,
            "source_split_used": "train_only",
            "validation_or_test_loaded": False,
            "fresh_test_loaded": False,
            "role_episode_counts": {
                name: len(rows) for name, rows in self.roles.items()
            },
            "role_transition_counts": {
                name: len(rows) for name, rows in self.indices.items()
            },
            "transition_overlap": overlap,
            "contracts": self.CONTRACT_FEATURES,
            "fixed_input_width": input_width,
            "parameter_count_per_contract": parameter_count,
            "parameter_matched": True,
            "paired_initialization_by_seed": True,
            "state_sketch_dim": self.config.state_sketch_dim,
            "netcon_semantic_mapping": {
                name: mapping[name]
                for name in (
                    "mapped_coordinate_count",
                    "unmapped_coordinate_count",
                    "class_coordinate_counts",
                    "semantic_coordinate_counts",
                )
            },
            "bounded_recency_coordinate_count": int(
                len(self.semantic_encoder.recency_indices)
            ),
            "teacher_future_state_used_as_input": False,
            "support_expansion": expansion,
        }
        self.prepare_report = report
        _write_json(self.output_dir / "preflight_report.json", report)
        _write_json(
            self.output_dir / "nonlinear_observability_reassessment_config.json",
            {
                "schema_version": "05o-config-v1",
                "config": asdict(self.nonlinear_config),
                "artifact_05n": contract,
                "dataset_fingerprint": self.bundle.fingerprint,
                "code_revision": self.code_revision,
            },
        )
        if blockers:
            raise RuntimeError(f"05o preflight failed: {blockers}")
        return report

    @staticmethod
    def _component_statistics(
        fit: Mapping[str, np.ndarray]
    ) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        result = {}
        for name in ("base", "graph", "state"):
            center = fit[name].mean(axis=0)
            scale = fit[name].std(axis=0)
            result[name] = (center, np.where(scale > 1e-8, scale, 1.0))
        return result

    def _fixed_matrix(
        self,
        arrays: Mapping[str, np.ndarray],
        components: Tuple[str, ...],
        statistics: Mapping[str, Tuple[np.ndarray, np.ndarray]],
    ) -> np.ndarray:
        rows = len(arrays["target"])
        values = []
        for name in ("base", "graph", "state"):
            if name in components:
                center, scale = statistics[name]
                values.append(((arrays[name] - center) / scale).astype(np.float32))
            else:
                values.append(
                    np.zeros((rows, arrays[name].shape[1]), dtype=np.float32)
                )
        return np.concatenate(values, axis=1)

    def _training_rows(self, regenerative: np.ndarray, seed: int) -> np.ndarray:
        regenerative_rows = np.flatnonzero(regenerative)
        ordinary_rows = np.flatnonzero(~regenerative)
        if not len(regenerative_rows):
            raise RuntimeError("05o fit role has no regenerative rows")
        rng = np.random.default_rng(int(seed) + 905001)
        ordinary_count = min(
            len(ordinary_rows),
            self.nonlinear_config.nonregenerative_per_regenerative
            * len(regenerative_rows),
        )
        ordinary = rng.choice(ordinary_rows, size=ordinary_count, replace=False)
        rows = np.concatenate([regenerative_rows, ordinary])
        if len(rows) > self.nonlinear_config.maximum_training_rows:
            keep_regenerative = min(
                len(regenerative_rows),
                self.nonlinear_config.maximum_training_rows // 2,
            )
            selected_regenerative = rng.choice(
                regenerative_rows, size=keep_regenerative, replace=False
            )
            remaining = self.nonlinear_config.maximum_training_rows - keep_regenerative
            selected_ordinary = rng.choice(
                ordinary_rows, size=min(remaining, len(ordinary_rows)), replace=False
            )
            rows = np.concatenate([selected_regenerative, selected_ordinary])
        rng.shuffle(rows)
        return rows.astype(np.int64)

    @staticmethod
    def _model_state_sha256(model: Any) -> str:
        digest = hashlib.sha256()
        for name, value in sorted(model.state_dict().items()):
            digest.update(name.encode("utf-8"))
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
        return digest.hexdigest()

    @staticmethod
    def _torch_metrics(
        model: Any,
        features: Any,
        target_center: float,
        target_scale: float,
        target: np.ndarray,
        regenerative: np.ndarray,
    ) -> Dict[str, Any]:
        model.eval()
        values = []
        with torch.no_grad():
            for start in range(0, len(features), 16384):
                prediction = model(features[start : start + 16384])
                values.append(prediction.detach().cpu().numpy())
        prediction = np.concatenate(values) * target_scale + target_center
        return _metrics(prediction, target, regenerative)

    def _train_probe(
        self,
        contract: str,
        seed: int,
        fit_x: np.ndarray,
        calibration_x: np.ndarray,
        fit: Mapping[str, np.ndarray],
        calibration: Mapping[str, np.ndarray],
        device: Any,
    ) -> Tuple[Any, Dict[str, Any]]:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.use_deterministic_algorithms(True)
        model = FixedWidthObservabilityProbe(
            fit_x.shape[1],
            self.nonlinear_config.hidden_width,
            self.nonlinear_config.residual_blocks,
        ).to(device)
        initial_state_sha256 = self._model_state_sha256(model)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.nonlinear_config.learning_rate,
            weight_decay=self.nonlinear_config.weight_decay,
        )
        rows = self._training_rows(fit["regenerative"], seed)
        target_center = float(fit["target"][rows].mean())
        target_scale = float(fit["target"][rows].std())
        target_scale = max(target_scale, 1e-6)
        fit_tensor = torch.as_tensor(fit_x, dtype=torch.float32, device=device)
        calibration_tensor = torch.as_tensor(
            calibration_x, dtype=torch.float32, device=device
        )
        target_tensor = torch.as_tensor(
            (fit["target"] - target_center) / target_scale,
            dtype=torch.float32,
            device=device,
        )
        regenerative_tensor = torch.as_tensor(
            fit["regenerative"], dtype=torch.bool, device=device
        )
        rng = np.random.default_rng(seed + 905002)
        best = None
        history = []
        started = time.monotonic()
        for epoch in range(self.nonlinear_config.epochs):
            order = rows[rng.permutation(len(rows))]
            losses = []
            model.train()
            for start in range(0, len(order), self.nonlinear_config.batch_size):
                position = torch.as_tensor(
                    order[start : start + self.nonlinear_config.batch_size],
                    dtype=torch.long,
                    device=device,
                )
                optimizer.zero_grad(set_to_none=True)
                prediction = model(fit_tensor.index_select(0, position))
                target = target_tensor.index_select(0, position)
                per_row = torch_functional.smooth_l1_loss(
                    prediction, target, reduction="none"
                )
                weight = torch.where(
                    regenerative_tensor.index_select(0, position),
                    per_row.new_tensor(
                        self.nonlinear_config.regenerative_training_weight
                    ),
                    per_row.new_tensor(1.0),
                )
                loss = torch.mean(per_row * weight)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), self.nonlinear_config.gradient_clip_norm
                )
                optimizer.step()
                losses.append(float(loss.detach().cpu()))
            evaluate = (
                epoch == 0
                or (epoch + 1) % self.nonlinear_config.evaluation_interval == 0
                or epoch + 1 == self.nonlinear_config.epochs
            )
            row = {"epoch": epoch + 1, "fit_loss": float(np.mean(losses))}
            if evaluate:
                metric = self._torch_metrics(
                    model,
                    calibration_tensor,
                    target_center,
                    target_scale,
                    calibration["target"],
                    calibration["regenerative"],
                )
                score = metric["rmse_mv"] + self.config.calibration_regenerative_weight * metric["regenerative_rmse_mv"]
                row.update(
                    calibration_rmse_mv=metric["rmse_mv"],
                    calibration_regenerative_rmse_mv=metric[
                        "regenerative_rmse_mv"
                    ],
                    calibration_selection_score=score,
                )
                if best is None or score < best[0]:
                    best = (
                        score,
                        epoch + 1,
                        {
                            name: value.detach().cpu().clone()
                            for name, value in model.state_dict().items()
                        },
                        metric,
                    )
            history.append(row)
            if (
                epoch == 0
                or (epoch + 1) % self.nonlinear_config.progress_interval == 0
                or epoch + 1 == self.nonlinear_config.epochs
            ):
                elapsed = time.monotonic() - started
                eta = elapsed / (epoch + 1) * (
                    self.nonlinear_config.epochs - epoch - 1
                )
                print(
                    f"[HayFlow 05o][{contract} seed={seed}] "
                    f"{epoch + 1}/{self.nonlinear_config.epochs} "
                    f"ETA {eta / 60:.1f} min loss={row['fit_loss']:.4g}",
                    flush=True,
                )
        if best is None:
            raise RuntimeError("05o produced no calibration checkpoint")
        model.load_state_dict(best[2])
        return model, {
            "seed": seed,
            "best_epoch": best[1],
            "calibration": best[3],
            "target_center_mv": target_center,
            "target_scale_mv": target_scale,
            "training_row_count": int(len(rows)),
            "training_regenerative_row_count": int(
                fit["regenerative"][rows].sum()
            ),
            "training_rows_sha256": hashlib.sha256(
                rows.astype("<i8", copy=False).tobytes()
            ).hexdigest(),
            "initial_state_sha256": initial_state_sha256,
            "history": history,
        }

    def run(self) -> Dict[str, Any]:
        if not self.prepare_report:
            self.prepare()
        if torch is None:
            raise RuntimeError("05o requires PyTorch")
        self._fit_normalizer()
        arrays = {}
        for position, role in enumerate(("fit", "calibration", "development"), 1):
            arrays[role] = self._role_arrays(role)
            print(f"[HayFlow 05o][features] {position}/3 role={role}", flush=True)
        statistics = self._component_statistics(arrays["fit"])
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint_dir = self.output_dir / "checkpoints"
        checkpoint_dir.mkdir()
        runs: Dict[str, Dict[str, Any]] = {
            name: {} for name in self.CONTRACT_FEATURES
        }
        for contract, raw_components in self.CONTRACT_FEATURES.items():
            components = tuple(raw_components)
            fit_x = self._fixed_matrix(arrays["fit"], components, statistics)
            calibration_x = self._fixed_matrix(
                arrays["calibration"], components, statistics
            )
            development_x = self._fixed_matrix(
                arrays["development"], components, statistics
            )
            development_tensor = torch.as_tensor(
                development_x, dtype=torch.float32, device=device
            )
            for seed in self.nonlinear_config.seeds:
                model, row = self._train_probe(
                    contract,
                    int(seed),
                    fit_x,
                    calibration_x,
                    arrays["fit"],
                    arrays["calibration"],
                    device,
                )
                development = self._torch_metrics(
                    model,
                    development_tensor,
                    row["target_center_mv"],
                    row["target_scale_mv"],
                    arrays["development"]["target"],
                    arrays["development"]["regenerative"],
                )
                checkpoint = checkpoint_dir / f"{contract}-seed{seed}.pt"
                torch.save(
                    {
                        "state_dict": {
                            name: value.detach().cpu()
                            for name, value in model.state_dict().items()
                        },
                        "contract": contract,
                        "seed": int(seed),
                        "target_center_mv": row["target_center_mv"],
                        "target_scale_mv": row["target_scale_mv"],
                        "dataset_fingerprint": self.bundle.fingerprint,
                        "code_revision": self.code_revision,
                    },
                    checkpoint,
                )
                row.update(
                    development=development,
                    checkpoint=checkpoint.relative_to(self.output_dir).as_posix(),
                    checkpoint_sha256=sha256_file(checkpoint),
                )
                runs[contract][str(seed)] = row
        paired_initialization = {
            str(seed): len({
                runs[contract][str(seed)]["initial_state_sha256"]
                for contract in self.CONTRACT_FEATURES
            }) == 1
            for seed in self.nonlinear_config.seeds
        }
        paired_training_rows = {
            str(seed): len({
                runs[contract][str(seed)]["training_rows_sha256"]
                for contract in self.CONTRACT_FEATURES
            }) == 1
            for seed in self.nonlinear_config.seeds
        }
        comparisons = {
            "graph_given_voltage": (
                "voltage_input_axial_graph", "voltage_input_local"
            ),
            "graph_given_rich_state": (
                "rich_state_axial_graph", "rich_state_local"
            ),
            "state_given_local": ("rich_state_local", "voltage_input_local"),
            "state_given_graph": (
                "rich_state_axial_graph", "voltage_input_axial_graph"
            ),
        }
        gains = {}
        for name, (candidate, baseline) in comparisons.items():
            rows = {}
            for seed in map(str, self.nonlinear_config.seeds):
                rows[seed] = {
                    metric: 1.0
                    - runs[candidate][seed]["development"][metric]
                    / max(runs[baseline][seed]["development"][metric], 1e-12)
                    for metric in ("rmse_mv", "regenerative_rmse_mv")
                }
            gains[name] = {
                "by_seed": rows,
                "median_rmse_gain_fraction": float(
                    np.median([row["rmse_mv"] for row in rows.values()])
                ),
                "median_regenerative_gain_fraction": float(
                    np.median(
                        [row["regenerative_rmse_mv"] for row in rows.values()]
                    )
                ),
                "positive_win_count": sum(
                    row["rmse_mv"] > 0 for row in rows.values()
                ),
            }
        def signal(names: Tuple[str, str]) -> bool:
            return bool(
                all(
                    gains[name]["median_rmse_gain_fraction"]
                    >= self.config.material_gain_fraction
                    and gains[name]["median_regenerative_gain_fraction"]
                    >= -self.config.noninferiority_margin_fraction
                    and gains[name]["positive_win_count"]
                    >= self.nonlinear_config.minimum_passing_seed_count
                    for name in names
                )
            )
        graph_signal = signal(("graph_given_voltage", "graph_given_rich_state"))
        state_signal = signal(("state_given_local", "state_given_graph"))
        if graph_signal and state_signal:
            diagnosis = "NONLINEAR_JOINT_GRAPH_AND_STATE_SIGNAL"
            next_step = "05p_axial_graphgru_rich_state_micro_canary"
        elif state_signal:
            diagnosis = "NONLINEAR_INITIAL_STATE_INFORMATION_SIGNAL"
            next_step = "05p_rich_state_recurrent_micro_canary"
        elif graph_signal:
            diagnosis = "NONLINEAR_AXIAL_GRAPH_INFORMATION_SIGNAL"
            next_step = "05p_axial_graphgru_micro_canary"
        else:
            diagnosis = "NO_NONLINEAR_LOCAL_OBSERVABILITY_SIGNAL"
            next_step = "05p_temporal_state_observability_reassessment"
        report = {
            "schema_version": "05o-final-report-v1",
            "valid": bool(
                all(
                    row["development"]["finite"]
                    for family in runs.values()
                    for row in family.values()
                )
                and all(paired_initialization.values())
                and all(paired_training_rows.values())
            ),
            "decision_grade": True,
            "diagnosis": diagnosis,
            "code_revision": self.code_revision,
            "device": str(device),
            "artifact_05n": self.artifact_contract,
            "dataset_fingerprint": self.bundle.fingerprint,
            "preflight": self.prepare_report,
            "runs": runs,
            "paired_initialization_verified_by_seed": paired_initialization,
            "paired_training_rows_verified_by_seed": paired_training_rows,
            "deterministic_algorithms_enabled": True,
            "paired_incremental_gains": gains,
            "graph_information_signal": graph_signal,
            "initial_state_information_signal": state_signal,
            "interpretation_scope": "nonlinear_one_step_development_probe_only",
            "teacher_future_state_used_as_input": False,
            "validation_or_test_loaded": False,
            "fresh_test_loaded": False,
            "recurrent_training_authorized": False,
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
        _write_json(self.output_dir / "artifact_index.json", {
            "schema_version": "05o-artifact-index-v1",
            "artifact_count": len(records),
            "artifacts": records,
        })
        return report


__all__ = [
    "EXPECTED_05N_ARCHIVE_SHA256",
    "EXPECTED_05N_FINAL_SHA256",
    "EXPECTED_05N_INDEX_SHA256",
    "FixedWidthObservabilityProbe",
    "NonlinearObservabilityReassessment",
    "NonlinearObservabilityReassessmentConfig",
    "verified_graph_state_reassessment_artifact_root",
]
