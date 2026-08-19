"""Development-only state-consistent recommit after the 05k-b diagnosis."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np

from ..hayflow_data.hines_inputs import encode_realized_synaptic_drive
from ..hayflow_eval.flowmap_metrics import write_parquet
from .hines_experiment import Progress, _write_json
from .hines_frozen_candidate_micro_rollout import rollout_voltage_metrics
from .hines_isolation_experiment import sha256_file
from .hines_layer import require_torch
from .hines_regenerative_confirmation import _verified_artifact_root
from .hines_regenerative_decoder_refit import HinesRegenerativeDecoderRefit
from .hines_regenerative_fresh_test import (
    EXPECTED_05JN_ARCHIVE_SHA256,
    EXPECTED_05JN_FINAL_SHA256,
    EXPECTED_05JN_INDEX_SHA256,
)
from .hines_trainable_topology_canary import TrainableTopologyResidualHead

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


EXPECTED_05KB_ARCHIVE_SHA256 = (
    "ec0b721f3828a8428b1347bcd1126d783104d5abf512e44a31991c56a94e4700"
)
EXPECTED_05KB_INDEX_SHA256 = (
    "9cdf563becbbb26775155836460625e2f0746f1e02cc3e469c1067e79918f4c4"
)
EXPECTED_05KB_FINAL_SHA256 = (
    "d53b41923bc13b1b34aa5100c110c11f67e0714f3f24ef27fe40cfe0493ed8e7"
)


@dataclass(frozen=True)
class HinesDevelopmentAutoregressiveRepairConfig:
    seeds: Tuple[int, ...] = (17, 29, 43)
    horizons_ms: Tuple[int, ...] = (2, 4, 8)
    gate_horizons_ms: Tuple[int, ...] = (4, 8)
    development_pair_count: int = 24
    minimum_error_reduction_vs_standard_fraction: float = 0.25
    maximum_segment_error_ratio_vs_standard: float = 1.0
    minimum_passing_seeds: int = 2
    physical_voltage_min_mv: float = -150.0
    physical_voltage_max_mv: float = 100.0
    transform_reproduction_atol: float = 1e-6

    def validate(self) -> None:
        if self.seeds != (17, 29, 43):
            raise ValueError("05k-c must retain all three frozen seeds")
        if self.horizons_ms != (2, 4, 8) or self.gate_horizons_ms != (4, 8):
            raise ValueError("05k-c horizon contract is fixed")
        if self.development_pair_count != 24:
            raise ValueError("05k-c must use all 24 independent development pairs")
        if not 0 < self.minimum_error_reduction_vs_standard_fraction < 1:
            raise ValueError("05k-c improvement threshold is invalid")
        if self.maximum_segment_error_ratio_vs_standard <= 0:
            raise ValueError("05k-c maximum-error ratio must be positive")
        if not 1 <= self.minimum_passing_seeds <= len(self.seeds):
            raise ValueError("05k-c robust seed gate is invalid")
        if self.physical_voltage_min_mv >= self.physical_voltage_max_mv:
            raise ValueError("05k-c physical voltage bounds are reversed")
        if self.transform_reproduction_atol <= 0:
            raise ValueError("05k-c transform tolerance must be positive")

    @classmethod
    def from_mapping(
        cls, values: Mapping[str, Any]
    ) -> "HinesDevelopmentAutoregressiveRepairConfig":
        payload = dict(values)
        for name in ("seeds", "horizons_ms", "gate_horizons_ms"):
            if name in payload:
                payload[name] = tuple(map(int, payload[name]))
        result = cls(**payload)
        result.validate()
        return result


def verified_failure_reassessment_artifact_root(
    source: Path, cache_dir: Path
) -> Tuple[Path, Dict[str, Any], Dict[str, Any]]:
    return _verified_artifact_root(
        Path(source),
        Path(cache_dir),
        marker_name="autoregressive_failure_reassessment_config.json",
        archive_sha256=EXPECTED_05KB_ARCHIVE_SHA256,
        index_sha256=EXPECTED_05KB_INDEX_SHA256,
        final_sha256=EXPECTED_05KB_FINAL_SHA256,
    )


class HinesDevelopmentAutoregressiveRepair(HinesRegenerativeDecoderRefit):
    """Test a fixed state-consistent recurrent commit without fresh-test access."""

    def __init__(
        self,
        *args: Any,
        development_repair_config: HinesDevelopmentAutoregressiveRepairConfig,
        artifact_05jn_source: Path,
        artifact_05kb_source: Path,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        development_repair_config.validate()
        self.development_repair = development_repair_config
        self.artifact_05jn_source = Path(artifact_05jn_source).resolve()
        self.artifact_05kb_source = Path(artifact_05kb_source).resolve()
        self.artifact_05jn_root = Path()
        self.artifact_05jn_report: Dict[str, Any] = {}
        self.artifact_05jn_contract: Dict[str, Any] = {}
        self.artifact_05kb_root = Path()
        self.artifact_05kb_report: Dict[str, Any] = {}
        self.artifact_05kb_contract: Dict[str, Any] = {}
        self._frontend_specs: List[Dict[str, Any]] = []

    def prepare_development_autoregressive_repair(self) -> Dict[str, Any]:
        root_b, report_b, contract_b = verified_failure_reassessment_artifact_root(
            self.artifact_05kb_source,
            self.output_dir.parent / ".05k_c_artifact_cache" / "05kb",
        )
        blockers = []
        if not report_b.get("valid"):
            blockers.append("05k-b artifact is invalid")
        if report_b.get("diagnosis") != "CLOSED_LOOP_STATE_DISTRIBUTION_SHIFT":
            blockers.append("05k-b did not isolate closed-loop state shift")
        if report_b.get("candidate_reinstated") or report_b.get("training_authorized"):
            blockers.append("05k-b unexpectedly reinstated or authorized a model")
        if not report_b.get("fresh_test_is_now_consumed_for_diagnosis"):
            blockers.append("05k-b did not seal the consumed fresh test")
        if report_b.get("next_step") != "05k_c_development_only_autoregressive_repair_design":
            blockers.append("05k-b did not prescribe this experiment")
        if blockers:
            raise RuntimeError(f"05k-c provenance blockers: {blockers}")
        base = self.prepare_regenerative_decoder_refit()
        root_n, report_n, contract_n = _verified_artifact_root(
            self.artifact_05jn_source,
            self.output_dir.parent / ".05k_c_artifact_cache" / "05jn",
            marker_name="regenerative_decoder_refit_config.json",
            archive_sha256=EXPECTED_05JN_ARCHIVE_SHA256,
            index_sha256=EXPECTED_05JN_INDEX_SHA256,
            final_sha256=EXPECTED_05JN_FINAL_SHA256,
        )
        if report_n.get("diagnosis") != "REFIT_PASSES_DEVELOPMENT_FRESH_TEST_GENERATION_AUTHORIZED":
            raise RuntimeError("05k-c frozen 05j-n candidate identity mismatch")
        self.artifact_05kb_root, self.artifact_05kb_report, self.artifact_05kb_contract = (
            root_b,
            report_b,
            contract_b,
        )
        self.artifact_05jn_root, self.artifact_05jn_report, self.artifact_05jn_contract = (
            root_n,
            report_n,
            contract_n,
        )
        payload = {
            "schema_version": "05k-c-development-autoregressive-repair-config-v1",
            "valid": True,
            "development_repair": asdict(self.development_repair),
            "artifact_05kb": contract_b,
            "artifact_05jn": contract_n,
            "fixed_architectural_repair": "corrected_voltage_recommit_into_local_and_global_state",
            "fresh_05jo_state_or_outcomes_loaded": False,
            "retraining_performed": False,
            "checkpoint_selection_performed": False,
            "teacher_boundary_reset_role": "descriptive_oracle_only",
            "full_training_authorized": False,
            "code_revision": self.code_revision,
        }
        _write_json(self.output_dir / "development_autoregressive_repair_config.json", payload)
        return {**base, **payload}

    def _load_frozen_decoder(self, seed: int, feature_width: int, device: Any) -> Any:
        registered = {
            int(row["seed"]): row
            for row in self.artifact_05jn_report["decoder_refit"]["runs"]
        }[int(seed)]
        checkpoint = torch.load(
            self.artifact_05jn_root / str(registered["checkpoint"]),
            map_location=device,
            weights_only=False,
        )
        if checkpoint.get("family") != "direct_tree_refit" or int(
            checkpoint.get("seed", -1)
        ) != int(seed):
            raise RuntimeError(f"05k-c checkpoint identity mismatch for seed {seed}")
        model = TrainableTopologyResidualHead(
            feature_width,
            self.layout.segment_count,
            self.topology.hidden_width,
            self.topology.segment_embedding_dim,
            self.topology.target_residual_limit_mv,
        ).to(device)
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        return model

    def _load_registered_transform(self) -> Tuple[Dict[str, Any], float]:
        with np.load(self.artifact_05jn_root / "refit_feature_transform.npz") as archive:
            loaded = {
                "surfaces": {
                    "h2_raw": {
                        "mean": archive["h2_mean"],
                        "scale": archive["h2_scale"],
                        "channel_mean": archive["h2_channel_mean"],
                        "components": archive["h2_components"],
                    },
                    "causal_raw": {
                        "mean": archive["causal_mean"],
                        "scale": archive["causal_scale"],
                        "channel_mean": archive["causal_channel_mean"],
                        "components": archive["causal_components"],
                    },
                },
                "raw_mean": archive["raw_mean"],
                "raw_scale": archive["raw_scale"],
            }
        errors = []
        for surface in ("h2_raw", "causal_raw"):
            for name in ("mean", "scale", "channel_mean", "components"):
                errors.append(
                    float(
                        np.max(
                            np.abs(
                                np.asarray(loaded["surfaces"][surface][name])
                                - np.asarray(self.refit_transform["surfaces"][surface][name])
                            )
                        )
                    )
                )
        for name in ("raw_mean", "raw_scale"):
            errors.append(
                float(np.max(np.abs(np.asarray(loaded[name]) - np.asarray(self.refit_transform[name]))))
            )
        return loaded, max(errors, default=0.0)

    def _synapse_frontend_specs(self) -> List[Dict[str, Any]]:
        if self._frontend_specs:
            return self._frontend_specs
        state_records: Dict[int, Dict[str, int]] = {}
        for position, record in enumerate(self.layout.core_records):
            if str(record.get("scope", "")) == "synapse":
                state_records.setdefault(int(record["owner_id"]), {})[
                    str(record.get("variable", record.get("name", "")))
                ] = position
        specs = []
        for synapse in self.layout.synapses:
            synapse_id = int(synapse["id"])
            variables = state_records.get(synapse_id, {})
            for component in synapse.get("components", []):
                name = str(component["name"]).upper()
                suffix = "NMDA" if "NMDA" in name else "AMPA" if "AMPA" in name else ""
                a_name = f"A_{suffix}" if suffix and f"A_{suffix}" in variables else "A"
                b_name = f"B_{suffix}" if suffix and f"B_{suffix}" in variables else "B"
                if a_name in variables and b_name in variables:
                    specs.append(
                        {
                            "a_index": int(variables[a_name]),
                            "b_index": int(variables[b_name]),
                        }
                    )
        if not specs:
            raise RuntimeError("05k-c could not resolve authentic synaptic A/B state")
        self._frontend_specs = specs
        return specs

    def _authentic_frontend_state(self, logical_index: int) -> np.ndarray:
        raw = self.confirmation_store.read_state([int(logical_index)], "t")[0]
        isolated = np.zeros_like(raw, dtype=np.float64)
        coordinates = sorted(
            {
                int(spec[name])
                for spec in self._synapse_frontend_specs()
                for name in ("a_index", "b_index")
            }
        )
        isolated[coordinates] = raw[coordinates]
        return isolated

    @staticmethod
    def _hide_teacher_state(batch: Mapping[str, Any], recurrent: Mapping[str, Any] | None) -> Dict[str, Any]:
        result = dict(batch)
        if recurrent is not None:
            for name in (
                "teacher_state_t",
                "voltage_t",
                "calcium_t",
                "synapse_state_t",
                "anchor_voltage_t",
            ):
                if name in result:
                    result[name] = torch.zeros_like(result[name])
        return result

    def _causal_frontend_batch(
        self, batch: Mapping[str, Any], logical_index: int, voltage_t: Any, device: Any
    ) -> Dict[str, Any]:
        result = dict(batch)
        drive = encode_realized_synaptic_drive(
            self.confirmation_store,
            [int(logical_index)],
            voltage_t.detach().cpu().numpy(),
            dt_ms=self.config.model.dt_ms,
            raw_state_t=self._authentic_frontend_state(logical_index)[None, :],
        )
        for name, values in drive.items():
            result[name] = torch.as_tensor(values, dtype=torch.float32, device=device)
        return result

    @staticmethod
    def _causal_surface(batch: Mapping[str, Any]) -> Any:
        return torch.cat(
            [
                batch["synaptic_features"],
                batch["synaptic_conductance_us"].unsqueeze(-1),
                batch["synaptic_source_na"].unsqueeze(-1),
                batch["somatic_current_na"].unsqueeze(-1),
            ],
            dim=-1,
        )

    def _development_windows(self, horizon: int) -> List[np.ndarray]:
        if self.confirmation_store is None:
            raise RuntimeError("05k-c independent confirmation store is unavailable")
        lookup = {
            int(value): index
            for index, value in enumerate(
                self.confirmation_store.metadata["transition_id"].tolist()
            )
        }
        windows = []
        for row in self.artifact_05ji_report["pair_rows"]:
            for key in ("low_transition_id", "high_transition_id"):
                start = lookup[int(row[key])]
                trajectory = str(self.confirmation_store.metadata["trajectory_id"][start])
                episode = self.confirmation_store.trajectory_indices[trajectory]
                position = int(np.flatnonzero(episode == start)[0])
                window = np.asarray(episode[position : position + horizon], dtype=np.int64)
                if len(window) != horizon:
                    raise RuntimeError(f"05k-c incomplete development window for {trajectory}")
                steps = self.confirmation_store.metadata["step_index"][window]
                if not np.array_equal(steps, np.arange(steps[0], steps[0] + horizon)):
                    raise RuntimeError(f"05k-c non-contiguous development window for {trajectory}")
                windows.append(window)
        if len(windows) != 2 * self.development_repair.development_pair_count:
            raise RuntimeError("05k-c development pair cardinality mismatch")
        return windows

    def _decode_voltage(
        self,
        output: Mapping[str, Any],
        batch: Mapping[str, Any],
        state_voltage: Any,
        decoder: Any,
        transform: Mapping[str, Any],
        device: Any,
    ) -> Any:
        role = {
            "h2_raw": output["boundary_features"].detach().cpu().double().numpy(),
            "causal_raw": self._causal_surface(batch).detach().cpu().double().numpy(),
            "voltage_t": state_voltage.detach().cpu().double().numpy(),
            "base": output["voltage"].detach().cpu().double().numpy(),
        }
        design = self._normalize_raw_topology(
            self._raw_topology_design(role, transform), transform
        )
        features = torch.as_tensor(design, dtype=torch.float32, device=device)
        return output["voltage"] + decoder(features)

    def _state_consistent_recommit(
        self,
        h2: Any,
        state: Mapping[str, Any],
        output: Mapping[str, Any],
        batch: Mapping[str, Any],
        corrected_voltage: Any,
    ) -> Dict[str, Any]:
        hidden = output["boundary_features"]
        transition = h2.transition_features(hidden)
        synaptic = h2.synaptic_encoder(batch["synaptic_features"])
        commit_input = torch.cat(
            [
                transition,
                (state["voltage"] / 100.0).unsqueeze(-1),
                (corrected_voltage / 100.0).unsqueeze(-1),
                output["event_local_gate"],
                synaptic,
            ],
            dim=-1,
        )
        local_next = h2.local_commit(
            commit_input.reshape(-1, commit_input.shape[-1]),
            state["local"].reshape(-1, state["local"].shape[-1]),
        ).reshape_as(state["local"])
        regional = h2._regional_pool(local_next)
        event_probability = torch.sigmoid(output["event_logits"])
        global_input = torch.cat(
            [
                local_next.mean(1),
                local_next.amax(1),
                regional.mean(1),
                torch.stack(
                    [
                        corrected_voltage.mean(1),
                        corrected_voltage.amax(1),
                        corrected_voltage.amin(1),
                        corrected_voltage.std(1),
                    ],
                    dim=-1,
                )
                / 100.0,
                event_probability,
            ],
            dim=-1,
        )
        return {
            "voltage": corrected_voltage,
            "local": local_next,
            "global": h2.global_commit(global_input, state["global"]),
            "calcium": output["calcium"],
            "synapse": output["synapse"],
        }

    def _rollout(
        self,
        h2: Any,
        decoder: Any | None,
        transform: Mapping[str, Any],
        window: np.ndarray,
        device: Any,
        mode: str,
    ) -> np.ndarray:
        recurrent = None
        trace = []
        with torch.no_grad():
            for logical_index in window:
                raw = self._batch([int(logical_index)], include_targets=False)
                original = self._torch_batch(raw, device)
                active = None if mode == "teacher_boundary_reset" else recurrent
                state_voltage = original["voltage_t"] if active is None else active["voltage"]
                batch = self._hide_teacher_state(original, active)
                batch = self._causal_frontend_batch(
                    batch, int(logical_index), state_voltage, device
                )
                state = h2.initialise(batch) if active is None else dict(active)
                output = h2.step(
                    state,
                    batch,
                    ablation="H2",
                    decode_teacher=False,
                    boundary_mode="no_event_jump",
                )
                corrected = (
                    output["voltage"]
                    if decoder is None
                    else self._decode_voltage(
                        output, batch, state_voltage, decoder, transform, device
                    )
                )
                if mode == "teacher_boundary_reset":
                    recurrent = None
                elif mode == "state_consistent_recommit" and decoder is not None:
                    recurrent = self._state_consistent_recommit(
                        h2, state, output, batch, corrected
                    )
                else:
                    recurrent = {
                        "voltage": corrected,
                        "local": output["local"],
                        "global": output["global"],
                        "calcium": output["calcium"],
                        "synapse": output["synapse"],
                    }
                trace.append(corrected[0].detach().cpu().numpy())
        return np.asarray(trace, dtype=np.float32)

    def evaluate_development_repair(self) -> Dict[str, Any]:
        require_torch()
        if self.confirmation_store is None or not self.refit_transform:
            raise RuntimeError("05k-c requires independent development roles and refit transform")
        registered_transform, transform_error = self._load_registered_transform()
        if transform_error > self.development_repair.transform_reproduction_atol:
            raise RuntimeError(f"05k-c registered transform mismatch: {transform_error}")
        max_horizon = max(self.development_repair.horizons_ms)
        windows = self._development_windows(max_horizon)
        targets = np.stack(
            [
                self.confirmation_store.read_state(window, "t_plus_1")[
                    :, : self.layout.segment_count
                ]
                for window in windows
            ]
        ).astype(np.float32)
        initial = np.stack(
            [
                self.confirmation_store.read_state([int(window[0])], "t")[
                    0, : self.layout.segment_count
                ]
                for window in windows
            ]
        ).astype(np.float32)
        persistence = np.repeat(initial[:, None, :], max_horizon, axis=1)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        h2, _ = self._load_h2_checkpoint(device)
        h2.eval()
        original_store = self.store
        self.store = self.confirmation_store
        predictions: Dict[str, np.ndarray] = {}
        try:
            progress = Progress("05k-c development H2", len(windows))
            h2_values = []
            for position, window in enumerate(windows, start=1):
                h2_values.append(
                    self._rollout(h2, None, registered_transform, window, device, "standard")
                )
                progress.update(position)
            predictions["h2"] = np.asarray(h2_values, dtype=np.float32)
            feature_width = int(self.refit_designs["fit"].shape[-1])
            total = len(self.development_repair.seeds) * 3 * len(windows)
            progress = Progress("05k-c fixed repair matrix", total)
            completed = 0
            for seed in self.development_repair.seeds:
                decoder = self._load_frozen_decoder(seed, feature_width, device)
                for mode in (
                    "standard",
                    "state_consistent_recommit",
                    "teacher_boundary_reset",
                ):
                    values = []
                    for window in windows:
                        values.append(
                            self._rollout(
                                h2,
                                decoder,
                                registered_transform,
                                window,
                                device,
                                mode,
                            )
                        )
                        completed += 1
                        progress.update(completed)
                    predictions[f"seed_{seed}__{mode}"] = np.asarray(
                        values, dtype=np.float32
                    )
                del decoder
        finally:
            self.store = original_store
            del h2
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        baselines: Dict[str, Dict[str, Any]] = {"h2": {}, "persistence": {}}
        runs: Dict[str, Dict[str, Dict[str, Any]]] = {}
        rows = []
        for horizon in self.development_repair.horizons_ms:
            key = str(horizon)
            target = targets[:, :horizon, :]
            for name, values in (("h2", predictions["h2"]), ("persistence", persistence)):
                baselines[name][key] = rollout_voltage_metrics(
                    values[:, :horizon, :],
                    target,
                    physical_min_mv=self.development_repair.physical_voltage_min_mv,
                    physical_max_mv=self.development_repair.physical_voltage_max_mv,
                )
            for seed in self.development_repair.seeds:
                seed_key = str(seed)
                runs.setdefault(seed_key, {})
                for mode in (
                    "standard",
                    "state_consistent_recommit",
                    "teacher_boundary_reset",
                ):
                    metrics = rollout_voltage_metrics(
                        predictions[f"seed_{seed}__{mode}"][:, :horizon, :],
                        target,
                        physical_min_mv=self.development_repair.physical_voltage_min_mv,
                        physical_max_mv=self.development_repair.physical_voltage_max_mv,
                    )
                    runs[seed_key].setdefault(mode, {})[key] = metrics
        seed_pass = {}
        for seed in self.development_repair.seeds:
            seed_key = str(seed)
            passed_horizons = []
            for horizon in self.development_repair.horizons_ms:
                key = str(horizon)
                standard = runs[seed_key]["standard"][key]
                repaired = runs[seed_key]["state_consistent_recommit"][key]
                reduction = 1.0 - repaired["endpoint_voltage_rmse_mv"] / max(
                    standard["endpoint_voltage_rmse_mv"], 1e-12
                )
                conditions = {
                    "minimum_error_reduction_passed": reduction
                    >= self.development_repair.minimum_error_reduction_vs_standard_fraction,
                    "maximum_error_passed": repaired["maximum_segment_error_mv"]
                    <= self.development_repair.maximum_segment_error_ratio_vs_standard
                    * standard["maximum_segment_error_mv"],
                    "physical_range_passed": repaired[
                        "physical_voltage_violation_count"
                    ]
                    == 0,
                    "numerically_finite": repaired["numerically_finite"],
                }
                horizon_passed = all(conditions.values())
                runs[seed_key]["state_consistent_recommit"][key] = {
                    **repaired,
                    "error_reduction_vs_standard_fraction": reduction,
                    "conditions": conditions,
                    "horizon_passed": horizon_passed,
                }
                rows.append(
                    {
                        "seed": int(seed),
                        "horizon_ms": int(horizon),
                        "standard_rmse_mv": standard["endpoint_voltage_rmse_mv"],
                        "repaired_rmse_mv": repaired["endpoint_voltage_rmse_mv"],
                        "error_reduction_vs_standard_fraction": reduction,
                        "standard_maximum_error_mv": standard["maximum_segment_error_mv"],
                        "repaired_maximum_error_mv": repaired["maximum_segment_error_mv"],
                        "repaired_physical_voltage_violation_count": repaired[
                            "physical_voltage_violation_count"
                        ],
                        "horizon_passed": horizon_passed,
                    }
                )
                if horizon in self.development_repair.gate_horizons_ms:
                    passed_horizons.append(horizon_passed)
            seed_pass[seed_key] = bool(all(passed_horizons))
        passing = sum(seed_pass.values())
        valid = bool(
            all(np.all(np.isfinite(value)) for value in predictions.values())
            and targets.shape
            == (
                2 * self.development_repair.development_pair_count,
                max_horizon,
                self.layout.segment_count,
            )
        )
        np.savez_compressed(
            self.output_dir / "development_repair_predictions.npz",
            target=targets,
            persistence=persistence,
            **predictions,
        )
        write_parquet(self.output_dir / "development_repair_gate.parquet", rows)
        report = {
            "schema_version": "05k-c-development-autoregressive-repair-v1",
            "valid": valid,
            "device": str(device),
            "pair_count": len(windows) // 2,
            "episode_count": len(windows),
            "horizons_ms": list(self.development_repair.horizons_ms),
            "baselines": baselines,
            "runs": runs,
            "seed_passed_gate": seed_pass,
            "passing_seed_count": passing,
            "minimum_passing_seed_count": self.development_repair.minimum_passing_seeds,
            "state_consistent_repair_supported": passing
            >= self.development_repair.minimum_passing_seeds,
            "evaluation_role": "existing_independent_development_confirmation",
            "fresh_05jo_state_or_outcomes_loaded": False,
            "retraining_performed": False,
            "checkpoint_selection_performed": False,
            "teacher_boundary_reset_role": "descriptive_oracle_only",
            "maximum_transform_reproduction_error": transform_error,
        }
        _write_json(self.output_dir / "development_autoregressive_repair.json", report)
        if not valid:
            raise RuntimeError("05k-c development repair evaluation is invalid")
        return report

    def finalize_development_repair(self, repair_report: Mapping[str, Any]) -> Dict[str, Any]:
        passed = bool(
            repair_report.get("valid")
            and repair_report.get("state_consistent_repair_supported")
        )
        report = {
            "schema_version": "05k-c-final-report-v1",
            "valid": bool(repair_report.get("valid")),
            "decision": "DEVELOPMENT_ONLY_STATE_CONSISTENT_RECOMMIT",
            "diagnosis": (
                "STATE_CONSISTENT_RECOMMIT_SUPPORTED_ON_DEVELOPMENT"
                if passed
                else "STATE_CONSISTENT_RECOMMIT_NOT_SUFFICIENT_ON_DEVELOPMENT"
            ),
            "code_revision": self.code_revision,
            "artifact_05kb": self.artifact_05kb_contract,
            "development_repair": dict(repair_report),
            "candidate_reinstated": False,
            "rollout_aware_training_canary_authorized": passed,
            "full_training_authorized": False,
            "mass_dataset_generation_authorized": False,
            "fresh_05jo_used": False,
            "future_candidate_requires_new_sealed_test": True,
            "next_step": (
                "05l_state_consistent_rollout_training_canary"
                if passed
                else "05k_d_architecture_reassessment"
            ),
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
                "schema_version": "05k-c-artifact-index-v1",
                "artifact_count": len(records),
                "artifacts": records,
            },
        )
        return report


__all__ = [
    "EXPECTED_05KB_ARCHIVE_SHA256",
    "EXPECTED_05KB_FINAL_SHA256",
    "EXPECTED_05KB_INDEX_SHA256",
    "HinesDevelopmentAutoregressiveRepair",
    "HinesDevelopmentAutoregressiveRepairConfig",
    "verified_failure_reassessment_artifact_root",
]
