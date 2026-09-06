"""Validated configuration for staged GIADA data scaling on RunPod."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Mapping


STAGE_TRANSITIONS = {
    "s0": 29_880,
    "s1": 600_000,
    # Exploratory input-support pilot only.  Its validation split is a
    # development confirmation split and must never be reported as a final
    # paper test.
    "s1b_pilot": 96_000,
    # Small causal pilot for the primary GIADA hybrid methodology.  It is
    # intentionally separate from the random-support S1/S1b controls.
    "s1c_hybrid_pilot": 15_360,
    # Prospective repair of the two S1c protocol-transcription failures.  It
    # leaves all successful S1c arms immutable and reruns only the somatic/BAP
    # boundary matrix from the historical validated parameters.
    "s1d_protocol_repair_pilot": 11_520,
    # Prospective production-scale confirmation of the now validated hybrid
    # data recipe.  Long stochastic trajectories and short targeted episodes
    # remain separate physical corpora and are joined only by a verified
    # composite manifest.
    "s1e_hybrid_background": 360_000,
    "s1e_hybrid_targeted": 240_000,
    # Six-fold prospective expansion of the validated S1e hybrid recipe.
    # The two components remain physically separate because their natural
    # trajectory lengths differ (6 s background versus 80 ms targeted).
    "s2_hybrid_background": 2_160_000,
    "s2_hybrid_targeted": 1_440_000,
    "s2": 3_600_000,
    "s3": 28_800_000,
    "s4": 230_400_000,
}


@dataclass(frozen=True)
class ScaleConfig:
    """One immutable generation stage.

    ``soma_paper`` is the storage profile used for the information-matched
    one-step claim.  ``spatial_probe`` is intentionally capped below paper
    parity because it stores several segment-local examples per millisecond.
    """

    stage: str
    target_transitions: int
    trajectory_duration_ms: int = 6000
    trajectories_per_shard: int = 4
    storage_profile: str = "soma_paper"
    sampled_segments_per_transition: int = 1
    root_seed: int = 9_100_001
    validation_trajectory_fraction: float = 10.0 / 300.0
    compression: str = "lzf"
    chunk_transitions: int = 256
    progress_interval_s: float = 30.0
    purpose: str = "paper_scale_confirmation"
    input_protocols: tuple[str, ...] = ("neuronio_nmda_ergodic_v1",)

    def validate(self) -> None:
        if self.stage not in STAGE_TRANSITIONS:
            raise ValueError(f"unsupported scale stage {self.stage!r}")
        if int(self.target_transitions) != STAGE_TRANSITIONS[self.stage]:
            raise ValueError(
                f"{self.stage} requires exactly {STAGE_TRANSITIONS[self.stage]:,} "
                "transitions"
            )
        if self.trajectory_duration_ms <= 0:
            raise ValueError("trajectory_duration_ms must be positive")
        if self.target_transitions % self.trajectory_duration_ms:
            raise ValueError("target transitions must contain complete trajectories")
        if self.trajectories_per_shard <= 0:
            raise ValueError("trajectories_per_shard must be positive")
        if self.storage_profile not in {"soma_paper", "spatial_probe"}:
            raise ValueError("unknown storage profile")
        if self.storage_profile == "soma_paper" and self.sampled_segments_per_transition != 1:
            raise ValueError("soma_paper stores exactly the canonical soma segment")
        if self.sampled_segments_per_transition <= 0:
            raise ValueError("sampled_segments_per_transition must be positive")
        if self.storage_profile == "spatial_probe" and self.stage in {"s3", "s4"}:
            raise ValueError(
                "spatial_probe is deliberately capped at s2; use soma_paper for s3/s4"
            )
        if not 0.0 < self.validation_trajectory_fraction < 0.5:
            if not (
                self.purpose in {
                    "input_support_pilot",
                    "giada_hybrid_pilot",
                    "giada_protocol_repair_pilot",
                }
                and self.validation_trajectory_fraction == 0.5
            ):
                raise ValueError(
                    "validation fraction must lie in (0, 0.5), except for a "
                    "paired input-support pilot"
                )
        if self.purpose not in {
            "paper_scale_confirmation",
            "input_support_pilot",
            "giada_hybrid_pilot",
            "giada_protocol_repair_pilot",
            "giada_hybrid_production_background",
            "giada_hybrid_production_targeted",
        }:
            raise ValueError("unknown generation purpose")
        if not self.input_protocols or len(set(self.input_protocols)) != len(
            self.input_protocols
        ):
            raise ValueError("input protocols must be non-empty and unique")
        if self.purpose == "input_support_pilot":
            if self.stage != "s1b_pilot":
                raise ValueError("input-support pilot requires stage s1b_pilot")
            if self.trajectory_count % (2 * len(self.input_protocols)):
                raise ValueError(
                    "input-support pilot requires paired discovery/confirmation "
                    "replicates for every protocol"
                )
        if self.purpose == "giada_hybrid_pilot":
            from .hybrid_inputs import HYBRID_PROTOCOLS

            if self.stage != "s1c_hybrid_pilot":
                raise ValueError("GIADA hybrid pilot requires stage s1c_hybrid_pilot")
            if tuple(self.input_protocols) != HYBRID_PROTOCOLS:
                raise ValueError(
                    "GIADA hybrid pilot requires the preregistered protocol registry"
                )
            if self.validation_trajectory_fraction != 0.5:
                raise ValueError(
                    "GIADA hybrid pilot requires balanced train/validation groups"
                )
            if self.trajectory_count % (2 * len(HYBRID_PROTOCOLS)):
                raise ValueError("hybrid trajectory count must contain complete paired replicates")
        if self.purpose == "giada_protocol_repair_pilot":
            from .hybrid_inputs import PROTOCOL_REPAIR_PROTOCOLS

            if self.stage != "s1d_protocol_repair_pilot":
                raise ValueError(
                    "GIADA protocol-repair pilot requires stage "
                    "s1d_protocol_repair_pilot"
                )
            if tuple(self.input_protocols) != PROTOCOL_REPAIR_PROTOCOLS:
                raise ValueError(
                    "GIADA protocol-repair pilot requires the preregistered "
                    "repair registry"
                )
            if self.validation_trajectory_fraction != 0.5:
                raise ValueError(
                    "GIADA protocol-repair pilot requires balanced "
                    "train/validation groups"
                )
            if self.trajectory_count % (2 * len(PROTOCOL_REPAIR_PROTOCOLS)):
                raise ValueError(
                    "repair trajectory count must contain complete paired replicates"
                )
        if self.purpose == "giada_hybrid_production_background":
            from .hybrid_inputs import PRODUCTION_BACKGROUND_PROTOCOLS

            if self.stage not in {
                "s1e_hybrid_background",
                "s2_hybrid_background",
            }:
                raise ValueError("hybrid production background requires an S1e/S2 component stage")
            if tuple(self.input_protocols) != PRODUCTION_BACKGROUND_PROTOCOLS:
                raise ValueError("hybrid production background protocol registry changed")
            if self.storage_profile != "soma_paper" or self.trajectory_duration_ms != 6000:
                raise ValueError("hybrid production background requires 6000 ms soma trajectories")
            if self.validation_trajectory_fraction != 0.2:
                raise ValueError("hybrid production uses a fixed 80/20 trajectory split")
        if self.purpose == "giada_hybrid_production_targeted":
            from .hybrid_inputs import PRODUCTION_TARGET_PROTOCOLS

            if self.stage not in {
                "s1e_hybrid_targeted",
                "s2_hybrid_targeted",
            }:
                raise ValueError("hybrid production targeted requires an S1e/S2 component stage")
            if tuple(self.input_protocols) != PRODUCTION_TARGET_PROTOCOLS:
                raise ValueError("hybrid production targeted protocol registry changed")
            if self.storage_profile != "soma_paper" or self.trajectory_duration_ms != 80:
                raise ValueError("hybrid production targeted requires 80 ms soma episodes")
            if self.validation_trajectory_fraction != 0.2:
                raise ValueError("hybrid production uses a fixed 80/20 trajectory split")
            per_protocol, remainder = divmod(
                self.trajectory_count, len(PRODUCTION_TARGET_PROTOCOLS)
            )
            if remainder or per_protocol <= 1:
                raise ValueError("hybrid targeted plan must balance every protocol")
        if self.compression not in {"lzf", "gzip", "none"}:
            raise ValueError("compression must be lzf, gzip, or none")
        if self.chunk_transitions <= 0 or self.progress_interval_s <= 0:
            raise ValueError("chunk/progress settings must be positive")

    @property
    def trajectory_count(self) -> int:
        return self.target_transitions // self.trajectory_duration_ms

    @property
    def shard_count(self) -> int:
        count, remainder = divmod(self.trajectory_count, self.trajectories_per_shard)
        return count + int(bool(remainder))

    def to_dict(self) -> Dict[str, Any]:
        self.validate()
        return asdict(self)

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "ScaleConfig":
        payload = dict(values)
        if "stage" not in payload:
            raise ValueError("scale config requires stage")
        payload.setdefault("target_transitions", STAGE_TRANSITIONS[str(payload["stage"])])
        if "input_protocols" in payload:
            payload["input_protocols"] = tuple(map(str, payload["input_protocols"]))
        result = cls(**payload)
        result.validate()
        return result


def load_scale_config(path: Path) -> ScaleConfig:
    try:
        import yaml
    except ImportError as error:  # pragma: no cover
        raise RuntimeError("loading RunPod configs requires PyYAML") from error
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    values = payload.get("giada_runpod_scale", payload)
    return ScaleConfig.from_mapping(values)
