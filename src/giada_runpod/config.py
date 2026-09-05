"""Validated configuration for staged GIADA data scaling on RunPod."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Mapping


STAGE_TRANSITIONS = {
    "s0": 29_880,
    "s1": 600_000,
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
            raise ValueError("validation fraction must lie in (0, 0.5)")
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
