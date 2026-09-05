"""Deterministic, resumable shard plans for CPU teacher generation."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping

from .config import ScaleConfig


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


@dataclass(frozen=True)
class TrajectoryPlan:
    trajectory_id: str
    trajectory_index: int
    seed: int
    split: str
    duration_ms: int
    protocol: str = "neuronio_nmda_ergodic_v1"


@dataclass(frozen=True)
class ShardPlan:
    shard_id: str
    shard_index: int
    trajectories: tuple[TrajectoryPlan, ...]
    expected_transition_count: int
    plan_sha256: str

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result["trajectories"] = [asdict(row) for row in self.trajectories]
        return result


def _split_for(index: int, count: int, validation_fraction: float) -> str:
    # Match the original 290/10 train/validation proportion while distributing
    # validation trajectories throughout a large run instead of placing them
    # in one contiguous tail.
    validation_count = max(1, int(round(count * validation_fraction)))
    digest = hashlib.sha256(f"giada-paper-split|{index}".encode()).digest()
    ranks = int.from_bytes(digest[:8], "big")
    # The exact lowest-rank set is computed by build_shard_plan; this helper is
    # retained only for readable type intent.
    return "validation" if ranks < validation_count else "train"


def build_shard_plan(config: ScaleConfig) -> List[ShardPlan]:
    config.validate()
    count = config.trajectory_count
    validation_count = max(1, int(round(count * config.validation_trajectory_fraction)))
    ranked = sorted(
        range(count),
        key=lambda index: hashlib.sha256(
            f"giada-paper-split|{config.root_seed}|{index}".encode()
        ).digest(),
    )
    validation = set(ranked[:validation_count])
    trajectories = [
        TrajectoryPlan(
            trajectory_id=f"{config.stage}-neuronio-{index:06d}",
            trajectory_index=index,
            seed=config.root_seed + index,
            split="validation" if index in validation else "train",
            duration_ms=config.trajectory_duration_ms,
        )
        for index in range(count)
    ]
    shards: List[ShardPlan] = []
    width = max(5, len(str(config.shard_count - 1)))
    for shard_index, start in enumerate(
        range(0, count, config.trajectories_per_shard)
    ):
        rows = tuple(trajectories[start : start + config.trajectories_per_shard])
        identity = {
            "schema_version": "giada-runpod-plan-v1",
            "config": config.to_dict(),
            "shard_index": shard_index,
            "trajectories": [asdict(row) for row in rows],
        }
        shards.append(
            ShardPlan(
                shard_id=f"shard-{shard_index:0{width}d}",
                shard_index=shard_index,
                trajectories=rows,
                expected_transition_count=sum(row.duration_ms for row in rows),
                plan_sha256=hashlib.sha256(_canonical_json(identity).encode()).hexdigest(),
            )
        )
    if sum(row.expected_transition_count for row in shards) != config.target_transitions:
        raise RuntimeError("shard plan transition accounting failed")
    return shards


def write_shard_plan(path: Path, config: ScaleConfig, shards: Iterable[ShardPlan]) -> None:
    payload = {
        "schema_version": "giada-runpod-plan-v1",
        "config": config.to_dict(),
        "shards": [row.to_dict() for row in shards],
    }
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(destination)


def load_shard_plan(path: Path) -> tuple[ScaleConfig, List[ShardPlan]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    config = ScaleConfig.from_mapping(payload["config"])
    shards = []
    for raw in payload["shards"]:
        trajectories = tuple(TrajectoryPlan(**row) for row in raw["trajectories"])
        shards.append(
            ShardPlan(
                shard_id=str(raw["shard_id"]),
                shard_index=int(raw["shard_index"]),
                trajectories=trajectories,
                expected_transition_count=int(raw["expected_transition_count"]),
                plan_sha256=str(raw["plan_sha256"]),
            )
        )
    if len(shards) != config.shard_count:
        raise ValueError("stored shard count differs from config")
    return config, shards
