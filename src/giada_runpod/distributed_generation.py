"""Static multi-pod planning and exclusive claims for CPU generation."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import socket
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable

from .config import ScaleConfig
from .planning import ShardPlan, TrajectoryPlan, iter_shard_plan


DISTRIBUTED_PLAN_SCHEMA = "giada-runpod-distributed-plan-v1"
WORKER_PARTITION_SCHEMA = "giada-runpod-worker-partition-v1"
CLAIM_SCHEMA = "giada-runpod-exclusive-claim-v1"


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def _shard_from_mapping(raw: Dict[str, Any], config: ScaleConfig) -> ShardPlan:
    trajectories = tuple(TrajectoryPlan(**row) for row in raw["trajectories"])
    shard = ShardPlan(
        shard_id=str(raw["shard_id"]),
        shard_index=int(raw["shard_index"]),
        trajectories=trajectories,
        expected_transition_count=int(raw["expected_transition_count"]),
        plan_sha256=str(raw["plan_sha256"]),
    )
    identity = {
        "schema_version": "giada-runpod-plan-v1",
        "config": config.to_dict(),
        "shard_index": shard.shard_index,
        "trajectories": [asdict(row) for row in shard.trajectories],
    }
    observed = hashlib.sha256(_canonical_json(identity).encode()).hexdigest()
    if observed != shard.plan_sha256:
        raise ValueError(f"immutable shard identity mismatch for {shard.shard_id}")
    if sum(row.duration_ms for row in trajectories) != shard.expected_transition_count:
        raise ValueError(f"transition accounting mismatch for {shard.shard_id}")
    return shard


def write_distributed_plan(
    root: Path,
    config: ScaleConfig,
    global_worker_count: int,
) -> Dict[str, Any]:
    """Stream one canonical plan into immutable modulo-assigned JSONL partitions."""

    config.validate()
    if global_worker_count <= 0:
        raise ValueError("global worker count must be positive")
    destination = Path(root)
    if destination.exists():
        raise FileExistsError(f"distributed plan already exists: {destination}")
    temporary = destination.with_name(f"{destination.name}.building-{os.getpid()}")
    if temporary.exists():
        raise FileExistsError(f"temporary distributed plan already exists: {temporary}")
    workers_root = temporary / "workers"
    workers_root.mkdir(parents=True)
    handles = []
    counts = [0] * global_worker_count
    transitions = [0] * global_worker_count
    try:
        for worker_index in range(global_worker_count):
            path = workers_root / f"worker-{worker_index:05d}.jsonl"
            handle = path.open("w", encoding="utf-8", newline="\n")
            header = {
                "record_type": "header",
                "schema_version": WORKER_PARTITION_SCHEMA,
                "config": config.to_dict(),
                "global_worker_count": global_worker_count,
                "worker_index": worker_index,
            }
            handle.write(_canonical_json(header) + "\n")
            handles.append(handle)
        for shard in iter_shard_plan(config):
            worker_index = shard.shard_index % global_worker_count
            handles[worker_index].write(
                _canonical_json({"record_type": "shard", **shard.to_dict()}) + "\n"
            )
            counts[worker_index] += 1
            transitions[worker_index] += shard.expected_transition_count
    except Exception:
        for handle in handles:
            handle.close()
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    for handle in handles:
        handle.flush()
        os.fsync(handle.fileno())
        handle.close()

    partitions = []
    for worker_index in range(global_worker_count):
        relative = Path("workers") / f"worker-{worker_index:05d}.jsonl"
        path = temporary / relative
        partitions.append(
            {
                "worker_index": worker_index,
                "path": relative.as_posix(),
                "sha256": _sha256_file(path),
                "shard_count": counts[worker_index],
                "transition_count": transitions[worker_index],
            }
        )
    manifest = {
        "schema_version": DISTRIBUTED_PLAN_SCHEMA,
        "project": "GIADA",
        "config": config.to_dict(),
        "global_worker_count": global_worker_count,
        "shard_count": sum(counts),
        "trajectory_count": config.trajectory_count,
        "transition_count": sum(transitions),
        "partition_rule": "shard_index_modulo_global_worker_count",
        "partitions": partitions,
    }
    manifest["plan_identity_sha256"] = hashlib.sha256(
        _canonical_json(manifest).encode()
    ).hexdigest()
    if manifest["shard_count"] != config.shard_count:
        raise RuntimeError("distributed shard count accounting failed")
    if manifest["transition_count"] != config.target_transitions:
        raise RuntimeError("distributed transition accounting failed")
    _atomic_json(temporary / "manifest.json", manifest)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary.replace(destination)
    return manifest


def load_distributed_manifest(root: Path) -> tuple[ScaleConfig, Dict[str, Any]]:
    """Authenticate the coordinator manifest before any partition is trusted."""

    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != DISTRIBUTED_PLAN_SCHEMA:
        raise ValueError("unknown distributed plan schema")
    identity = dict(manifest)
    expected_identity = identity.pop("plan_identity_sha256", None)
    observed_identity = hashlib.sha256(_canonical_json(identity).encode()).hexdigest()
    if expected_identity != observed_identity:
        raise ValueError("distributed plan manifest identity mismatch")
    config = ScaleConfig.from_mapping(manifest["config"])
    global_worker_count = int(manifest["global_worker_count"])
    partitions = manifest.get("partitions", [])
    indices = [int(row["worker_index"]) for row in partitions]
    if len(partitions) != global_worker_count or sorted(indices) != list(
        range(global_worker_count)
    ):
        raise ValueError("distributed manifest partition coverage mismatch")
    if sum(int(row["shard_count"]) for row in partitions) != int(
        manifest["shard_count"]
    ):
        raise ValueError("distributed manifest shard accounting mismatch")
    if sum(int(row["transition_count"]) for row in partitions) != int(
        manifest["transition_count"]
    ):
        raise ValueError("distributed manifest transition accounting mismatch")
    if int(manifest["shard_count"]) != config.shard_count:
        raise ValueError("distributed manifest differs from config shard count")
    if int(manifest["transition_count"]) != config.target_transitions:
        raise ValueError("distributed manifest differs from config transition count")
    return config, manifest


def load_worker_partition(
    root: Path,
    worker_index: int,
    global_worker_count: int,
) -> tuple[ScaleConfig, list[ShardPlan], Dict[str, Any]]:
    """Load and fully authenticate one worker's small partition."""

    root = Path(root)
    config, manifest = load_distributed_manifest(root)
    if int(manifest["global_worker_count"]) != global_worker_count:
        raise ValueError("worker count differs from immutable distributed plan")
    if not 0 <= worker_index < global_worker_count:
        raise ValueError("worker index must lie in [0, global_worker_count)")
    metadata = next(
        (
            row
            for row in manifest["partitions"]
            if int(row["worker_index"]) == worker_index
        ),
        None,
    )
    if metadata is None:
        raise ValueError(f"missing partition metadata for worker {worker_index}")
    path = (root / str(metadata["path"])).resolve()
    if root.resolve() not in path.parents:
        raise ValueError("worker partition path escapes distributed plan root")
    if _sha256_file(path) != metadata["sha256"]:
        raise ValueError(f"partition hash mismatch for worker {worker_index}")
    shards: list[ShardPlan] = []
    with path.open("r", encoding="utf-8") as handle:
        header = json.loads(next(handle))
        if header.get("record_type") != "header" or header.get("schema_version") != WORKER_PARTITION_SCHEMA:
            raise ValueError("unknown worker partition schema")
        if int(header["worker_index"]) != worker_index:
            raise ValueError("worker partition index mismatch")
        if int(header["global_worker_count"]) != global_worker_count:
            raise ValueError("worker partition count mismatch")
        if ScaleConfig.from_mapping(header["config"]) != config:
            raise ValueError("worker partition config mismatch")
        for line in handle:
            raw = json.loads(line)
            if raw.pop("record_type", None) != "shard":
                raise ValueError("unexpected worker partition record")
            shard = _shard_from_mapping(raw, config)
            if shard.shard_index % global_worker_count != worker_index:
                raise ValueError(f"misassigned shard {shard.shard_id}")
            shards.append(shard)
    if len(shards) != int(metadata["shard_count"]):
        raise ValueError("worker partition shard count mismatch")
    if sum(row.expected_transition_count for row in shards) != int(
        metadata["transition_count"]
    ):
        raise ValueError("worker partition transition count mismatch")
    if len({row.shard_index for row in shards}) != len(shards):
        raise ValueError("duplicate shard index within worker partition")
    return config, shards, manifest


@dataclass
class ExclusiveClaim:
    path: Path
    owner: Dict[str, Any]
    acquired: bool = True

    def release(self) -> None:
        if self.acquired:
            self.path.unlink(missing_ok=False)
            self.acquired = False


def acquire_exclusive_claim(
    path: Path,
    *,
    kind: str,
    identity: str,
    worker_index: int,
    global_worker_count: int,
    recover_stale: bool = False,
) -> ExclusiveClaim:
    """Atomically claim a worker or shard; recovery always requires an explicit flag."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if recover_stale and destination.exists():
        destination.unlink()
    owner = {
        "schema_version": CLAIM_SCHEMA,
        "kind": kind,
        "identity": identity,
        "worker_index": int(worker_index),
        "global_worker_count": int(global_worker_count),
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    try:
        descriptor = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
    except FileExistsError as error:
        existing = destination.read_text(encoding="utf-8", errors="replace")
        raise RuntimeError(
            f"exclusive {kind} claim already exists at {destination}; "
            f"refusing concurrent write. Existing owner: {existing}"
        ) from error
    try:
        payload = (json.dumps(owner, indent=2, sort_keys=True) + "\n").encode()
        os.write(descriptor, payload)
        os.fsync(descriptor)
    except Exception:
        os.close(descriptor)
        destination.unlink(missing_ok=True)
        raise
    os.close(descriptor)
    return ExclusiveClaim(destination, owner)
