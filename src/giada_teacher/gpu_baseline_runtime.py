"""Executable runtime helpers for the common GIADA GPU baseline."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import statistics
from typing import Any, Callable, Iterator


def paired_index_stream(length: int, batch_size: int, steps: int, seed: int) -> dict[str, Any]:
    if min(length, batch_size, steps) <= 0:
        raise ValueError("length, batch_size and steps must be positive")
    rng = random.Random(int(seed))
    batches: list[list[int]] = []
    permutation: list[int] = []
    cursor = 0
    while len(batches) < steps:
        if cursor >= len(permutation):
            permutation = list(range(length))
            rng.shuffle(permutation)
            cursor = 0
        take = min(batch_size, len(permutation) - cursor)
        batch = permutation[cursor : cursor + take]
        cursor += take
        if take < batch_size:
            permutation = list(range(length))
            rng.shuffle(permutation)
            remainder = batch_size - take
            batch.extend(permutation[:remainder])
            cursor = remainder
        batches.append(batch)
    encoded = json.dumps(batches, separators=(",", ":")).encode("ascii")
    return {
        "length": length,
        "batch_size": batch_size,
        "steps": steps,
        "seed": seed,
        "sha256": hashlib.sha256(encoded).hexdigest(),
        "batches": batches,
    }


def paired_index_generator(length: int, batch_size: int, seed: int) -> Iterator[list[int]]:
    """Yield the same paired batches as ``paired_index_stream`` in O(length) memory."""

    if min(length, batch_size) <= 0:
        raise ValueError("length and batch_size must be positive")
    rng = random.Random(int(seed))
    permutation: list[int] = []
    cursor = 0
    while True:
        if cursor >= len(permutation):
            permutation = list(range(length))
            rng.shuffle(permutation)
            cursor = 0
        take = min(batch_size, len(permutation) - cursor)
        batch = permutation[cursor : cursor + take]
        cursor += take
        if take < batch_size:
            permutation = list(range(length))
            rng.shuffle(permutation)
            remainder = batch_size - take
            batch.extend(permutation[:remainder])
            cursor = remainder
        yield batch


def configure_torch_runtime(seed: int):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch

    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if hasattr(torch.backends.cuda.matmul, "allow_tf32"):
        torch.backends.cuda.matmul.allow_tf32 = False
    if hasattr(torch.backends.cudnn, "allow_tf32"):
        torch.backends.cudnn.allow_tf32 = False
    return torch


def environment_manifest(torch_module=None) -> dict[str, Any]:
    torch = torch_module
    if torch is None:
        try:
            import torch
        except ImportError:
            torch = None
    result = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    }
    if torch is not None:
        result.update({
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cuda_available": bool(torch.cuda.is_available()),
        })
        if torch.cuda.is_available():
            result.update({
                "gpu_name": torch.cuda.get_device_name(0),
                "gpu_count": torch.cuda.device_count(),
            })
    return result


def benchmark_cuda(
    function: Callable[[], Any], *, warmup_iterations: int = 100, timed_iterations: int = 1000
) -> dict[str, Any]:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the registered GPU latency benchmark")
    if min(warmup_iterations, timed_iterations) <= 0:
        raise ValueError("benchmark iteration counts must be positive")
    with torch.inference_mode():
        for _ in range(warmup_iterations):
            function()
        torch.cuda.synchronize()
        timings = []
        for _ in range(timed_iterations):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            function()
            end.record()
            torch.cuda.synchronize()
            timings.append(float(start.elapsed_time(end)))
    ordered = sorted(timings)
    percentile = lambda fraction: ordered[round((len(ordered) - 1) * fraction)]
    return {
        "unit": "ms",
        "warmup_iterations": warmup_iterations,
        "timed_iterations": timed_iterations,
        "median": statistics.median(ordered),
        "p05": percentile(0.05),
        "p95": percentile(0.95),
        "minimum": ordered[0],
    }
