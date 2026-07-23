"""Small, NEURON-independent helpers for the canonical teacher audit."""

import ast
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple


def sha256_file(path: Path) -> str:
    """Return a stable SHA-256 digest without loading a whole file in memory."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit(repository: Path) -> str:
    """Resolve the commit recorded by a local Git checkout."""

    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=str(Path(repository)),
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def repository_file_record(path: Path, repository: Path) -> Dict[str, Any]:
    """Describe a source file using a repository-relative path and digest."""

    resolved_path = Path(path).resolve()
    resolved_repository = Path(repository).resolve()
    relative_path = resolved_path.relative_to(resolved_repository)
    return {
        "path": relative_path.as_posix(),
        "sha256": sha256_file(resolved_path),
        "size_bytes": resolved_path.stat().st_size,
    }


def load_source_functions(
    source_path: Path,
    function_names: Sequence[str],
    namespace: Optional[Mapping[str, Any]] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Load selected function definitions without executing module top level.

    The upstream Hay generator is a monolithic Python 2-era script whose top
    level launches the full dataset job. Extracting only named ``FunctionDef``
    nodes lets an audit call the authoritative factory implementations while
    avoiding that unrelated side effect.
    """

    path = Path(source_path)
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    requested = tuple(function_names)
    selected = {
        node.name: node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name in requested
    }
    missing = [name for name in requested if name not in selected]
    if missing:
        raise ValueError(f"source functions not found in {path}: {missing}")

    ordered_nodes = [selected[name] for name in requested]
    module = ast.Module(body=ordered_nodes, type_ignores=[])
    ast.fix_missing_locations(module)
    execution_namespace: Dict[str, Any] = dict(namespace or {})
    execution_namespace.setdefault("__builtins__", __builtins__)
    exec(compile(module, str(path), "exec"), execution_namespace)
    functions = {name: execution_namespace[name] for name in requested}
    provenance = {
        "source_path": str(path),
        "source_sha256": sha256_file(path),
        "function_names": list(requested),
        "strategy": "selected_ast_function_definitions_without_module_top_level",
    }
    return functions, provenance


def validate_parent_tree(
    parent_by_id: Mapping[int, Optional[int]],
) -> Dict[str, Any]:
    """Validate a single rooted parent graph and return audit statistics."""

    node_ids = set(parent_by_id)
    roots = [node_id for node_id, parent in parent_by_id.items() if parent is None]
    invalid_parents = {
        node_id: parent
        for node_id, parent in parent_by_id.items()
        if parent is not None and parent not in node_ids
    }
    if invalid_parents:
        raise ValueError(f"unknown parent ids: {invalid_parents}")
    visiting = set()
    visited = set()

    def visit(node_id: int) -> None:
        if node_id in visited:
            return
        if node_id in visiting:
            raise ValueError(f"parent graph contains a cycle at {node_id}")
        visiting.add(node_id)
        parent = parent_by_id[node_id]
        if parent is not None:
            visit(parent)
        visiting.remove(node_id)
        visited.add(node_id)

    for node_id in parent_by_id:
        visit(node_id)
    if len(roots) != 1:
        raise ValueError(f"expected exactly one root, found {roots}")
    return {
        "root_id": roots[0],
        "node_count": len(node_ids),
        "acyclic": True,
        "parent_indices_valid": True,
    }


def json_ready(value: Any) -> Any:
    """Convert common NumPy/path-like values into strict JSON values."""

    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_ready(item) for item in value]
    if hasattr(value, "item"):
        try:
            return json_ready(value.item())
        except (TypeError, ValueError):
            pass
    if isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):
            return None
    return value


def write_json(path: Path, value: Any) -> None:
    """Write deterministic, human-readable JSON."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(json_ready(value), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def detect_spikes(
    times_ms: Iterable[float],
    voltages_mv: Iterable[float],
    threshold_mv: float = -20.0,
) -> list:
    """Return upward threshold-crossing times."""

    times = list(times_ms)
    voltages = list(voltages_mv)
    if len(times) != len(voltages):
        raise ValueError("time and voltage arrays must have equal length")
    return [
        float(times[index])
        for index in range(1, len(times))
        if voltages[index - 1] < threshold_mv <= voltages[index]
    ]
