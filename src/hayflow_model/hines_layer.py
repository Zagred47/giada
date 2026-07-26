"""Differentiable Hines elimination for the canonical Hay morphology.

The layer solves a symmetric tree system without materialising a dense matrix.
The public contract uses positive child-to-parent couplings ``g`` and represents
the off-diagonal matrix entries as ``-g``::

    diagonal[i] * x[i] - g[i] * x[parent[i]] = rhs[i]

Every non-root row also contributes the same ``-g[i]`` entry to its parent.
The implementation is functional (no in-place mutation of differentiable
tensors), so gradients propagate through the diagonal, coupling and right-hand
side.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Sequence, Tuple

try:  # PyTorch is supplied by the Kaggle runtime.
    import torch
    from torch import nn
except ImportError:  # pragma: no cover - keeps schema-only imports usable.
    torch = None
    nn = None


def require_torch() -> None:
    if torch is None:
        raise RuntimeError("the differentiable Hines layer requires PyTorch")


def tree_depths(parent_ids: Sequence[int]) -> Tuple[Tuple[int, ...], ...]:
    """Return nodes grouped by depth and reject cycles/disconnected parents."""

    parents = tuple(int(value) for value in parent_ids)
    count = len(parents)
    if not count:
        raise ValueError("a Hines tree cannot be empty")
    roots = [index for index, parent in enumerate(parents) if parent == index]
    if len(roots) != 1:
        raise ValueError("a Hines tree must contain exactly one self-parented root")
    root = roots[0]
    depth = [-1] * count
    depth[root] = 0
    for node in range(count):
        if depth[node] >= 0:
            continue
        trail = []
        current = node
        seen = set()
        while depth[current] < 0:
            if current in seen:
                raise ValueError("cycle detected in Hines parent array")
            seen.add(current)
            trail.append(current)
            parent = parents[current]
            if not 0 <= parent < count:
                raise ValueError(f"parent index {parent} is outside the tree")
            current = parent
        value = depth[current]
        for child in reversed(trail):
            value += 1
            depth[child] = value
    groups = []
    for value in range(max(depth) + 1):
        groups.append(tuple(index for index, item in enumerate(depth) if item == value))
    return tuple(groups)


@dataclass(frozen=True)
class HinesDiagnostics:
    minimum_diagonal: float
    minimum_reduced_diagonal: float
    maximum_elimination_ratio: float
    finite: bool
    positive_diagonal: bool
    well_conditioned: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "minimum_diagonal": self.minimum_diagonal,
            "minimum_reduced_diagonal": self.minimum_reduced_diagonal,
            "maximum_elimination_ratio": self.maximum_elimination_ratio,
            "finite": self.finite,
            "positive_diagonal": self.positive_diagonal,
            "well_conditioned": self.well_conditioned,
        }


if nn is not None:

    class DifferentiableHinesSolve(nn.Module):
        """Batched differentiable Hines solve on one fixed tree."""

        def __init__(
            self,
            parent_ids: Sequence[int],
            *,
            minimum_diagonal: float = 1e-8,
            maximum_elimination_ratio: float = 1e8,
        ) -> None:
            super().__init__()
            parents = tuple(int(value) for value in parent_ids)
            groups = tree_depths(parents)
            self.root = int(groups[0][0])
            self.minimum_diagonal = float(minimum_diagonal)
            self.maximum_elimination_ratio = float(maximum_elimination_ratio)
            if self.minimum_diagonal <= 0.0:
                raise ValueError("minimum_diagonal must be positive")
            self.register_buffer("parent_ids", torch.as_tensor(parents, dtype=torch.long))
            self.register_buffer(
                "non_root_ids",
                torch.as_tensor(
                    [index for index in range(len(parents)) if index != self.root],
                    dtype=torch.long,
                ),
            )
            self.depth_count = len(groups)
            for depth, nodes in enumerate(groups):
                self.register_buffer(
                    f"depth_{depth}", torch.as_tensor(nodes, dtype=torch.long)
                )

        @property
        def segment_count(self) -> int:
            return int(self.parent_ids.numel())

        def _as_batched(self, value: torch.Tensor, name: str) -> torch.Tensor:
            if value.ndim == 1:
                value = value.unsqueeze(0)
            if value.ndim != 2 or value.shape[1] != self.segment_count:
                raise ValueError(
                    f"{name} must have shape [batch, {self.segment_count}]"
                )
            return value

        def eliminate(
            self,
            diagonal: torch.Tensor,
            coupling_to_parent: torch.Tensor,
            rhs: torch.Tensor,
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            diagonal = self._as_batched(diagonal, "diagonal")
            rhs = self._as_batched(rhs, "rhs")
            coupling = self._as_batched(coupling_to_parent, "coupling_to_parent")
            if diagonal.shape != rhs.shape or coupling.shape != diagonal.shape:
                raise ValueError("Hines tensors must share the same batched shape")
            if bool((diagonal <= self.minimum_diagonal).any()):
                raise RuntimeError("Hines diagonal is non-positive or numerically unsafe")
            if bool((coupling[:, self.non_root_ids] < 0.0).any()):
                raise RuntimeError("Hines coupling magnitudes must be non-negative")
            reduced_diagonal = diagonal
            reduced_rhs = rhs
            maximum_ratio = diagonal.new_zeros(())
            for depth in range(self.depth_count - 1, 0, -1):
                nodes = getattr(self, f"depth_{depth}")
                parents = self.parent_ids[nodes]
                child_diagonal = reduced_diagonal[:, nodes]
                child_coupling = coupling[:, nodes]
                ratio = child_coupling / child_diagonal
                maximum_ratio = torch.maximum(maximum_ratio, ratio.abs().amax())
                diagonal_change = child_coupling * ratio
                rhs_change = ratio * reduced_rhs[:, nodes]
                diagonal_scatter = torch.zeros_like(reduced_diagonal).scatter_add(
                    1, parents.unsqueeze(0).expand(diagonal.shape[0], -1), diagonal_change
                )
                rhs_scatter = torch.zeros_like(reduced_rhs).scatter_add(
                    1, parents.unsqueeze(0).expand(rhs.shape[0], -1), rhs_change
                )
                reduced_diagonal = reduced_diagonal - diagonal_scatter
                reduced_rhs = reduced_rhs + rhs_scatter
                if bool((reduced_diagonal[:, parents] <= self.minimum_diagonal).any()):
                    raise RuntimeError("Hines elimination produced a non-positive pivot")
            return reduced_diagonal, reduced_rhs, maximum_ratio

        def forward(
            self,
            diagonal: torch.Tensor,
            coupling_to_parent: torch.Tensor,
            rhs: torch.Tensor,
            *,
            return_diagnostics: bool = False,
        ) -> Any:
            original_was_vector = diagonal.ndim == 1
            diagonal_b = self._as_batched(diagonal, "diagonal")
            coupling_b = self._as_batched(coupling_to_parent, "coupling_to_parent")
            rhs_b = self._as_batched(rhs, "rhs")
            reduced_diagonal, reduced_rhs, maximum_ratio = self.eliminate(
                diagonal_b, coupling_b, rhs_b
            )
            root_index = torch.as_tensor([self.root], device=diagonal_b.device)
            root_value = reduced_rhs[:, root_index] / reduced_diagonal[:, root_index]
            solution = torch.zeros_like(reduced_rhs).scatter(
                1, root_index.unsqueeze(0).expand(diagonal_b.shape[0], -1), root_value
            )
            for depth in range(1, self.depth_count):
                nodes = getattr(self, f"depth_{depth}")
                parents = self.parent_ids[nodes]
                values = (
                    reduced_rhs[:, nodes]
                    + coupling_b[:, nodes] * solution[:, parents]
                ) / reduced_diagonal[:, nodes]
                solution = solution.scatter(
                    1, nodes.unsqueeze(0).expand(solution.shape[0], -1), values
                )
            finite = bool(torch.isfinite(solution).all())
            diagnostic = HinesDiagnostics(
                minimum_diagonal=float(diagonal_b.detach().amin().cpu()),
                minimum_reduced_diagonal=float(
                    reduced_diagonal.detach().amin().cpu()
                ),
                maximum_elimination_ratio=float(maximum_ratio.detach().cpu()),
                finite=finite,
                positive_diagonal=bool(
                    (reduced_diagonal.detach() > self.minimum_diagonal).all()
                ),
                well_conditioned=bool(
                    finite
                    and maximum_ratio.detach() < self.maximum_elimination_ratio
                ),
            )
            if not diagnostic.finite or not diagnostic.well_conditioned:
                raise RuntimeError(f"unsafe Hines solve: {diagnostic.to_dict()}")
            result = solution[0] if original_was_vector else solution
            return (result, diagnostic.to_dict()) if return_diagnostics else result

        def dense_matrix(
            self,
            diagonal: torch.Tensor,
            coupling_to_parent: torch.Tensor,
        ) -> torch.Tensor:
            """Materialise the system only for tests and diagnostics."""

            diagonal_b = self._as_batched(diagonal, "diagonal")
            coupling_b = self._as_batched(coupling_to_parent, "coupling_to_parent")
            batch = diagonal_b.shape[0]
            matrix = torch.diag_embed(diagonal_b)
            for node in range(self.segment_count):
                parent = int(self.parent_ids[node])
                if node == parent:
                    continue
                edge = coupling_b[:, node]
                rows = torch.full((batch,), node, dtype=torch.long, device=matrix.device)
                parents = torch.full((batch,), parent, dtype=torch.long, device=matrix.device)
                batches = torch.arange(batch, device=matrix.device)
                matrix = matrix.index_put((batches, rows, parents), -edge)
                matrix = matrix.index_put((batches, parents, rows), -edge)
            return matrix


else:

    class DifferentiableHinesSolve:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            require_torch()


def morphology_arrays(layout: Any) -> Dict[str, Any]:
    """Extract raw physical arrays used to construct the 1 ms Hines system."""

    import numpy as np

    segments = list(layout.segments)
    parents = np.asarray(
        [
            int(row["parent_segment_id"])
            if row.get("parent_segment_id") is not None
            else int(row["id"])
            for row in segments
        ],
        dtype=np.int64,
    )
    return {
        "parent_ids": parents,
        "capacitance_uf": np.asarray(
            [float(row["membrane_capacitance_uf"]) for row in segments],
            dtype=np.float64,
        ),
        "leak_conductance_us": np.asarray(
            [float(row["passive_leak_conductance_us"]) for row in segments],
            dtype=np.float64,
        ),
        "leak_reversal_mv": np.asarray(
            [float(row["passive_reversal_mv"]) for row in segments],
            dtype=np.float64,
        ),
        "axial_conductance_to_parent_us": np.asarray(
            [float(row["axial_conductance_to_parent_us"]) for row in segments],
            dtype=np.float64,
        ),
    }
