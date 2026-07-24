"""Unchanged B3 backbone plus zero-inflated output heads for notebook 02b."""

from __future__ import annotations

from typing import Any, Dict, Mapping

try:
    import torch
    from torch import nn
except ImportError:  # pragma: no cover - Kaggle supplies PyTorch.
    torch = None
    nn = None

from .full_state_flowmap import (
    FlowmapModelConfig,
    StructuredSharedResidual,
    require_torch,
)


if nn is not None:

    class ReconditionedStructuredResidual(nn.Module):
        """B3 with an activity head; the shared backbone is unchanged."""

        def __init__(
            self,
            config: FlowmapModelConfig,
            metadata: Mapping[str, Any],
            arrays: Mapping[str, Any],
        ) -> None:
            super().__init__()
            self.config = config
            self.backbone = StructuredSharedResidual(config, metadata, arrays)
            hidden = int(config.hidden_dim)
            self.activity_head = nn.Sequential(
                nn.LayerNorm(hidden),
                nn.Linear(hidden, 1),
            )

        def forward(self, batch: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
            result = self.backbone(batch)
            result["activity_logits"] = self.activity_head(
                result["token_hidden"]
            ).squeeze(-1)
            return result


else:

    class ReconditionedStructuredResidual:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            require_torch()
