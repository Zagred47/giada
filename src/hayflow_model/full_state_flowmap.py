"""Diagnostic baselines for the one-millisecond full-state flow map."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np

try:  # Keep B0/B1 and schema tests usable without a local PyTorch install.
    import torch
    from torch import nn
except ImportError:  # pragma: no cover - exercised in the Kaggle runtime.
    torch = None
    nn = None


@dataclass(frozen=True)
class FlowmapModelConfig:
    model_kind: str
    state_mode: str = "full"
    input_encoding: str = "U1"
    privileged_loss: bool = False
    hidden_dim: int = 64
    embedding_dim: int = 12
    residual_blocks: int = 2
    dropout: float = 0.05
    flat_projection_dim: int = 192
    auxiliary_dense_dim: int = 0

    def validate(self) -> None:
        if self.model_kind not in {"B2_flat_mlp", "B3_structured"}:
            raise ValueError(f"unsupported model kind {self.model_kind!r}")
        if self.state_mode not in {"voltage_only", "full"}:
            raise ValueError(f"unsupported state mode {self.state_mode!r}")
        if self.input_encoding not in {"none", "U1", "U2"}:
            raise ValueError(f"unsupported input encoding {self.input_encoding!r}")
        if min(self.hidden_dim, self.embedding_dim, self.residual_blocks) <= 0:
            raise ValueError("model dimensions must be positive")

    def to_dict(self) -> Dict[str, Any]:
        self.validate()
        return asdict(self)


class PersistenceBaseline:
    """B0: zero raw delta, hence S-hat(t+1) = S(t)."""

    name = "B0_persistence"

    @staticmethod
    def predict_raw(state_t: np.ndarray) -> np.ndarray:
        return np.asarray(state_t, dtype=np.float64).copy()


class DualRidgeBaseline:
    """B1: multi-output ridge solved in the sample-space dual."""

    name = "B1_affine_delta"

    def __init__(self, alpha: float = 1.0) -> None:
        if float(alpha) <= 0.0:
            raise ValueError("ridge alpha must be positive")
        self.alpha = float(alpha)
        self.feature_center: Optional[np.ndarray] = None
        self.feature_scale: Optional[np.ndarray] = None
        self.target_center: Optional[np.ndarray] = None
        self.train_features: Optional[np.ndarray] = None
        self.dual_coefficients: Optional[np.ndarray] = None

    def fit(self, features: np.ndarray, targets: np.ndarray) -> "DualRidgeBaseline":
        x = np.asarray(features, dtype=np.float64)
        y = np.asarray(targets, dtype=np.float64)
        if x.ndim != 2 or y.ndim != 2 or x.shape[0] != y.shape[0]:
            raise ValueError("ridge features/targets must be aligned matrices")
        self.feature_center = x.mean(axis=0)
        self.feature_scale = x.std(axis=0)
        self.feature_scale[self.feature_scale < 1e-8] = 1.0
        standardized = (x - self.feature_center) / self.feature_scale
        self.target_center = y.mean(axis=0)
        centered_target = y - self.target_center
        gram = standardized @ standardized.T
        gram.flat[:: gram.shape[0] + 1] += self.alpha
        self.dual_coefficients = np.linalg.solve(gram, centered_target)
        self.train_features = standardized.astype(np.float32)
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        if self.dual_coefficients is None:
            raise RuntimeError("ridge baseline has not been fit")
        x = (np.asarray(features) - self.feature_center) / self.feature_scale
        kernel = x @ self.train_features.T
        return kernel @ self.dual_coefficients + self.target_center

    def save(self, path: Path) -> None:
        if self.dual_coefficients is None:
            raise RuntimeError("ridge baseline has not been fit")
        np.savez_compressed(
            path,
            alpha=self.alpha,
            feature_center=self.feature_center,
            feature_scale=self.feature_scale,
            target_center=self.target_center,
            train_features=self.train_features,
            dual_coefficients=self.dual_coefficients,
        )

    @classmethod
    def load(cls, path: Path) -> "DualRidgeBaseline":
        with np.load(path) as data:
            result = cls(float(data["alpha"]))
            for name in (
                "feature_center",
                "feature_scale",
                "target_center",
                "train_features",
                "dual_coefficients",
            ):
                setattr(result, name, np.asarray(data[name]))
        return result


def ridge_design_matrix(
    batch: Mapping[str, np.ndarray],
    *,
    voltage_width: int,
    state_mode: str,
    input_encoding: str,
    maximum_events: int = 128,
) -> np.ndarray:
    """Build a deterministic causal design matrix for B1."""

    state = np.asarray(batch["state_t"], dtype=np.float32)
    pieces = [state[:, :voltage_width] if state_mode == "voltage_only" else state]
    if input_encoding == "U1":
        pieces.append(np.asarray(batch["u1"], dtype=np.float32).reshape(len(state), -1))
    elif input_encoding == "U2":
        events = np.asarray(batch["u2_features"], dtype=np.float32)
        mask = np.asarray(batch["u2_mask"], dtype=np.float32)
        segments = np.asarray(batch["u2_segment_ids"], dtype=np.float32)
        width = events.shape[-1]
        padded = np.zeros((len(state), int(maximum_events), width + 2), dtype=np.float32)
        count = min(events.shape[1], int(maximum_events))
        padded[:, :count, :width] = events[:, :count]
        padded[:, :count, width] = segments[:, :count] / 641.0
        padded[:, :count, width + 1] = mask[:, :count]
        pieces.append(padded.reshape(len(state), -1))
    return np.concatenate(pieces, axis=1)


def require_torch() -> None:
    if torch is None:
        raise RuntimeError("B2/B3 training requires PyTorch")


if nn is not None:

    class ResidualBlock(nn.Module):
        def __init__(self, width: int, dropout: float) -> None:
            super().__init__()
            self.network = nn.Sequential(
                nn.LayerNorm(width),
                nn.Linear(width, 2 * width),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(2 * width, width),
            )

        def forward(self, values: torch.Tensor) -> torch.Tensor:
            return values + self.network(values)


    class GlobalEventEncoder(nn.Module):
        def __init__(self, u1_width: int, u2_width: int, hidden: int) -> None:
            super().__init__()
            self.u1 = nn.Sequential(nn.Linear(u1_width, hidden), nn.SiLU())
            self.u2 = nn.Sequential(
                nn.Linear(u2_width, hidden), nn.SiLU(), nn.Linear(hidden, hidden)
            )

        def forward(
            self,
            batch: Mapping[str, torch.Tensor],
            encoding: str,
            segment_count: int,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            batch_size = batch["state_t"].shape[0]
            if encoding == "none":
                local = batch["state_t"].new_zeros(
                    (batch_size, segment_count, self.u1[0].out_features)
                )
            elif encoding == "U1":
                local = self.u1(batch["u1"])
            else:
                encoded = self.u2(batch["u2_features"])
                encoded = encoded * batch["u2_mask"].unsqueeze(-1)
                local = encoded.new_zeros(
                    (batch_size, segment_count, encoded.shape[-1])
                )
                for row in range(batch_size):
                    local[row].index_add_(
                        0, batch["u2_segment_ids"][row], encoded[row]
                    )
            total = local.sum(dim=1)
            maximum = local.amax(dim=1)
            return local, torch.cat([total, maximum], dim=-1)


    class EventHeads(nn.Module):
        def __init__(self, hidden: int, event_count: int, region_count: int) -> None:
            super().__init__()
            self.event_count = event_count
            self.region_count = region_count
            self.presence = nn.Linear(hidden, event_count)
            self.timing = nn.Linear(hidden, event_count * 4)
            self.region = nn.Linear(hidden, event_count * region_count)

        def forward(self, hidden: torch.Tensor) -> Dict[str, torch.Tensor]:
            return {
                "event_logits": self.presence(hidden),
                "event_timing": self.timing(hidden).view(-1, self.event_count, 4),
                "event_region_logits": self.region(hidden).view(
                    -1, self.event_count, self.region_count
                ),
            }


    class FlatResidualMLP(nn.Module):
        """B2 sanity check with a controlled flat projection."""

        def __init__(
            self,
            config: FlowmapModelConfig,
            *,
            state_width: int,
            voltage_width: int,
            u1_width: int,
            u2_width: int,
            segment_count: int,
            event_count: int,
            region_count: int,
        ) -> None:
            super().__init__()
            config.validate()
            self.config = config
            self.state_width = state_width
            self.voltage_width = voltage_width
            self.segment_count = segment_count
            input_width = voltage_width if config.state_mode == "voltage_only" else state_width
            self.state_projection = nn.Linear(input_width, config.flat_projection_dim)
            self.events = GlobalEventEncoder(u1_width, u2_width, config.hidden_dim)
            merged = config.flat_projection_dim + 2 * config.hidden_dim
            self.input = nn.Linear(merged, config.hidden_dim)
            self.blocks = nn.Sequential(
                *[
                    ResidualBlock(config.hidden_dim, config.dropout)
                    for _ in range(config.residual_blocks)
                ]
            )
            self.delta = nn.Linear(config.hidden_dim, state_width)
            self.event_heads = EventHeads(
                config.hidden_dim, event_count, region_count
            )

        def forward(self, batch: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
            state = batch["state_t"]
            if self.config.state_mode == "voltage_only":
                state = state[:, : self.voltage_width]
            _, global_events = self.events(
                batch, self.config.input_encoding, self.segment_count
            )
            hidden = torch.nn.functional.silu(
                self.input(torch.cat([self.state_projection(state), global_events], dim=-1))
            )
            hidden = self.blocks(hidden)
            result = {"delta": self.delta(hidden), "global_hidden": hidden}
            result.update(self.event_heads(hidden))
            return result


    class StructuredSharedResidual(nn.Module):
        """B3 shared local residual model over semantically indexed variables."""

        def __init__(
            self,
            config: FlowmapModelConfig,
            metadata: Mapping[str, Any],
            arrays: Mapping[str, np.ndarray],
        ) -> None:
            super().__init__()
            config.validate()
            self.config = config
            self.state_width = int(metadata["state_width"])
            self.segment_count = int(metadata["segment_count"])
            self.voltage_width = int(metadata["category_widths"]["voltage"])
            self.category_slices = {
                name: slice(*bounds)
                for name, bounds in metadata["category_slices"].items()
            }
            emb = config.embedding_dim
            hidden = config.hidden_dim
            self.category_embedding = nn.Embedding(len(metadata["category_names"]), emb)
            self.mechanism_embedding = nn.Embedding(len(metadata["mechanism_names"]), emb)
            self.variable_embedding = nn.Embedding(len(metadata["variable_names"]), emb)
            self.kind_embedding = nn.Embedding(len(metadata["kind_names"]), emb)
            self.region_embedding = nn.Embedding(len(metadata["region_names"]), emb)
            self.events = GlobalEventEncoder(
                len(metadata["u1_feature_names"]),
                len(metadata["u2_event_feature_names"]),
                hidden,
            )
            self.register_buffer("core_segment_ids", torch.as_tensor(arrays["core_segment_ids"], dtype=torch.long))
            self.register_buffer("core_category_ids", torch.as_tensor(arrays["core_category_ids"], dtype=torch.long))
            self.register_buffer("core_mechanism_ids", torch.as_tensor(arrays["core_mechanism_ids"], dtype=torch.long))
            self.register_buffer("core_variable_ids", torch.as_tensor(arrays["core_variable_ids"], dtype=torch.long))
            self.register_buffer("core_kind_ids", torch.as_tensor(arrays["core_kind_ids"], dtype=torch.long))
            self.register_buffer("segment_region_ids", torch.as_tensor(arrays["segment_region_ids"], dtype=torch.long))
            self.register_buffer("segment_static", torch.as_tensor(arrays["segment_static"], dtype=torch.float32))
            self.register_buffer("parent_ids", torch.as_tensor(arrays["parent_ids"], dtype=torch.long))
            child_ids = arrays["child_ids"]
            child_mask = arrays["child_mask"]
            self.register_buffer("child_ids", torch.as_tensor(child_ids, dtype=torch.long))
            self.register_buffer("child_mask", torch.as_tensor(child_mask, dtype=torch.float32))
            segment_input = 3 + arrays["segment_static"].shape[1] + emb + hidden + 2 * hidden
            self.segment_projection = nn.Sequential(
                nn.Linear(segment_input, hidden), nn.SiLU(), nn.Linear(hidden, hidden)
            )
            token_input = 1 + hidden + 4 * emb
            self.token_projection = nn.Sequential(
                nn.Linear(token_input, hidden), nn.SiLU(), nn.Linear(hidden, hidden)
            )
            self.blocks = nn.Sequential(
                *[
                    ResidualBlock(hidden, config.dropout)
                    for _ in range(config.residual_blocks)
                ]
            )
            self.delta_heads = nn.ModuleDict(
                {
                    name: nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, 1))
                    for name in metadata["category_names"]
                }
            )
            self.global_projection = nn.Sequential(
                nn.Linear(3 * hidden, hidden), nn.SiLU(), nn.Linear(hidden, hidden)
            )
            self.event_heads = EventHeads(
                hidden, len(metadata["event_kinds"]), len(metadata["region_names"])
            )
            self.privileged_decoder = None
            self.aux_dense = None
            if config.privileged_loss:
                self.register_buffer("privileged_segment_ids", torch.as_tensor(arrays["privileged_segment_ids"], dtype=torch.long))
                self.register_buffer("privileged_mechanism_ids", torch.as_tensor(arrays["privileged_mechanism_ids"], dtype=torch.long))
                self.register_buffer("privileged_variable_ids", torch.as_tensor(arrays["privileged_variable_ids"], dtype=torch.long))
                self.register_buffer("privileged_kind_ids", torch.as_tensor(arrays["privileged_kind_ids"], dtype=torch.long))
                self.privileged_decoder = nn.Sequential(
                    nn.Linear(hidden + 3 * emb, hidden),
                    nn.SiLU(),
                    nn.Linear(hidden, 1),
                )
                self.aux_dense = nn.Linear(hidden, int(config.auxiliary_dense_dim))

        def _child_mean(self, voltage: torch.Tensor) -> torch.Tensor:
            gathered = voltage[:, self.child_ids]
            weighted = gathered * self.child_mask.unsqueeze(0)
            count = self.child_mask.sum(dim=1).clamp_min(1.0)
            return weighted.sum(dim=-1) / count.unsqueeze(0)

        def forward(self, batch: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
            state = batch["state_t"]
            batch_size = state.shape[0]
            voltage = state[:, : self.voltage_width]
            local_events, global_events = self.events(
                batch, self.config.input_encoding, self.segment_count
            )
            voltage_summary = torch.stack(
                [voltage.mean(1), voltage.std(1), voltage.amin(1), voltage.amax(1)],
                dim=-1,
            )
            global_seed = torch.cat([global_events, voltage_summary], dim=-1)
            # Pad the four scalar summaries to the event-global width without
            # introducing a second large state projection.
            if global_seed.shape[-1] < 2 * self.config.hidden_dim:
                global_seed = torch.nn.functional.pad(
                    global_seed,
                    (0, 2 * self.config.hidden_dim - global_seed.shape[-1]),
                )
            global_seed = global_seed[:, : 2 * self.config.hidden_dim]
            global_expanded = global_seed.unsqueeze(1).expand(-1, self.segment_count, -1)
            segment_input = torch.cat(
                [
                    voltage.unsqueeze(-1),
                    voltage[:, self.parent_ids].unsqueeze(-1),
                    self._child_mean(voltage).unsqueeze(-1),
                    self.segment_static.unsqueeze(0).expand(batch_size, -1, -1),
                    self.region_embedding(self.segment_region_ids).unsqueeze(0).expand(batch_size, -1, -1),
                    local_events,
                    global_expanded,
                ],
                dim=-1,
            )
            segment_hidden = self.segment_projection(segment_input)
            token_state = state
            if self.config.state_mode == "voltage_only":
                token_state = token_state.clone()
                token_state[:, self.voltage_width :] = 0.0
            token_input = torch.cat(
                [
                    token_state.unsqueeze(-1),
                    segment_hidden[:, self.core_segment_ids, :],
                    self.category_embedding(self.core_category_ids).unsqueeze(0).expand(batch_size, -1, -1),
                    self.mechanism_embedding(self.core_mechanism_ids).unsqueeze(0).expand(batch_size, -1, -1),
                    self.variable_embedding(self.core_variable_ids).unsqueeze(0).expand(batch_size, -1, -1),
                    self.kind_embedding(self.core_kind_ids).unsqueeze(0).expand(batch_size, -1, -1),
                ],
                dim=-1,
            )
            token_hidden = self.blocks(self.token_projection(token_input))
            delta = token_hidden.new_zeros((batch_size, self.state_width))
            for name, state_slice in self.category_slices.items():
                delta[:, state_slice] = self.delta_heads[name](
                    token_hidden[:, state_slice, :]
                ).squeeze(-1)
            soma = segment_hidden[:, 0, :]
            mean = segment_hidden.mean(dim=1)
            maximum = segment_hidden.amax(dim=1)
            global_hidden = self.global_projection(torch.cat([soma, mean, maximum], dim=-1))
            result = {
                "delta": delta,
                "global_hidden": global_hidden,
                "segment_hidden": segment_hidden,
            }
            result.update(self.event_heads(global_hidden))
            if self.privileged_decoder is not None:
                privileged_input = torch.cat(
                    [
                        segment_hidden[:, self.privileged_segment_ids, :],
                        self.mechanism_embedding(self.privileged_mechanism_ids).unsqueeze(0).expand(batch_size, -1, -1),
                        self.variable_embedding(self.privileged_variable_ids).unsqueeze(0).expand(batch_size, -1, -1),
                        self.kind_embedding(self.privileged_kind_ids).unsqueeze(0).expand(batch_size, -1, -1),
                    ],
                    dim=-1,
                )
                result["privileged_current"] = self.privileged_decoder(
                    privileged_input
                ).squeeze(-1)
                result["aux_dense"] = self.aux_dense(global_hidden)
            return result


else:

    class FlatResidualMLP:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            require_torch()


    class StructuredSharedResidual:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            require_torch()


def structured_arrays(layout: Any) -> Dict[str, np.ndarray]:
    """Convert a FlowmapLayout into fixed arrays registered by B3."""

    max_children = max(1, max(map(len, layout.children)))
    child_ids = np.zeros((layout.segment_count, max_children), dtype=np.int64)
    child_mask = np.zeros((layout.segment_count, max_children), dtype=np.float32)
    for segment_id, children in enumerate(layout.children):
        if children:
            child_ids[segment_id, : len(children)] = children
            child_mask[segment_id, : len(children)] = 1.0
        else:
            child_ids[segment_id, 0] = segment_id
            child_mask[segment_id, 0] = 1.0
    return {
        "core_segment_ids": layout.core_segment_ids,
        "core_category_ids": layout.core_category_ids,
        "core_mechanism_ids": layout.core_mechanism_ids,
        "core_variable_ids": layout.core_variable_ids,
        "core_kind_ids": layout.core_kind_ids,
        "privileged_segment_ids": layout.privileged_segment_ids,
        "privileged_mechanism_ids": layout.privileged_mechanism_ids,
        "privileged_variable_ids": layout.privileged_variable_ids,
        "privileged_kind_ids": layout.privileged_kind_ids,
        "segment_region_ids": layout.segment_region_ids,
        "segment_static": layout.segment_static,
        "parent_ids": layout.parent_ids,
        "child_ids": child_ids,
        "child_mask": child_mask,
    }


def parameter_count(model: Any) -> int:
    require_torch()
    return int(sum(parameter.numel() for parameter in model.parameters()))
