"""Morphology-aware recurrent HayFlow prototype with a true Hines solve.

This module intentionally does not reuse the B3 forward path.  B3 remains a
numerical baseline in notebook 05, while :class:`HayFlowHines` follows the
causal sequence specified for the prototype: realized synaptic drive, teacher
state encoding (initialisation only), recurrent local dynamics, one Hines
solve, dedicated event heads, local event-conditioned jump, and recurrent
state commits.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

try:  # Kaggle supplies torch; local schema tests may run without it.
    import torch
    import torch.nn.functional as functional
    from torch import nn
except ImportError:  # pragma: no cover
    torch = None
    functional = None
    nn = None

from ..hayflow_data.composite_flowmap import EVENT_KINDS
from ..hayflow_data.hines_inputs import (
    HINES_SYNAPTIC_FEATURE_NAMES,
    SYNAPTIC_COMPONENTS,
    SYNAPTIC_STATISTICS,
)
from .hines_layer import DifferentiableHinesSolve, morphology_arrays, require_torch


@dataclass(frozen=True)
class HayFlowHinesConfig:
    local_latent_dim: int = 16
    global_latent_dim: int = 32
    region_embedding_dim: int = 8
    token_embedding_dim: int = 8
    hidden_width: int = 96
    residual_blocks: int = 2
    synaptic_hidden_width: int = 32
    dt_ms: float = 1.0
    effective_conductance_scale_us: float = 0.02
    source_current_scale_na: float = 5.0
    continuous_residual_limit_mv: float = 12.0
    event_jump_limit_mv: float = 120.0
    event_spatial_temperature: float = 0.25
    event_boundary_decay_per_ms: float = 4.0
    convgru_voltage_delta_limit_mv: float = 120.0
    calcium_state_dim: int = 1
    synapse_state_dim: int = 4
    dropout: float = 0.0

    def validate(self) -> None:
        integer = (
            self.local_latent_dim,
            self.global_latent_dim,
            self.region_embedding_dim,
            self.token_embedding_dim,
            self.hidden_width,
            self.residual_blocks,
            self.synaptic_hidden_width,
            self.calcium_state_dim,
            self.synapse_state_dim,
        )
        if min(integer) <= 0:
            raise ValueError("HayFlow-Hines dimensions must be positive")
        if self.dt_ms <= 0.0:
            raise ValueError("dt_ms must be positive")
        if min(
            self.effective_conductance_scale_us,
            self.source_current_scale_na,
            self.continuous_residual_limit_mv,
            self.event_jump_limit_mv,
            self.event_spatial_temperature,
            self.event_boundary_decay_per_ms,
            self.convgru_voltage_delta_limit_mv,
        ) <= 0.0:
            raise ValueError("physical output scales must be positive")

    def to_dict(self) -> Dict[str, Any]:
        self.validate()
        return asdict(self)


def _vocabulary_ids(records: Sequence[Mapping[str, Any]], field: str) -> Tuple[np.ndarray, Tuple[str, ...]]:
    names = tuple(sorted({str(row.get(field, "unknown")) for row in records}))
    lookup = {name: index for index, name in enumerate(names)}
    return np.asarray([lookup[str(row.get(field, "unknown"))] for row in records], dtype=np.int64), names


def _biological_record(record: Mapping[str, Any]) -> bool:
    text = " ".join(
        str(record.get(name, ""))
        for name in ("category", "mechanism", "variable", "kind", "scope")
    ).lower()
    terms = (
        "nata", "nats", "nap_", "na_t", "k_", "kad", "kap", "kdr",
        "sk_e", "skv", "ca_", "calcium", "cai", "ih", "hcn", "ampa",
        "nmda", "gaba", "synap",
    )
    return any(term in text for term in terms)


def hayflow_hines_arrays(layout: Any) -> Dict[str, np.ndarray]:
    """Static semantic and physical arrays required by the prototype."""

    physical = morphology_arrays(layout)
    core = list(layout.core_records)
    privileged = list(layout.privileged_records)
    category_ids, category_names = _vocabulary_ids(core, "category")
    mechanism_ids, mechanism_names = _vocabulary_ids(core + privileged, "mechanism")
    variable_ids, variable_names = _vocabulary_ids(core + privileged, "variable")
    kind_ids, kind_names = _vocabulary_ids(core + privileged, "kind")
    mechanism_lookup = {name: index for index, name in enumerate(mechanism_names)}
    variable_lookup = {name: index for index, name in enumerate(variable_names)}
    kind_lookup = {name: index for index, name in enumerate(kind_names)}
    selected_core = np.asarray(
        [index for index, row in enumerate(core) if _biological_record(row)], dtype=np.int64
    )
    if not len(selected_core):
        selected_core = np.arange(min(len(core), layout.segment_count), dtype=np.int64)
    selected_privileged = np.asarray(
        [index for index, row in enumerate(privileged) if _biological_record(row)],
        dtype=np.int64,
    )
    # A selective diagnostic decoder is intentional: thousands of unrelated
    # currents must not dominate the primary voltage/event objective.
    if len(selected_core) > 2048:
        selected_core = selected_core[
            np.linspace(0, len(selected_core) - 1, 2048, dtype=np.int64)
        ]
    if len(selected_privileged) > 1024:
        selected_privileged = selected_privileged[
            np.linspace(0, len(selected_privileged) - 1, 1024, dtype=np.int64)
        ]
    selected_core_records = [core[int(index)] for index in selected_core]
    selected_privileged_records = [privileged[int(index)] for index in selected_privileged]
    max_children = max(1, max(len(children) for children in layout.children))
    child_ids = np.zeros((layout.segment_count, max_children), dtype=np.int64)
    child_mask = np.zeros((layout.segment_count, max_children), dtype=np.float32)
    child_axial = np.zeros((layout.segment_count, max_children), dtype=np.float32)
    axial = physical["axial_conductance_to_parent_us"]
    for parent, children in enumerate(layout.children):
        if not children:
            child_ids[parent, 0] = parent
            continue
        child_ids[parent, : len(children)] = children
        child_mask[parent, : len(children)] = 1.0
        child_axial[parent, : len(children)] = axial[np.asarray(children, dtype=np.int64)]
    concentration_indices = np.asarray(
        [
            index
            for index, row in enumerate(core)
            if str(row.get("kind", "")) == "concentration"
            or str(row.get("category", "")) == "calcium_ions"
        ],
        dtype=np.int64,
    )
    synapse_indices = np.asarray(
        [index for index, row in enumerate(core) if str(row.get("scope", "")) == "synapse"],
        dtype=np.int64,
    )
    synapse_channels = []
    for index in synapse_indices:
        row = core[int(index)]
        text = f"{row.get('mechanism', '')} {row.get('variable', '')}".lower()
        if "nmda" in text:
            channel = 1
        elif "gaba" in text or "inhib" in text:
            channel = 2
        elif "ampa" in text:
            channel = 0
        else:
            channel = 3
        synapse_channels.append(channel)
    axial_total = axial.copy()
    for node, parent in enumerate(physical["parent_ids"]):
        if node != int(parent):
            axial_total[int(parent)] += axial[node]
    region_names = tuple(layout.region_names)
    event_allowed = np.zeros((len(EVENT_KINDS), layout.segment_count), dtype=np.float32)
    allowed_by_event = {
        "axonal_spike": {"ais", "axon"},
        "somatic_spike": {"soma", "ais"},
        "backpropagating_ap": {"apical_trunk", "nexus", "hot_zone", "nexus_hot_zone", "tuft", "basal"},
        "calcium_spike": {"apical_trunk", "nexus", "hot_zone", "nexus_hot_zone", "tuft"},
        "nmda_spike": {"basal", "apical_trunk", "nexus", "hot_zone", "nexus_hot_zone", "tuft"},
        "nmda_plateau": {"basal", "apical_trunk", "nexus", "hot_zone", "nexus_hot_zone", "tuft"},
    }
    for event_index, event in enumerate(EVENT_KINDS):
        allowed = allowed_by_event[event]
        for segment_id, row in enumerate(layout.segments):
            if str(row["region"]).lower() in allowed:
                event_allowed[event_index, segment_id] = 1.0
        if not event_allowed[event_index].any():
            event_allowed[event_index, :] = 1.0
    result = {
        **physical,
        "axial_total_us": axial_total.astype(np.float32),
        "child_ids": child_ids,
        "child_mask": child_mask,
        "child_axial_us": child_axial,
        "core_segment_ids": np.asarray(layout.core_segment_ids, dtype=np.int64),
        "core_category_ids": category_ids,
        "core_mechanism_ids": np.asarray(
            [mechanism_lookup[str(row.get("mechanism", "unknown"))] for row in core],
            dtype=np.int64,
        ),
        "core_variable_ids": np.asarray(
            [variable_lookup[str(row.get("variable", "unknown"))] for row in core],
            dtype=np.int64,
        ),
        "core_kind_ids": np.asarray(
            [kind_lookup[str(row.get("kind", "unknown"))] for row in core], dtype=np.int64
        ),
        "selected_core_indices": selected_core,
        "selected_core_segment_ids": np.asarray(
            [layout.core_segment_ids[int(index)] for index in selected_core], dtype=np.int64
        ),
        "selected_core_mechanism_ids": np.asarray(
            [mechanism_lookup[str(row.get("mechanism", "unknown"))] for row in selected_core_records], dtype=np.int64
        ),
        "selected_core_variable_ids": np.asarray(
            [variable_lookup[str(row.get("variable", "unknown"))] for row in selected_core_records], dtype=np.int64
        ),
        "selected_core_kind_ids": np.asarray(
            [kind_lookup[str(row.get("kind", "unknown"))] for row in selected_core_records], dtype=np.int64
        ),
        "selected_privileged_indices": selected_privileged,
        "selected_privileged_segment_ids": np.asarray(
            [layout.privileged_segment_ids[int(index)] for index in selected_privileged], dtype=np.int64
        ),
        "selected_privileged_mechanism_ids": np.asarray(
            [mechanism_lookup[str(row.get("mechanism", "unknown"))] for row in selected_privileged_records], dtype=np.int64
        ),
        "selected_privileged_variable_ids": np.asarray(
            [variable_lookup[str(row.get("variable", "unknown"))] for row in selected_privileged_records], dtype=np.int64
        ),
        "selected_privileged_kind_ids": np.asarray(
            [kind_lookup[str(row.get("kind", "unknown"))] for row in selected_privileged_records], dtype=np.int64
        ),
        "concentration_indices": concentration_indices,
        "concentration_segment_ids": np.asarray(
            [layout.core_segment_ids[int(index)] for index in concentration_indices], dtype=np.int64
        ),
        "synapse_indices": synapse_indices,
        "synapse_segment_ids": np.asarray(
            [layout.core_segment_ids[int(index)] for index in synapse_indices], dtype=np.int64
        ),
        "synapse_channels": np.asarray(synapse_channels, dtype=np.int64),
        "segment_region_ids": np.asarray(layout.segment_region_ids, dtype=np.int64),
        "segment_static": np.asarray(layout.segment_static, dtype=np.float32),
        "event_allowed_mask": event_allowed,
        "category_names": np.asarray(category_names, dtype=object),
        "mechanism_names": np.asarray(mechanism_names, dtype=object),
        "variable_names": np.asarray(variable_names, dtype=object),
        "kind_names": np.asarray(kind_names, dtype=object),
        "region_names": np.asarray(region_names, dtype=object),
    }
    return result


if nn is not None:

    class ResidualLocalBlock(nn.Module):
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


    class TeacherStateEncoder(nn.Module):
        """Semantic token pooling used only to initialise recurrent state."""

        def __init__(self, config: HayFlowHinesConfig, arrays: Mapping[str, np.ndarray]) -> None:
            super().__init__()
            emb = config.token_embedding_dim
            hidden = config.hidden_width
            self.segment_count = int(len(arrays["parent_ids"]))
            self.register_buffer("segment_ids", torch.as_tensor(arrays["core_segment_ids"], dtype=torch.long))
            self.register_buffer("category_ids", torch.as_tensor(arrays["core_category_ids"], dtype=torch.long))
            self.register_buffer("mechanism_ids", torch.as_tensor(arrays["core_mechanism_ids"], dtype=torch.long))
            self.register_buffer("variable_ids", torch.as_tensor(arrays["core_variable_ids"], dtype=torch.long))
            self.register_buffer("kind_ids", torch.as_tensor(arrays["core_kind_ids"], dtype=torch.long))
            self.category_embedding = nn.Embedding(int(np.max(arrays["core_category_ids"])) + 1, emb)
            self.mechanism_embedding = nn.Embedding(len(arrays["mechanism_names"]), emb)
            self.variable_embedding = nn.Embedding(len(arrays["variable_names"]), emb)
            self.kind_embedding = nn.Embedding(len(arrays["kind_names"]), emb)
            self.token = nn.Sequential(
                nn.Linear(1 + 4 * emb, hidden), nn.SiLU(), nn.Linear(hidden, hidden)
            )
            self.local = nn.Sequential(
                nn.LayerNorm(hidden), nn.Linear(hidden, config.local_latent_dim)
            )

        def forward(self, normalized_teacher_state: torch.Tensor) -> torch.Tensor:
            batch = normalized_teacher_state.shape[0]
            token_input = torch.cat(
                [
                    normalized_teacher_state.unsqueeze(-1),
                    self.category_embedding(self.category_ids).unsqueeze(0).expand(batch, -1, -1),
                    self.mechanism_embedding(self.mechanism_ids).unsqueeze(0).expand(batch, -1, -1),
                    self.variable_embedding(self.variable_ids).unsqueeze(0).expand(batch, -1, -1),
                    self.kind_embedding(self.kind_ids).unsqueeze(0).expand(batch, -1, -1),
                ],
                dim=-1,
            )
            tokens = self.token(token_input)
            local_sum = tokens.new_zeros(batch, self.segment_count, tokens.shape[-1])
            local_sum = local_sum.scatter_add(
                1,
                self.segment_ids.view(1, -1, 1).expand(batch, -1, tokens.shape[-1]),
                tokens,
            )
            counts = torch.bincount(self.segment_ids, minlength=self.segment_count).clamp_min(1)
            return self.local(local_sum / counts.view(1, -1, 1))


    class MorphologyEventHeads(nn.Module):
        """One independent head and one dedicated morphology mask per event."""

        def __init__(
            self,
            hidden: int,
            global_dim: int,
            region_count: int,
            event_allowed_mask: np.ndarray,
            *,
            spatial_temperature: float = 0.25,
            boundary_decay_per_ms: float = 4.0,
            boundary_limit_mv: float = 120.0,
        ) -> None:
            super().__init__()
            self.event_count = len(EVENT_KINDS)
            self.region_count = int(region_count)
            self.spatial_temperature = float(spatial_temperature)
            self.boundary_decay_per_ms = float(boundary_decay_per_ms)
            self.boundary_limit_mv = float(boundary_limit_mv)
            self.register_buffer(
                "allowed", torch.as_tensor(event_allowed_mask, dtype=torch.float32)
            )
            self.queries = nn.Parameter(torch.randn(self.event_count, hidden) * 0.02)
            self.heads = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.LayerNorm(hidden + global_dim + 4),
                        nn.Linear(hidden + global_dim + 4, hidden),
                        nn.SiLU(),
                    )
                    for _ in EVENT_KINDS
                ]
            )
            self.presence = nn.ModuleList([nn.Linear(hidden, 1) for _ in EVENT_KINDS])
            self.timing = nn.ModuleList([nn.Linear(hidden, 4) for _ in EVENT_KINDS])
            self.region = nn.ModuleList([nn.Linear(hidden, region_count) for _ in EVENT_KINDS])
            self.amplitude = nn.ModuleList([nn.Linear(hidden, 1) for _ in EVENT_KINDS])
            self.boundary_voltage = nn.ModuleList(
                [nn.Linear(hidden, 1) for _ in EVENT_KINDS]
            )

        def forward(
            self,
            local_features: torch.Tensor,
            global_state: torch.Tensor,
            voltage_t: torch.Tensor,
            voltage_star: torch.Tensor,
        ) -> Dict[str, torch.Tensor]:
            presence = []
            timing = []
            regions = []
            amplitudes = []
            segment_logits = []
            local_gate = []
            boundary_delta = []
            boundary_raw_delta = []
            anchors = torch.stack(
                [
                    voltage_t.mean(1), voltage_t.amax(1),
                    voltage_star.mean(1), voltage_star.amax(1),
                ],
                dim=-1,
            ) / 100.0
            for index in range(self.event_count):
                scores = torch.einsum("bnh,h->bn", local_features, self.queries[index])
                mask = self.allowed[index].view(1, -1)
                masked_scores = scores.masked_fill(mask == 0, -1e4)
                attention = torch.softmax(
                    masked_scores / self.spatial_temperature, dim=1
                )
                pooled = torch.einsum("bn,bnh->bh", attention, local_features)
                hidden = self.heads[index](torch.cat([pooled, global_state, anchors], dim=-1))
                logit = self.presence[index](hidden).squeeze(-1)
                presence.append(logit)
                raw_timing = self.timing[index](hidden)
                decoded_timing = torch.cat(
                    [
                        torch.sigmoid(raw_timing[:, :3]),
                        functional.softplus(raw_timing[:, 3:4]),
                    ],
                    dim=-1,
                )
                timing.append(decoded_timing)
                regions.append(self.region[index](hidden))
                amplitudes.append(functional.softplus(self.amplitude[index](hidden).squeeze(-1)))
                segment_logits.append(masked_scores)
                # A probability distribution sums to one and therefore made a
                # 100 mV spike correction vanish when spread over hundreds of
                # segments.  Preserve the attention for localisation, but
                # normalise its maximum to one for a physical local jump.
                spatial_gate = attention / attention.amax(1, keepdim=True).clamp_min(1e-8)
                local_gate.append(
                    torch.sigmoid(logit).unsqueeze(-1) * spatial_gate
                )
                # The event amplitude and the voltage remaining at t+1 are
                # different quantities.  Decode a signed boundary correction
                # and attenuate events whose predicted offset precedes the
                # macro-step boundary.
                raw_boundary = torch.tanh(
                    self.boundary_voltage[index](hidden).squeeze(-1)
                ) * self.boundary_limit_mv
                remaining = (1.0 - decoded_timing[:, 2]).clamp_min(0.0)
                survival = torch.exp(-self.boundary_decay_per_ms * remaining)
                boundary_raw_delta.append(raw_boundary)
                boundary_delta.append(raw_boundary * survival)
            return {
                "event_logits": torch.stack(presence, dim=1),
                "event_timing": torch.stack(timing, dim=1),
                "event_region_logits": torch.stack(regions, dim=1),
                "event_amplitude": torch.stack(amplitudes, dim=1),
                "event_segment_logits": torch.stack(segment_logits, dim=1),
                "event_local_gate": torch.stack(local_gate, dim=-1),
                "event_boundary_raw_delta_mv": torch.stack(
                    boundary_raw_delta, dim=1
                ),
                "event_boundary_delta_mv": torch.stack(boundary_delta, dim=1),
            }


    class HayFlowHines(nn.Module):
        """Recurrent morphology-aware one-millisecond transition model."""

        def __init__(
            self,
            config: HayFlowHinesConfig,
            metadata: Mapping[str, Any],
            arrays: Mapping[str, np.ndarray],
        ) -> None:
            super().__init__()
            config.validate()
            self.config = config
            self.segment_count = int(metadata["segment_count"])
            self.state_width = int(metadata["state_width"])
            self.region_count = len(metadata["region_names"])
            self.teacher_encoder = TeacherStateEncoder(config, arrays)
            self.register_buffer("parent_ids", torch.as_tensor(arrays["parent_ids"], dtype=torch.long))
            self.register_buffer("child_ids", torch.as_tensor(arrays["child_ids"], dtype=torch.long))
            self.register_buffer("child_mask", torch.as_tensor(arrays["child_mask"], dtype=torch.float32))
            self.register_buffer("child_axial_us", torch.as_tensor(arrays["child_axial_us"], dtype=torch.float32))
            self.register_buffer("segment_static", torch.as_tensor(arrays["segment_static"], dtype=torch.float32))
            self.register_buffer("segment_region_ids", torch.as_tensor(arrays["segment_region_ids"], dtype=torch.long))
            self.register_buffer("capacitance_uf", torch.as_tensor(arrays["capacitance_uf"], dtype=torch.float32))
            self.register_buffer("leak_conductance_us", torch.as_tensor(arrays["leak_conductance_us"], dtype=torch.float32))
            self.register_buffer("leak_reversal_mv", torch.as_tensor(arrays["leak_reversal_mv"], dtype=torch.float32))
            self.register_buffer("axial_conductance_us", torch.as_tensor(arrays["axial_conductance_to_parent_us"], dtype=torch.float32))
            self.register_buffer("axial_total_us", torch.as_tensor(arrays["axial_total_us"], dtype=torch.float32))
            self.register_buffer("selected_core_indices", torch.as_tensor(arrays["selected_core_indices"], dtype=torch.long))
            self.register_buffer("selected_core_segment_ids", torch.as_tensor(arrays["selected_core_segment_ids"], dtype=torch.long))
            self.register_buffer("selected_privileged_indices", torch.as_tensor(arrays["selected_privileged_indices"], dtype=torch.long))
            self.register_buffer("selected_privileged_segment_ids", torch.as_tensor(arrays["selected_privileged_segment_ids"], dtype=torch.long))
            emb = config.token_embedding_dim
            self.selected_mechanism_embedding = nn.Embedding(len(arrays["mechanism_names"]), emb)
            self.selected_variable_embedding = nn.Embedding(len(arrays["variable_names"]), emb)
            self.selected_kind_embedding = nn.Embedding(len(arrays["kind_names"]), emb)
            for name in (
                "selected_core_mechanism_ids", "selected_core_variable_ids", "selected_core_kind_ids",
                "selected_privileged_mechanism_ids", "selected_privileged_variable_ids", "selected_privileged_kind_ids",
            ):
                self.register_buffer(name, torch.as_tensor(arrays[name], dtype=torch.long))
            self.region_embedding = nn.Embedding(self.region_count, config.region_embedding_dim)
            self.synaptic_encoder = nn.Sequential(
                nn.Linear(len(HINES_SYNAPTIC_FEATURE_NAMES), config.synaptic_hidden_width),
                nn.SiLU(),
                nn.Linear(config.synaptic_hidden_width, config.synaptic_hidden_width),
            )
            local_input = (
                1 + config.local_latent_dim + config.global_latent_dim
                + config.synaptic_hidden_width + 4 + config.calcium_state_dim
                + config.synapse_state_dim + arrays["segment_static"].shape[1]
                + config.region_embedding_dim
            )
            self.local_input = nn.Sequential(
                nn.Linear(local_input, config.hidden_width), nn.SiLU(),
                nn.Linear(config.hidden_width, config.hidden_width),
            )
            self.local_blocks = nn.Sequential(
                *[
                    ResidualLocalBlock(config.hidden_width, config.dropout)
                    for _ in range(config.residual_blocks)
                ]
            )
            self.effective_conductance = nn.Linear(config.hidden_width, 1)
            self.source_current = nn.Linear(config.hidden_width, 1)
            self.transition_features = nn.Linear(config.hidden_width, config.hidden_width)
            self.local_event_features = nn.Linear(config.hidden_width, config.hidden_width)
            self.slow_state_delta = nn.Linear(
                config.hidden_width, config.calcium_state_dim + config.synapse_state_dim
            )
            self.continuous_residual = nn.Linear(config.hidden_width, 1)
            self.direct_boundary_residual = nn.Linear(config.hidden_width, 1)
            self.hines = DifferentiableHinesSolve(arrays["parent_ids"])
            global_input = config.local_latent_dim * 3 + 5
            self.global_encoder = nn.Sequential(
                nn.Linear(global_input, config.global_latent_dim), nn.Tanh()
            )
            self.events = MorphologyEventHeads(
                config.hidden_width,
                config.global_latent_dim,
                self.region_count,
                arrays["event_allowed_mask"],
                spatial_temperature=config.event_spatial_temperature,
                boundary_decay_per_ms=config.event_boundary_decay_per_ms,
                boundary_limit_mv=config.event_jump_limit_mv,
            )
            local_commit_input = config.hidden_width + 2 + len(EVENT_KINDS) + config.synaptic_hidden_width
            self.local_commit = nn.GRUCell(local_commit_input, config.local_latent_dim)
            global_commit_input = config.local_latent_dim * 3 + 4 + len(EVENT_KINDS)
            self.global_commit = nn.GRUCell(global_commit_input, config.global_latent_dim)
            selected_input = config.local_latent_dim + 3 * emb
            self.selected_state_decoder = nn.Sequential(
                nn.Linear(selected_input, config.hidden_width), nn.SiLU(), nn.Linear(config.hidden_width, 1)
            )
            self.privileged_decoder = nn.Sequential(
                nn.Linear(selected_input, config.hidden_width), nn.SiLU(), nn.Linear(config.hidden_width, 1)
            )
            self.probe_trace_decoder = nn.Sequential(
                nn.Linear(config.global_latent_dim + 5 * config.local_latent_dim, config.hidden_width),
                nn.SiLU(), nn.Linear(config.hidden_width, 5 * 41),
            )

        def _child_mean(self, values: torch.Tensor) -> torch.Tensor:
            gathered = values[:, self.child_ids]
            count = self.child_mask.sum(1).clamp_min(1.0)
            return (gathered * self.child_mask.unsqueeze(0)).sum(-1) / count.unsqueeze(0)

        def _axial_current(self, voltage: torch.Tensor) -> torch.Tensor:
            parent = self.axial_conductance_us.unsqueeze(0) * (voltage[:, self.parent_ids] - voltage)
            children = self.child_axial_us.unsqueeze(0) * (
                voltage[:, self.child_ids] - voltage.unsqueeze(-1)
            )
            return parent + (children * self.child_mask.unsqueeze(0)).sum(-1)

        def _regional_pool(self, values: torch.Tensor) -> torch.Tensor:
            means = []
            for region in range(self.region_count):
                mask = (self.segment_region_ids == region).to(values.dtype)
                means.append(
                    (values * mask.view(1, -1, 1)).sum(1)
                    / mask.sum().clamp_min(1.0)
                )
            return torch.stack(means, dim=1)

        def initialise(self, batch: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
            local = self.teacher_encoder(batch["teacher_state_t"])
            voltage = batch["voltage_t"]
            calcium = batch["calcium_t"]
            synapse = batch["synapse_state_t"]
            regional = self._regional_pool(local)
            anchors = batch["anchor_voltage_t"] / 100.0
            pooled = torch.cat(
                [local.mean(1), local.amax(1), regional.mean(1), anchors], dim=-1
            )
            # Fixed-width summaries keep the encoder independent of the number
            # of named regions while retaining the five canonical anchors.
            required = self.global_encoder[0].in_features
            if pooled.shape[-1] < required:
                pooled = functional.pad(pooled, (0, required - pooled.shape[-1]))
            global_state = self.global_encoder(pooled[:, :required])
            return {
                "voltage": voltage,
                "local": local,
                "global": global_state,
                "calcium": calcium,
                "synapse": synapse,
            }

        def _decoder_input(self, local: torch.Tensor, privileged: bool) -> torch.Tensor:
            if privileged:
                segments = self.selected_privileged_segment_ids
                mechanisms = self.selected_privileged_mechanism_ids
                variables = self.selected_privileged_variable_ids
                kinds = self.selected_privileged_kind_ids
            else:
                segments = self.selected_core_segment_ids
                mechanisms = self.selected_core_mechanism_ids
                variables = self.selected_core_variable_ids
                kinds = self.selected_core_kind_ids
            batch = local.shape[0]
            return torch.cat(
                [
                    local[:, segments],
                    self.selected_mechanism_embedding(mechanisms).unsqueeze(0).expand(batch, -1, -1),
                    self.selected_variable_embedding(variables).unsqueeze(0).expand(batch, -1, -1),
                    self.selected_kind_embedding(kinds).unsqueeze(0).expand(batch, -1, -1),
                ],
                dim=-1,
            )

        def step(
            self,
            recurrent: Mapping[str, torch.Tensor],
            batch: Mapping[str, torch.Tensor],
            *,
            ablation: str = "H2",
            decode_teacher: bool = True,
            boundary_mode: str = "timed_masked",
        ) -> Dict[str, torch.Tensor]:
            valid_boundary_modes = {
                "timed_masked", "untimed_masked", "no_event_jump",
                "direct_residual",
            }
            if boundary_mode not in valid_boundary_modes:
                raise ValueError(
                    f"unknown boundary_mode {boundary_mode!r}; expected one of "
                    f"{sorted(valid_boundary_modes)}"
                )
            if ablation not in {"H0", "H1", "H2"}:
                raise ValueError("ablation must be H0, H1 or H2")
            voltage = recurrent["voltage"]
            local = recurrent["local"]
            global_state = recurrent["global"]
            calcium = recurrent["calcium"]
            synapse_state = recurrent["synapse"]
            synaptic_features = self.synaptic_encoder(batch["synaptic_features"])
            parent_voltage = voltage[:, self.parent_ids]
            child_voltage = self._child_mean(voltage)
            axial_current = self._axial_current(voltage)
            expanded_global = global_state.unsqueeze(1).expand(-1, self.segment_count, -1)
            region = self.region_embedding(self.segment_region_ids).unsqueeze(0).expand(voltage.shape[0], -1, -1)
            local_input = torch.cat(
                [
                    (voltage / 100.0).unsqueeze(-1), local, expanded_global,
                    synaptic_features, (parent_voltage / 100.0).unsqueeze(-1),
                    (child_voltage / 100.0).unsqueeze(-1), (axial_current / 10.0).unsqueeze(-1),
                    ((parent_voltage - child_voltage) / 100.0).unsqueeze(-1),
                    calcium, synapse_state,
                    self.segment_static.unsqueeze(0).expand(voltage.shape[0], -1, -1), region,
                ],
                dim=-1,
            )
            hidden = self.local_blocks(self.local_input(local_input))
            effective = functional.softplus(self.effective_conductance(hidden).squeeze(-1))
            effective = effective * self.config.effective_conductance_scale_us
            source = torch.tanh(self.source_current(hidden).squeeze(-1)) * self.config.source_current_scale_na
            syn_g = batch["synaptic_conductance_us"].clamp_min(0.0)
            syn_source = batch["synaptic_source_na"] + batch["somatic_current_na"]
            mass = 1000.0 * self.capacitance_uf / self.config.dt_ms
            diagonal = (
                mass.unsqueeze(0) + self.leak_conductance_us.unsqueeze(0)
                + self.axial_total_us.unsqueeze(0) + effective + syn_g
            )
            rhs = (
                mass.unsqueeze(0) * voltage
                + self.leak_conductance_us.unsqueeze(0) * self.leak_reversal_mv.unsqueeze(0)
                + source + syn_source
            )
            voltage_star, hines_diagnostics = self.hines(
                diagonal,
                self.axial_conductance_us.unsqueeze(0).expand_as(diagonal),
                rhs,
                return_diagnostics=True,
            )
            transition = self.transition_features(hidden)
            local_event = self.local_event_features(hidden)
            event_output = self.events(local_event, global_state, voltage, voltage_star)
            continuous = torch.tanh(self.continuous_residual(hidden).squeeze(-1))
            continuous = continuous * self.config.continuous_residual_limit_mv
            if ablation == "H0":
                continuous = torch.zeros_like(continuous)
            jump = torch.zeros_like(voltage)
            raw_boundary_delta = event_output["event_boundary_raw_delta_mv"]
            boundary_delta = (
                event_output["event_boundary_delta_mv"]
                if boundary_mode == "timed_masked" else raw_boundary_delta
            )
            direct_boundary = self.config.event_jump_limit_mv * torch.tanh(
                self.direct_boundary_residual(hidden).squeeze(-1)
            )
            if ablation == "H2" and boundary_mode in {
                "timed_masked", "untimed_masked"
            }:
                gates = event_output["event_local_gate"]
                jump = (gates * boundary_delta.unsqueeze(1)).sum(-1)
                jump = 0.5 * jump + 0.25 * jump[:, self.parent_ids] + 0.25 * self._child_mean(jump)
            elif ablation == "H2" and boundary_mode == "direct_residual":
                jump = direct_boundary
            voltage_next = voltage_star + continuous + jump
            slow = 0.1 * torch.tanh(self.slow_state_delta(hidden))
            calcium_next = calcium + slow[..., : self.config.calcium_state_dim]
            synapse_next = synapse_state + slow[..., self.config.calcium_state_dim :]
            commit_input = torch.cat(
                [
                    transition, (voltage / 100.0).unsqueeze(-1),
                    (voltage_next / 100.0).unsqueeze(-1),
                    event_output["event_local_gate"], synaptic_features,
                ],
                dim=-1,
            )
            local_next = self.local_commit(
                commit_input.reshape(-1, commit_input.shape[-1]),
                local.reshape(-1, local.shape[-1]),
            ).reshape_as(local)
            regional = self._regional_pool(local_next)
            event_probability = torch.sigmoid(event_output["event_logits"])
            global_input = torch.cat(
                [
                    local_next.mean(1), local_next.amax(1), regional.mean(1),
                    torch.stack(
                        [voltage_next.mean(1), voltage_next.amax(1), voltage_next.amin(1), voltage_next.std(1)],
                        dim=-1,
                    ) / 100.0,
                    event_probability,
                ],
                dim=-1,
            )
            global_next = self.global_commit(global_input, global_state)
            result: Dict[str, Any] = {
                "voltage": voltage_next,
                "voltage_star": voltage_star,
                "continuous_residual": continuous,
                "event_jump": jump,
                "direct_boundary_residual": direct_boundary,
                # Diagnostic feature surface used by 05d. Exposing it does
                # not alter the forward path or checkpoint compatibility.
                "boundary_features": hidden,
                "local": local_next,
                "global": global_next,
                "calcium": calcium_next,
                "synapse": synapse_next,
                "effective_conductance_us": effective,
                "effective_source_current_na": source,
                "hines_diagnostics": hines_diagnostics,
                **event_output,
            }
            if decode_teacher:
                result["selected_state"] = self.selected_state_decoder(
                    self._decoder_input(local_next, False)
                ).squeeze(-1)
                if self.selected_privileged_indices.numel():
                    result["selected_privileged"] = self.privileged_decoder(
                        self._decoder_input(local_next, True)
                    ).squeeze(-1)
                anchor_local = batch["anchor_segment_ids"]
                anchor_values = local_next[:, anchor_local].reshape(local_next.shape[0], -1)
                result["probe_microtrace"] = self.probe_trace_decoder(
                    torch.cat([global_next, anchor_values], dim=-1)
                ).view(-1, 5, 41)
            return result

        def forward(
            self,
            batch: Mapping[str, torch.Tensor],
            recurrent: Optional[Mapping[str, torch.Tensor]] = None,
            *,
            ablation: str = "H2",
            decode_teacher: bool = True,
            boundary_mode: str = "timed_masked",
        ) -> Dict[str, torch.Tensor]:
            state = self.initialise(batch) if recurrent is None else dict(recurrent)
            return self.step(
                state, batch, ablation=ablation,
                decode_teacher=decode_teacher, boundary_mode=boundary_mode,
            )


    class OrderedSegmentConvGRU(nn.Module):
        """Conventional fixed-order ConvGRU control with comparable recurrence."""

        def __init__(
            self,
            config: HayFlowHinesConfig,
            metadata: Mapping[str, Any],
            arrays: Mapping[str, np.ndarray],
        ) -> None:
            super().__init__()
            self.config = config
            self.segment_count = int(metadata["segment_count"])
            self.register_buffer("segment_static", torch.as_tensor(arrays["segment_static"], dtype=torch.float32))
            input_width = (
                1 + config.calcium_state_dim + config.synapse_state_dim
                + len(HINES_SYNAPTIC_FEATURE_NAMES) + arrays["segment_static"].shape[1]
            )
            self.input = nn.Conv1d(input_width, config.hidden_width, kernel_size=3, padding=1)
            self.gru = nn.GRUCell(config.hidden_width, config.local_latent_dim)
            self.voltage = nn.Sequential(
                nn.Linear(config.local_latent_dim, config.hidden_width), nn.SiLU(), nn.Linear(config.hidden_width, 1)
            )
            self.event_projection = nn.Linear(config.local_latent_dim, config.hidden_width)
            self.events = MorphologyEventHeads(
                config.hidden_width, config.global_latent_dim,
                len(metadata["region_names"]), arrays["event_allowed_mask"],
                spatial_temperature=config.event_spatial_temperature,
                boundary_decay_per_ms=config.event_boundary_decay_per_ms,
                boundary_limit_mv=config.event_jump_limit_mv,
            )
            self.global_projection = nn.Linear(config.local_latent_dim * 2, config.global_latent_dim)

        def initialise(self, batch: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
            batch_size = batch["voltage_t"].shape[0]
            return {
                "voltage": batch["voltage_t"],
                "local": batch["voltage_t"].new_zeros(
                    batch_size, self.segment_count, self.config.local_latent_dim
                ),
                "global": batch["voltage_t"].new_zeros(batch_size, self.config.global_latent_dim),
                "calcium": batch["calcium_t"],
                "synapse": batch["synapse_state_t"],
            }

        def forward(
            self,
            batch: Mapping[str, torch.Tensor],
            recurrent: Optional[Mapping[str, torch.Tensor]] = None,
            **_: Any,
        ) -> Dict[str, torch.Tensor]:
            state = self.initialise(batch) if recurrent is None else dict(recurrent)
            features = torch.cat(
                [
                    (state["voltage"] / 100.0).unsqueeze(-1), state["calcium"],
                    state["synapse"], batch["synaptic_features"],
                    self.segment_static.unsqueeze(0).expand(state["voltage"].shape[0], -1, -1),
                ],
                dim=-1,
            )
            ordered = functional.silu(self.input(features.transpose(1, 2))).transpose(1, 2)
            local = self.gru(
                ordered.reshape(-1, ordered.shape[-1]),
                state["local"].reshape(-1, state["local"].shape[-1]),
            ).reshape_as(state["local"])
            delta = self.config.convgru_voltage_delta_limit_mv * torch.tanh(
                self.voltage(local).squeeze(-1)
            )
            voltage = state["voltage"] + delta
            global_state = torch.tanh(
                self.global_projection(torch.cat([local.mean(1), local.amax(1)], dim=-1))
            )
            event_output = self.events(
                self.event_projection(local), global_state, state["voltage"], voltage
            )
            return {
                "voltage": voltage,
                "voltage_star": voltage,
                "continuous_residual": delta,
                "event_jump": torch.zeros_like(voltage),
                "local": local,
                "global": global_state,
                "calcium": state["calcium"],
                "synapse": state["synapse"],
                **event_output,
            }


else:

    class HayFlowHines:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            require_torch()


    class OrderedSegmentConvGRU:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            require_torch()


def model_parameter_count(model: Any) -> int:
    require_torch()
    return int(sum(parameter.numel() for parameter in model.parameters()))
