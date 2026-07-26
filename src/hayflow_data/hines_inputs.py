"""Causal structured inputs and explicit state views for HayFlow-Hines."""

from __future__ import annotations

import math
from typing import Any, Dict, Mapping, Sequence, Tuple

import numpy as np


SYNAPTIC_COMPONENTS = ("AMPA", "NMDA", "GABAA", "GABAB")
SYNAPTIC_STATISTICS = (
    "state_increment",
    "end_conductance_us",
    "integrated_conductance_us_ms",
    "event_count",
    "released_count",
    "time_mean_ms",
    "time_second_moment_ms2",
    "time_min_ms",
    "time_max_ms",
    "order_moment",
)
HINES_SYNAPTIC_FEATURE_NAMES = tuple(
    f"{component}.{statistic}"
    for component in SYNAPTIC_COMPONENTS
    for statistic in SYNAPTIC_STATISTICS
) + ("somatic_current_na", "somatic_current_charge_na_ms")


def canonical_anchor_segment_ids(layout: Any) -> np.ndarray:
    """Stable soma/AIS/nexus/tuft/basal anchors for pooling and probes."""

    wanted = (
        ("soma",),
        ("ais", "axon"),
        ("nexus_hot_zone", "nexus", "hot_zone", "apical_trunk"),
        ("tuft", "apical_trunk"),
        ("basal",),
    )
    regions = [str(row["region"]).lower() for row in layout.segments]
    result = []
    for alternatives in wanted:
        match = next(
            (index for name in alternatives for index, region in enumerate(regions) if region == name),
            0,
        )
        result.append(match)
    return np.asarray(result, dtype=np.int64)


def explicit_teacher_views(
    normalized_state: np.ndarray,
    arrays: Mapping[str, np.ndarray],
    *,
    segment_count: int,
    calcium_dim: int = 1,
    synapse_dim: int = 4,
) -> Tuple[np.ndarray, np.ndarray]:
    """Scatter normalized calcium and authentic synapse states to segments."""

    state = np.asarray(normalized_state, dtype=np.float32)
    batch = state.shape[0]
    calcium = np.zeros((batch, segment_count, calcium_dim), dtype=np.float32)
    calcium_count = np.zeros((segment_count, calcium_dim), dtype=np.float32)
    for index, segment in zip(
        arrays["concentration_indices"], arrays["concentration_segment_ids"]
    ):
        calcium[:, int(segment), 0] += state[:, int(index)]
        calcium_count[int(segment), 0] += 1.0
    calcium /= np.maximum(calcium_count[None, :, :], 1.0)
    synapse = np.zeros((batch, segment_count, synapse_dim), dtype=np.float32)
    synapse_count = np.zeros((segment_count, synapse_dim), dtype=np.float32)
    for index, segment, channel in zip(
        arrays["synapse_indices"], arrays["synapse_segment_ids"], arrays["synapse_channels"]
    ):
        channel = min(int(channel), synapse_dim - 1)
        synapse[:, int(segment), channel] += state[:, int(index)]
        synapse_count[int(segment), channel] += 1.0
    synapse /= np.maximum(synapse_count[None, :, :], 1.0)
    return calcium, synapse


def _component_channel(name: str) -> int:
    upper = str(name).upper()
    if "NMDA" in upper:
        return 1
    if "GABAB" in upper or "GABA_B" in upper:
        return 3
    if "GABA" in upper:
        return 2
    return 0


def _dual_exponential(
    increment: float,
    gmax_us: float,
    tau_rise_ms: float,
    tau_decay_ms: float,
    elapsed_ms: float,
) -> Tuple[float, float]:
    if elapsed_ms <= 0.0 or increment == 0.0 or gmax_us == 0.0:
        return 0.0, 0.0
    rise = math.exp(-elapsed_ms / tau_rise_ms)
    decay = math.exp(-elapsed_ms / tau_decay_ms)
    end = gmax_us * increment * (decay - rise)
    integrated = gmax_us * increment * (
        tau_decay_ms * (1.0 - decay) - tau_rise_ms * (1.0 - rise)
    )
    return max(0.0, end), max(0.0, integrated)


def encode_realized_synaptic_drive(
    store: Any,
    indices: Sequence[int],
    voltage_t: np.ndarray,
    *,
    dt_ms: float = 1.0,
    raw_state_t: np.ndarray | None = None,
) -> Dict[str, np.ndarray]:
    """Encode ordered realized events without collapsing receptor components.

    Conductances are analytically propagated from each exact causal state
    increment through the authentic double-exponential component parameters.
    NMDA magnesium block is evaluated causally at ``V_t``.
    """

    layout = store.layout
    batch = len(indices)
    segments = layout.segment_count
    component_count = len(SYNAPTIC_COMPONENTS)
    statistic_count = len(SYNAPTIC_STATISTICS)
    dense = np.zeros((batch, segments, len(HINES_SYNAPTIC_FEATURE_NAMES)), dtype=np.float32)
    conductance = np.zeros((batch, segments), dtype=np.float32)
    source = np.zeros((batch, segments), dtype=np.float32)
    somatic_current = np.zeros((batch, segments), dtype=np.float32)
    synapse_rows = {int(row["id"]): row for row in layout.synapses}
    state_records: Dict[int, Dict[str, int]] = {}
    if raw_state_t is not None:
        for position, record in enumerate(layout.core_records):
            if str(record.get("scope", "")) != "synapse":
                continue
            state_records.setdefault(int(record["owner_id"]), {})[
                str(record.get("variable", record.get("name", "")))
            ] = position
    for batch_index, logical_index in enumerate(indices):
        time_sum = np.zeros((segments, component_count), dtype=np.float64)
        time_square = np.zeros_like(time_sum)
        time_min = np.full_like(time_sum, np.inf)
        time_max = np.zeros_like(time_sum)
        order_sum = np.zeros_like(time_sum)
        weight_sum = np.zeros_like(time_sum)
        # Propagate the authentic point-process A/B state already present at
        # S_t.  This prevents old events from disappearing at macro-step
        # boundaries; new realized increments are added below.
        if raw_state_t is not None:
            for synapse_id, row in synapse_rows.items():
                variables = state_records.get(synapse_id, {})
                segment = int(row["segment_id"])
                parameters = row.get("parameters", {})
                gmax = float(parameters.get("gmax", 0.0) or 0.0)
                for component in row.get("components", []):
                    channel = _component_channel(str(component["name"]))
                    suffix = str(component["name"]).upper()
                    a_name = f"A_{suffix}" if f"A_{suffix}" in variables else "A"
                    b_name = f"B_{suffix}" if f"B_{suffix}" in variables else "B"
                    if a_name not in variables or b_name not in variables:
                        continue
                    a_value = float(raw_state_t[batch_index, variables[a_name]])
                    b_value = float(raw_state_t[batch_index, variables[b_name]])
                    tau_rise = float(component["tau_rise_ms"])
                    tau_decay = float(component["tau_decay_ms"])
                    rise = math.exp(-dt_ms / tau_rise)
                    decay = math.exp(-dt_ms / tau_decay)
                    end = max(0.0, gmax * (b_value * decay - a_value * rise))
                    integrated = max(
                        0.0,
                        gmax
                        * (
                            b_value * tau_decay * (1.0 - decay)
                            - a_value * tau_rise * (1.0 - rise)
                        ),
                    )
                    if bool(component.get("voltage_dependent", False)):
                        alpha = float(component.get("magnesium_alpha") or 0.08)
                        beta = float(component.get("magnesium_beta") or 3.57)
                        block = 1.0 / (
                            1.0
                            + math.exp(
                                -alpha * float(voltage_t[batch_index, segment])
                            )
                            / beta
                        )
                        end *= block
                        integrated *= block
                    base = channel * statistic_count
                    dense[batch_index, segment, base + 1] += end
                    dense[batch_index, segment, base + 2] += integrated
                    conductance[batch_index, segment] += end
                    source[batch_index, segment] += end * float(component["reversal_mv"])
        for order, action in enumerate(store.actions(int(logical_index), "U_realized")):
            if action.get("kind") == "somatic_current":
                amplitude = float(action.get("amplitude_na") or 0.0)
                duration = max(0.0, min(dt_ms - float(action.get("offset_ms", 0.0)), float(action.get("duration_ms") or 0.0)))
                average = amplitude * duration / dt_ms
                somatic_current[batch_index, 0] += average
                dense[batch_index, 0, -2] += amplitude
                dense[batch_index, 0, -1] += amplitude * duration
                continue
            synapse_id = int(action["synapse_id"])
            row = synapse_rows[synapse_id]
            segment = int(row["segment_id"])
            parameters = row.get("parameters", {})
            gmax = float(parameters.get("gmax", action.get("gmax", 0.0)) or 0.0)
            offset = float(action.get("offset_ms", 0.0))
            elapsed = max(0.0, dt_ms - offset)
            released = float(action.get("released_quantity", 0.0))
            for component in row.get("components", []):
                channel = _component_channel(str(component["name"]))
                if channel == 0:
                    increment = float(action.get("ampa_state_increment", 0.0))
                elif channel == 1:
                    increment = float(action.get("nmda_state_increment", 0.0))
                else:
                    increment = float(action.get("inhibitory_state_increment", 0.0))
                end, integrated = _dual_exponential(
                    increment,
                    gmax,
                    float(component["tau_rise_ms"]),
                    float(component["tau_decay_ms"]),
                    elapsed,
                )
                if bool(component.get("voltage_dependent", False)):
                    alpha = float(component.get("magnesium_alpha") or 0.08)
                    beta = float(component.get("magnesium_beta") or 3.57)
                    block = 1.0 / (1.0 + math.exp(-alpha * float(voltage_t[batch_index, segment])) / beta)
                    end *= block
                    integrated *= block
                base = channel * statistic_count
                dense[batch_index, segment, base + 0] += increment
                dense[batch_index, segment, base + 1] += end
                dense[batch_index, segment, base + 2] += integrated
                dense[batch_index, segment, base + 3] += 1.0
                dense[batch_index, segment, base + 4] += released
                moment_weight = max(abs(increment), released, 1e-12)
                time_sum[segment, channel] += moment_weight * offset
                time_square[segment, channel] += moment_weight * offset * offset
                time_min[segment, channel] = min(time_min[segment, channel], offset)
                time_max[segment, channel] = max(time_max[segment, channel], offset)
                order_sum[segment, channel] += moment_weight * float(order)
                weight_sum[segment, channel] += moment_weight
                conductance[batch_index, segment] += end
                source[batch_index, segment] += end * float(component["reversal_mv"])
        for segment in range(segments):
            for channel in range(component_count):
                weight = weight_sum[segment, channel]
                if weight <= 0.0:
                    continue
                base = channel * statistic_count
                dense[batch_index, segment, base + 5] = time_sum[segment, channel] / weight
                dense[batch_index, segment, base + 6] = time_square[segment, channel] / weight
                dense[batch_index, segment, base + 7] = time_min[segment, channel]
                dense[batch_index, segment, base + 8] = time_max[segment, channel]
                dense[batch_index, segment, base + 9] = order_sum[segment, channel] / weight
    return {
        "synaptic_features": dense,
        "synaptic_conductance_us": conductance,
        "synaptic_source_na": source,
        "somatic_current_na": somatic_current,
    }
