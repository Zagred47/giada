"""Preregistered hybrid GIADA inputs for the RunPod data-quality pilot.

This module deliberately keeps the published NeuronIO sampler as one control
component and adds the causal, paired protocols validated during the GIADA
diagnostic-data programme.  It contains no model code and does not modify the
teacher.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Sequence, Tuple

from src.hayflow_data import InputAction

from .neuronio_inputs import (
    CANONICAL_NEURONIO_PROTOCOL,
    DendriticSynapseMap,
    neuronio_input_config_for_protocol,
    sample_neuronio_actions,
)


HYBRID_SCHEMA_VERSION = "giada-hybrid-inputs-v1"
ACTIVE_BACKGROUND_PROTOCOL = "neuronio_nmda_pilot_ex_high_inh_low_time_fast_v2"

# Exact canonical synapses selected and independently confirmed by the 01b
# calibration.  We preserve their authentic NetCon weights (multiplier 1.0).
TUFT_NMDA_SYNAPSES = (918, 916, 920, 914, 922, 924, 930, 912, 926, 928, 934, 932)
HOT_ZONE_CALCIUM_SYNAPSES = (772, 770, 774, 768, 776, 766, 778, 764, 780, 872, 874, 936)


@dataclass(frozen=True)
class HybridProtocolSpec:
    protocol: str
    family: str
    arm: str
    source: str
    expected_role: str
    target_segment_id: int
    stimulus_step: int = 20


HYBRID_PROTOCOL_SPECS = (
    HybridProtocolSpec("giada_hybrid_background_canonical_v1", "background_canonical", "canonical", "NeuronIO", "stochastic_control", 0),
    HybridProtocolSpec("giada_hybrid_background_active_v1", "background_active", "active_support", "NeuronIO", "stochastic_active_control", 0),
    HybridProtocolSpec("giada_hybrid_nmda_n8_v1", "nmda_boundary", "n8_hard_negative", "GIADA_01b", "candidate_hard_negative", 460),
    HybridProtocolSpec("giada_hybrid_nmda_n12_v1", "nmda_boundary", "n12_positive", "GIADA_01b", "confirmed_nmda_plateau", 460),
    HybridProtocolSpec("giada_hybrid_calcium_unpaired_n12_v1", "calcium_boundary", "unpaired_control", "GIADA_01b", "causal_negative_control", 387),
    HybridProtocolSpec("giada_hybrid_calcium_paired_n8_v1", "calcium_boundary", "paired_n8_control", "GIADA_01c", "candidate_hard_negative", 387),
    HybridProtocolSpec("giada_hybrid_calcium_paired_n12_v1", "calcium_boundary", "paired_n12_positive", "GIADA_01b", "confirmed_calcium_spike", 387),
    HybridProtocolSpec("giada_hybrid_somatic_subthreshold_v1", "somatic_boundary", "subthreshold", "GIADA_01c", "hard_negative", 0),
    HybridProtocolSpec("giada_hybrid_somatic_spike_v1", "somatic_boundary", "calibrated_spike", "GIADA_01b", "positive", 0),
    HybridProtocolSpec("giada_hybrid_bap_soma_only_v1", "bap_counterfactual", "soma_only", "GIADA_01c", "counterfactual_control", 361),
    HybridProtocolSpec("giada_hybrid_bap_assist_only_v1", "bap_counterfactual", "assist_only", "GIADA_01c", "counterfactual_control", 361),
    HybridProtocolSpec("giada_hybrid_bap_combined_v1", "bap_counterfactual", "combined", "GIADA_01c", "candidate_causal_positive", 361),
)
HYBRID_PROTOCOLS = tuple(row.protocol for row in HYBRID_PROTOCOL_SPECS)
_BY_PROTOCOL = {row.protocol: row for row in HYBRID_PROTOCOL_SPECS}


def hybrid_protocol_spec(protocol: str) -> HybridProtocolSpec:
    try:
        return _BY_PROTOCOL[str(protocol)]
    except KeyError as error:
        raise ValueError(f"unknown GIADA hybrid protocol {protocol!r}") from error


def _offsets(count: int, window_ms: float) -> Tuple[float, ...]:
    left = 0.5 - 0.5 * float(window_ms)
    return tuple(left + float(window_ms) * (index + 0.5) / count for index in range(count))


def _bursts(
    synapse_ids: Sequence[int], *, start: int, count: int, window_ms: float
) -> Dict[int, Tuple[InputAction, ...]]:
    offsets = _offsets(len(synapse_ids), window_ms)
    return {
        start + burst: tuple(
            InputAction("synaptic_event", offset, synapse_id=int(synapse_id))
            for synapse_id, offset in zip(synapse_ids, offsets)
        )
        for burst in range(count)
    }


def _current(amplitude_na: float) -> InputAction:
    return InputAction(
        "somatic_current", 0.05, duration_ms=0.9, amplitude_na=float(amplitude_na)
    )


def _merge(*schedules: Mapping[int, Sequence[InputAction]]) -> Dict[int, Tuple[InputAction, ...]]:
    result: Dict[int, list[InputAction]] = {}
    for schedule in schedules:
        for step, actions in schedule.items():
            result.setdefault(int(step), []).extend(actions)
    return {
        step: tuple(
            sorted(
                actions,
                key=lambda row: (
                    row.offset_ms,
                    row.kind,
                    -1 if row.synapse_id is None else row.synapse_id,
                ),
            )
        )
        for step, actions in result.items()
    }


def sample_hybrid_actions(
    duration_ms: int,
    mapping: DendriticSynapseMap,
    *,
    seed: int,
    protocol: str,
) -> tuple[Dict[int, Tuple[InputAction, ...]], Dict[str, Any]]:
    """Materialize one deterministic arm of the hybrid causal design."""

    if int(duration_ms) < 40:
        raise ValueError("hybrid causal episodes require at least 40 ms")
    spec = hybrid_protocol_spec(protocol)
    step = int(spec.stimulus_step)
    if spec.family in {"background_canonical", "background_active"}:
        source_protocol = (
            CANONICAL_NEURONIO_PROTOCOL
            if spec.arm == "canonical"
            else ACTIVE_BACKGROUND_PROTOCOL
        )
        actions, sampled = sample_neuronio_actions(
            duration_ms,
            mapping,
            seed=seed,
            config=neuronio_input_config_for_protocol(source_protocol),
            protocol=source_protocol,
        )
        schedule = actions
        source_metadata: Dict[str, Any] = {"background_sampler": sampled}
    elif spec.family == "nmda_boundary":
        ids = TUFT_NMDA_SYNAPSES[: 8 if spec.arm.startswith("n8") else 12]
        schedule = _bursts(ids, start=step, count=2, window_ms=0.4)
        source_metadata = {"synapse_ids": list(ids), "burst_count": 2}
    elif spec.family == "calcium_boundary":
        ids = HOT_ZONE_CALCIUM_SYNAPSES[: 8 if "n8" in spec.arm else 12]
        dendritic = _bursts(ids, start=step, count=3, window_ms=0.8)
        paired = "paired" in spec.arm and not spec.arm.startswith("unpaired")
        currents = {step - 1: (_current(3.0),), step: (_current(3.0),)} if paired else {}
        schedule = _merge(dendritic, currents)
        source_metadata = {"synapse_ids": list(ids), "burst_count": 3, "paired": paired}
    elif spec.family == "somatic_boundary":
        amplitude = 0.05 if spec.arm == "subthreshold" else 3.0
        schedule = {step: (_current(amplitude),)}
        source_metadata = {"somatic_current_na": amplitude}
    elif spec.family == "bap_counterfactual":
        # The causal arms share the same equilibrium state and Random123 seed.
        # The assist uses a conservative local hot-zone cluster; its biological
        # outcome remains a pilot measurement, not a hard-coded label.
        assist = _bursts(HOT_ZONE_CALCIUM_SYNAPSES[:8], start=step, count=2, window_ms=0.8)
        soma = {step - 1: (_current(3.0),), step: (_current(3.0),)}
        schedule = soma if spec.arm == "soma_only" else assist if spec.arm == "assist_only" else _merge(soma, assist)
        source_metadata = {"counterfactual_arm": spec.arm, "assist_synapse_ids": list(HOT_ZONE_CALCIUM_SYNAPSES[:8])}
    else:  # pragma: no cover - exhaustive registry above
        raise RuntimeError(f"unsupported hybrid family {spec.family!r}")
    for actions in schedule.values():
        for action in actions:
            action.validate()
    metadata = {
        "schema_version": HYBRID_SCHEMA_VERSION,
        "protocol": spec.protocol,
        "family": spec.family,
        "arm": spec.arm,
        "source": spec.source,
        "expected_role": spec.expected_role,
        "target_segment_id": spec.target_segment_id,
        "stimulus_step": spec.stimulus_step,
        "canonical_synaptic_weights_unchanged": True,
        "seed": int(seed),
        **source_metadata,
    }
    return schedule, metadata
