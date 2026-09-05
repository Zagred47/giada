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
REPAIR_SCHEMA_VERSION = "giada-protocol-repair-inputs-v1"
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

# Prospective S1d top-up.  These values are copied from the immutable v1.1.2
# teacher artifacts, rather than inferred from the S1c outcome:
#
# * the calibrated one-pulse somatic boundary is 3 nA (negative) versus 6 nA
#   (positive);
# * the selected BAP assist is the canonical-weight n12/b3/w400 unpaired
#   hot-zone schedule;
# * p2-factor3 and p3-factor3 were the registered soma-only negative/positive
#   BAP bracket, while p3-factor2 is retained as the historical near-boundary
#   assisted arm.
PROTOCOL_REPAIR_SPECS = (
    HybridProtocolSpec("giada_repair_somatic_3na_negative_v1", "somatic_repair", "single_3na_negative", "GIADA_v1_1_2", "validated_hard_negative", 0),
    HybridProtocolSpec("giada_repair_somatic_6na_positive_v1", "somatic_repair", "single_6na_positive", "GIADA_v1_1_2", "validated_positive", 0),
    HybridProtocolSpec("giada_repair_bap_assist_only_n12_b3_w400_v1", "bap_repair_matrix", "assist_only", "GIADA_v1_1_2", "validated_subthreshold_assist", 361),
    HybridProtocolSpec("giada_repair_bap_p2_factor3_soma_only_v1", "bap_repair_matrix", "p2_factor3_soma_only", "GIADA_v1_1_2", "validated_hard_negative", 361),
    HybridProtocolSpec("giada_repair_bap_p2_factor3_combined_v1", "bap_repair_matrix", "p2_factor3_combined", "GIADA_v1_1_2", "factorial_assist_contrast", 361),
    HybridProtocolSpec("giada_repair_bap_p3_factor2_soma_only_v1", "bap_repair_matrix", "p3_factor2_soma_only", "GIADA_v1_1_2", "near_boundary_control", 361),
    HybridProtocolSpec("giada_repair_bap_p3_factor2_combined_v1", "bap_repair_matrix", "p3_factor2_combined", "GIADA_v1_1_2", "historical_assisted_candidate", 361),
    HybridProtocolSpec("giada_repair_bap_p3_factor3_soma_only_v1", "bap_repair_matrix", "p3_factor3_soma_only", "GIADA_v1_1_2", "validated_positive", 361),
    HybridProtocolSpec("giada_repair_bap_p3_factor3_combined_v1", "bap_repair_matrix", "p3_factor3_combined", "GIADA_v1_1_2", "factorial_assist_contrast", 361),
)
PROTOCOL_REPAIR_PROTOCOLS = tuple(row.protocol for row in PROTOCOL_REPAIR_SPECS)
PRODUCTION_BACKGROUND_PROTOCOLS = (
    CANONICAL_NEURONIO_PROTOCOL,
    ACTIVE_BACKGROUND_PROTOCOL,
)

# Only arms whose semantics were confirmed prospectively in S1c/S1d enter
# production.  The matrix retains negative/positive boundaries and paired
# interventions; it does not include the failed S1c somatic/BAP transcriptions.
PRODUCTION_TARGET_PROTOCOLS = (
    "giada_repair_somatic_3na_negative_v1",
    "giada_repair_somatic_6na_positive_v1",
    "giada_hybrid_nmda_n8_v1",
    "giada_hybrid_nmda_n12_v1",
    "giada_hybrid_calcium_unpaired_n12_v1",
    "giada_hybrid_calcium_paired_n8_v1",
    "giada_hybrid_calcium_paired_n12_v1",
    "giada_repair_bap_assist_only_n12_b3_w400_v1",
    "giada_repair_bap_p2_factor3_soma_only_v1",
    "giada_repair_bap_p2_factor3_combined_v1",
    "giada_repair_bap_p3_factor3_soma_only_v1",
    "giada_repair_bap_p3_factor3_combined_v1",
)
_BY_PROTOCOL = {
    row.protocol: row for row in (*HYBRID_PROTOCOL_SPECS, *PROTOCOL_REPAIR_SPECS)
}


def protocol_specs_for_purpose(purpose: str) -> tuple[HybridProtocolSpec, ...]:
    if purpose == "giada_hybrid_pilot":
        return HYBRID_PROTOCOL_SPECS
    if purpose == "giada_protocol_repair_pilot":
        return PROTOCOL_REPAIR_SPECS
    if purpose == "giada_hybrid_production_targeted":
        return tuple(_BY_PROTOCOL[name] for name in PRODUCTION_TARGET_PROTOCOLS)
    raise ValueError(f"no paired GIADA protocol registry for purpose {purpose!r}")


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
    elif spec.family == "somatic_repair":
        amplitude = 3.0 if spec.arm == "single_3na_negative" else 6.0
        schedule = {step: (_current(amplitude),)}
        source_metadata = {
            "somatic_current_na": amplitude,
            "historical_calibration": "somatic_single_spike_current_calibration.json",
        }
    elif spec.family == "bap_repair_matrix":
        assist = _bursts(
            HOT_ZONE_CALCIUM_SYNAPSES,
            start=step,
            count=3,
            window_ms=0.4,
        )
        if spec.arm == "assist_only":
            schedule = assist
            pulse_count = 0
            factor = 0.0
        else:
            tokens = spec.arm.split("_")
            pulse_count = int(tokens[0][1:])
            factor = float(tokens[1].replace("factor", ""))
            soma = {
                step - 1 + pulse_index: (_current(3.0 * factor),)
                for pulse_index in range(pulse_count)
            }
            schedule = (
                soma if spec.arm.endswith("soma_only") else _merge(soma, assist)
            )
        source_metadata = {
            "counterfactual_arm": spec.arm,
            "assist_protocol_id": "targeted_calcium-hot_zone-n12-b3-r1-unpaired-nearest-w400",
            "assist_synapse_ids": list(HOT_ZONE_CALCIUM_SYNAPSES),
            "assist_burst_count": 3,
            "assist_event_window_ms": 0.4,
            "somatic_base_current_na": 3.0,
            "somatic_pulse_count": pulse_count,
            "somatic_current_factor": factor,
        }
    else:  # pragma: no cover - exhaustive registry above
        raise RuntimeError(f"unsupported hybrid family {spec.family!r}")
    for actions in schedule.values():
        for action in actions:
            action.validate()
    metadata = {
        "schema_version": (
            REPAIR_SCHEMA_VERSION
            if spec in PROTOCOL_REPAIR_SPECS
            else HYBRID_SCHEMA_VERSION
        ),
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
