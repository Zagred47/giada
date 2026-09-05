"""Faithful, explicit reimplementation of the NeuronIO NMDA input sampler."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Sequence, Tuple

import numpy as np

from src.hayflow_data import InputAction


CANONICAL_NEURONIO_PROTOCOL = "neuronio_nmda_ergodic_v1"
PILOT_PROTOCOLS = (
    "neuronio_nmda_pilot_ex_full_inh_full_time_broad_v2",
    "neuronio_nmda_pilot_ex_high_inh_full_time_broad_v2",
    "neuronio_nmda_pilot_ex_full_inh_low_time_broad_v2",
    "neuronio_nmda_pilot_ex_high_inh_low_time_broad_v2",
    "neuronio_nmda_pilot_ex_full_inh_full_time_fast_v2",
    "neuronio_nmda_pilot_ex_high_inh_full_time_fast_v2",
    "neuronio_nmda_pilot_ex_full_inh_low_time_fast_v2",
    "neuronio_nmda_pilot_ex_high_inh_low_time_fast_v2",
)


@dataclass(frozen=True)
class NeuronIOInputConfig:
    basal_excitatory_per_100ms: Tuple[float, float] = (0.0, 800.0)
    basal_inhibitory_difference_per_100ms: Tuple[float, float] = (-600.0, 200.0)
    apical_excitatory_per_100ms: Tuple[float, float] = (0.0, 800.0)
    apical_inhibitory_difference_per_100ms: Tuple[float, float] = (-600.0, 200.0)
    rate_intervals_ms: Tuple[int, ...] = (
        25, 30, 35, 40, 45, 55, 60, 65, 70, 75, 80, 85, 90, 100, 150, 200, 300, 450
    )
    smoothing_sigmas_ms: Tuple[int, ...] = (
        25, 30, 35, 40, 50, 60, 80, 100, 150, 200, 300, 400, 500, 600
    )
    interval_jitter_ms: int = 20
    smoothing_jitter_ms: int = 20
    minimum_segment_length_um: float = 10.0
    spatial_multiplier_range: Tuple[float, float] = (0.5, 1.5)

    def validate(self) -> None:
        for label, ex_range, diff_range in (
            (
                "basal",
                self.basal_excitatory_per_100ms,
                self.basal_inhibitory_difference_per_100ms,
            ),
            (
                "apical",
                self.apical_excitatory_per_100ms,
                self.apical_inhibitory_difference_per_100ms,
            ),
        ):
            if ex_range[0] < 0 or ex_range[1] < ex_range[0]:
                raise ValueError(f"{label} excitatory range is invalid")
            if diff_range[1] < diff_range[0]:
                raise ValueError(f"{label} inhibitory-difference range is invalid")
            # The sampler draws inhibition from
            # [max(0, excitation + diff_low), excitation + diff_high].
            # This condition guarantees a non-empty interval even at the
            # minimum admissible excitation.
            if ex_range[0] + diff_range[1] < 0:
                raise ValueError(
                    f"{label} inhibitory range can invert at low excitation"
                )
        if not self.rate_intervals_ms or not self.smoothing_sigmas_ms:
            raise ValueError("NeuronIO temporal option lists cannot be empty")
        if min(self.rate_intervals_ms) <= self.interval_jitter_ms:
            raise ValueError("interval jitter can produce a non-positive interval")
        if min(self.smoothing_sigmas_ms) <= self.smoothing_jitter_ms:
            raise ValueError("smoothing jitter can produce a non-positive sigma")
        if self.minimum_segment_length_um < 0:
            raise ValueError("minimum segment length must be non-negative")


def neuronio_input_config_for_protocol(protocol: str) -> NeuronIOInputConfig:
    """Return a preregistered conditional slice of the NeuronIO support.

    The pilot never changes synaptic locations, weights, mechanisms, release,
    or the teacher.  Each factorial arm only conditions values already inside
    the published NeuronIO ranges.  It is therefore a support diagnostic, not
    a replacement data distribution.
    """

    name = str(protocol)
    if name == CANONICAL_NEURONIO_PROTOCOL:
        return NeuronIOInputConfig()
    if name not in PILOT_PROTOCOLS:
        raise ValueError(f"unknown NeuronIO input protocol {name!r}")
    high_excitation = "_ex_high_" in name
    low_inhibition = "_inh_low_" in name
    fast_temporal = "_time_fast_" in name
    excitatory = (600.0, 800.0) if high_excitation else (0.0, 800.0)
    # A negative upper difference is not valid when the full excitation arm
    # includes zero: it would make the upper inhibitory-rate bound negative.
    # Conditioning the canonical [-600, 200] support to [-600, 0] preserves
    # the intended lower-inhibition contrast while remaining valid and
    # orthogonal to the excitation factor over its complete range.
    inhibitory_difference = (-600.0, 0.0) if low_inhibition else (-600.0, 200.0)
    return NeuronIOInputConfig(
        basal_excitatory_per_100ms=excitatory,
        basal_inhibitory_difference_per_100ms=inhibitory_difference,
        apical_excitatory_per_100ms=excitatory,
        apical_inhibitory_difference_per_100ms=inhibitory_difference,
        rate_intervals_ms=(25, 30, 35, 40, 45)
        if fast_temporal
        else NeuronIOInputConfig.rate_intervals_ms,
        smoothing_sigmas_ms=(25, 30, 35, 40, 50, 60)
        if fast_temporal
        else NeuronIOInputConfig.smoothing_sigmas_ms,
    )


@dataclass(frozen=True)
class DendriticSynapseMap:
    segment_ids: np.ndarray
    segment_lengths_um: np.ndarray
    is_basal: np.ndarray
    excitatory_synapse_ids: np.ndarray
    inhibitory_synapse_ids: np.ndarray

    def validate(self) -> None:
        lengths = {
            len(self.segment_ids), len(self.segment_lengths_um), len(self.is_basal),
            len(self.excitatory_synapse_ids), len(self.inhibitory_synapse_ids),
        }
        if len(lengths) != 1 or not lengths or next(iter(lengths)) <= 0:
            raise ValueError("dendritic synapse map arrays are inconsistent")
        if len(np.unique(self.segment_ids)) != len(self.segment_ids):
            raise ValueError("dendritic segment ids must be unique")
        if np.any(self.segment_lengths_um <= 0):
            raise ValueError("dendritic segment lengths must be positive")
        if not np.any(self.is_basal) or np.all(self.is_basal):
            raise ValueError("map must contain basal and apical segments")


def build_dendritic_synapse_map(session: object) -> DendriticSynapseMap:
    """Bind the original one-excitatory/one-inhibitory synapse per dendrite."""

    records = list(session.audit.synapse_records)
    ex = {int(row["segment_id"]): int(row["synapse_id"]) for row in records if row["class_name"] == "ProbAMPANMDA2"}
    inh = {int(row["segment_id"]): int(row["synapse_id"]) for row in records if row["class_name"] == "ProbUDFsyn2"}
    shared = sorted(set(ex) & set(inh))
    rows = session.audit.segment_df.set_index("segment_id")
    dendritic = [
        segment for segment in shared
        if str(rows.loc[segment, "region"]) not in {"soma", "ais", "axon"}
    ]
    result = DendriticSynapseMap(
        segment_ids=np.asarray(dendritic, dtype=np.int64),
        segment_lengths_um=np.asarray([rows.loc[s, "length_um"] for s in dendritic], dtype=np.float64),
        is_basal=np.asarray([str(rows.loc[s, "region"]) == "basal" for s in dendritic], dtype=bool),
        excitatory_synapse_ids=np.asarray([ex[s] for s in dendritic], dtype=np.int64),
        inhibitory_synapse_ids=np.asarray([inh[s] for s in dendritic], dtype=np.int64),
    )
    result.validate()
    if len(result.segment_ids) != 639:
        raise RuntimeError(f"canonical NeuronIO mapping requires 639 dendritic segments, got {len(result.segment_ids)}")
    return result


def _smooth_same(values: np.ndarray, sigma: int) -> np.ndarray:
    # NumPy FFT implementation of scipy.signal.convolve(..., mode="same").
    # Keeping it local avoids making input planning depend on SciPy while
    # preserving the Gaussian kernel used by the published generator.
    length = 1 + 7 * int(sigma)
    coordinate = np.arange(length, dtype=np.float64) - (length - 1.0) / 2.0
    window = np.exp(-0.5 * (coordinate / float(sigma)) ** 2)
    window /= window.sum()
    full_length = values.shape[1] + length - 1
    fft_length = 1 << (full_length - 1).bit_length()
    transformed = np.fft.rfft(values, n=fft_length, axis=1)
    kernel = np.fft.rfft(window, n=fft_length)
    full = np.fft.irfft(transformed * kernel[None, :], n=fft_length, axis=1)[
        :, :full_length
    ]
    start = (length - 1) // 2
    return full[:, start : start + values.shape[1]]


def sample_neuronio_actions(
    duration_ms: int,
    mapping: DendriticSynapseMap,
    *,
    seed: int,
    config: NeuronIOInputConfig | None = None,
    protocol: str = CANONICAL_NEURONIO_PROTOCOL,
) -> tuple[Dict[int, Tuple[InputAction, ...]], Dict[str, Any]]:
    """Sample ordered millisecond events using the published NMDA methodology.

    The distributional recipe is reproduced exactly; ``Generator`` is used
    instead of NumPy's legacy global RNG, so bit-for-bit identity with a 2021
    file is neither claimed nor required.
    """

    cfg = config or NeuronIOInputConfig()
    cfg.validate()
    mapping.validate()
    rng = np.random.default_rng(int(seed))
    interval = int(rng.choice(cfg.rate_intervals_ms)) + int(
        rng.uniform(-cfg.interval_jitter_ms, cfg.interval_jitter_ms)
    )
    sigma = int(rng.choice(cfg.smoothing_sigmas_ms)) + int(
        rng.uniform(-cfg.smoothing_jitter_ms, cfg.smoothing_jitter_ms)
    )
    samples = int(np.ceil(duration_ms / interval))

    def rates(ex_range: Sequence[float], diff_range: Sequence[float], mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        ex_total = rng.uniform(ex_range[0], ex_range[1], size=(1, samples))
        low = np.maximum(0.0, ex_total + diff_range[0])
        high = ex_total + diff_range[1]
        inh_total = rng.uniform(low, high, size=(1, samples))
        adjusted = cfg.minimum_segment_length_um + mapping.segment_lengths_um[mask]
        ex = np.kron(ex_total / (adjusted.sum() * 100.0), np.ones((len(adjusted), 1)))
        inh = np.kron(inh_total / (adjusted.sum() * 100.0), np.ones((len(adjusted), 1)))
        ex *= rng.uniform(*cfg.spatial_multiplier_range, size=ex.shape) * adjusted[:, None]
        inh *= rng.uniform(*cfg.spatial_multiplier_range, size=inh.shape) * adjusted[:, None]
        ex = np.kron(ex, np.ones((1, interval)))[:, :duration_ms]
        inh = np.kron(inh, np.ones((1, interval)))[:, :duration_ms]
        return _smooth_same(ex, sigma), _smooth_same(inh, sigma)

    basal_ex, basal_inh = rates(
        cfg.basal_excitatory_per_100ms,
        cfg.basal_inhibitory_difference_per_100ms,
        mapping.is_basal,
    )
    apical_ex, apical_inh = rates(
        cfg.apical_excitatory_per_100ms,
        cfg.apical_inhibitory_difference_per_100ms,
        ~mapping.is_basal,
    )
    ex_rate = np.empty((len(mapping.segment_ids), duration_ms), dtype=np.float64)
    inh_rate = np.empty_like(ex_rate)
    ex_rate[mapping.is_basal], inh_rate[mapping.is_basal] = basal_ex, basal_inh
    ex_rate[~mapping.is_basal], inh_rate[~mapping.is_basal] = apical_ex, apical_inh
    # These two draws reproduce the unusual exponential-then-Bernoulli rule in
    # the source generator, rather than silently replacing it with Poisson.
    ex_probability = rng.exponential(scale=ex_rate)
    ex_spikes = rng.random(ex_rate.shape) < ex_probability
    inh_probability = rng.exponential(scale=inh_rate)
    inh_spikes = rng.random(inh_rate.shape) < inh_probability
    actions: Dict[int, Tuple[InputAction, ...]] = {}
    for step in np.flatnonzero(np.any(ex_spikes | inh_spikes, axis=0)):
        rows = []
        for local in np.flatnonzero(ex_spikes[:, step]):
            rows.append(InputAction("synaptic_event", 0.0, synapse_id=int(mapping.excitatory_synapse_ids[local])))
        for local in np.flatnonzero(inh_spikes[:, step]):
            rows.append(InputAction("synaptic_event", 0.0, synapse_id=int(mapping.inhibitory_synapse_ids[local])))
        rows.sort(key=lambda row: int(row.synapse_id))
        actions[int(step)] = tuple(rows)
    metadata = {
        "seed": int(seed),
        "duration_ms": int(duration_ms),
        "rate_interval_ms": int(interval),
        "smoothing_sigma_ms": int(sigma),
        "excitatory_event_count": int(ex_spikes.sum()),
        "inhibitory_event_count": int(inh_spikes.sum()),
        "sampler": "NeuronIO_NMDA_distributional_reproduction_v1",
        "protocol": str(protocol),
    }
    return actions, metadata
