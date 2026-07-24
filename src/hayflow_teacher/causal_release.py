"""Causal observation of the canonical Hay probabilistic synapses.

Instrumentation is implemented with same-time event callbacks around the
original ``NET_RECEIVE`` call.  It does not patch or replace either canonical
NMODL mechanism.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from ..hayflow_data import CausalReleaseOutcome, InputAction


_POINT_STATE_NAMES = {
    "ProbAMPANMDA2": (
        "A_AMPA",
        "B_AMPA",
        "A_NMDA",
        "B_NMDA",
        "g_AMPA",
        "g_NMDA",
        "i_AMPA",
        "i_NMDA",
        "factor_AMPA",
        "factor_NMDA",
        "Use",
        "Dep",
        "Fac",
        "u0",
    ),
    "ProbUDFsyn2": (
        "A",
        "B",
        "g",
        "i",
        "factor",
        "Use",
        "Dep",
        "Fac",
        "u0",
    ),
}

_NETCON_WEIGHT_NAMES = {
    "ProbAMPANMDA2": (
        "weight",
        "weight_AMPA",
        "weight_NMDA",
        "Pv",
        "Pr",
        "u",
        "tsyn_ms",
    ),
    "ProbUDFsyn2": ("weight", "Pv", "Pr", "u", "tsyn_ms"),
}


class CausalReleaseRecorder:
    """Bracket canonical NetCon deliveries and record their direct outcomes."""

    def __init__(
        self,
        session: Any,
        *,
        transition_id: int,
        random123_seed: int,
    ) -> None:
        self.session = session
        self.h = session.h
        self.cvode = session.cvode
        self.transition_id = int(transition_id)
        self.random123_seed = int(random123_seed)
        self._pending: Dict[int, Dict[str, Any]] = {}
        self._outcomes: Dict[int, CausalReleaseOutcome] = {}
        self._callback_references: List[Any] = []

    @staticmethod
    def _weight_values(record: Mapping[str, Any]) -> Dict[str, float]:
        names = _NETCON_WEIGHT_NAMES[str(record["class_name"])]
        netcon = record["netcon"]
        return {
            f"netcon.{name}": float(netcon.weight[index])
            for index, name in enumerate(names)
        }

    @staticmethod
    def _point_values(record: Mapping[str, Any]) -> Dict[str, float]:
        point = record["point_process"]
        values = {}
        for name in _POINT_STATE_NAMES[str(record["class_name"])]:
            if hasattr(point, name):
                values[f"point_process.{name}"] = float(getattr(point, name))
        return values

    def _snapshot_synapse(self, record: Mapping[str, Any]) -> Dict[str, float]:
        return {**self._weight_values(record), **self._point_values(record)}

    def _preview_next_draw(
        self, stream_id: int, sequence_before: float
    ) -> float:
        preview = self.h.Random()
        preview.Random123(self.random123_seed, int(stream_id), 0)
        preview.negexp(1.0)
        preview.seq(float(sequence_before))
        return float(preview.repick())

    @staticmethod
    def _increment(
        before: Mapping[str, float], after: Mapping[str, float], names: Sequence[str]
    ) -> float:
        values = [float(after[name]) - float(before[name]) for name in names]
        if max(values) - min(values) > 1.0e-9:
            raise RuntimeError(
                f"same-time dual-exponential state increments disagree: {values}"
            )
        return float(sum(values) / len(values))

    def _before_group(
        self, scheduled_time_ms: float, events: Sequence[Tuple[int, InputAction]]
    ) -> None:
        if abs(float(self.h.t) - float(scheduled_time_ms)) > 1.0e-9:
            raise RuntimeError("pre-release callback ran at the wrong teacher time")
        for event_index, action in events:
            record = self.session.audit.synapse_records[int(action.synapse_id)]
            sequence = float(record["rng"].seq())
            stream_id = int(record["rng_stream_id"])
            self._pending[event_index] = {
                "action": action,
                "record": record,
                "scheduled_time_ms": float(scheduled_time_ms),
                "pre": self._snapshot_synapse(record),
                "rng_sequence_before": sequence,
                "rng_preview_value": self._preview_next_draw(stream_id, sequence),
            }

    def _after_group(
        self, scheduled_time_ms: float, events: Sequence[Tuple[int, InputAction]]
    ) -> None:
        if abs(float(self.h.t) - float(scheduled_time_ms)) > 1.0e-9:
            raise RuntimeError("post-release callback ran at the wrong teacher time")
        for event_index, action in events:
            pending = self._pending[event_index]
            record = pending["record"]
            before = pending["pre"]
            after = self._snapshot_synapse(record)
            class_name = str(record["class_name"])
            if class_name == "ProbAMPANMDA2":
                ampa = self._increment(
                    before,
                    after,
                    ("point_process.A_AMPA", "point_process.B_AMPA"),
                )
                nmda = self._increment(
                    before,
                    after,
                    ("point_process.A_NMDA", "point_process.B_NMDA"),
                )
                inhibitory = 0.0
                probability = float(after["netcon.Pr"])
            elif class_name == "ProbUDFsyn2":
                ampa = 0.0
                nmda = 0.0
                inhibitory = self._increment(
                    before,
                    after,
                    ("point_process.A", "point_process.B"),
                )
                probability = float(after["netcon.Pr"])
            else:
                raise RuntimeError(f"unsupported probabilistic synapse {class_name}")
            success = any(abs(value) > 1.0e-12 for value in (ampa, nmda, inhibitory))
            predicted = float(pending["rng_preview_value"]) < probability
            if bool(success) != bool(predicted):
                raise RuntimeError(
                    "Random123 preview and direct NET_RECEIVE state jump disagree; "
                    "the causal release instrumentation is not valid"
                )
            sequence_after = float(record["rng"].seq())
            sequence_before = float(pending["rng_sequence_before"])
            outcome = CausalReleaseOutcome(
                transition_id=self.transition_id,
                event_index=int(event_index),
                synapse_id=int(action.synapse_id),
                scheduled_time_ms=float(scheduled_time_ms),
                offset_ms=float(action.offset_ms),
                synapse_type=class_name,
                functional_type=str(record["functional_type"]),
                weight=float(record["netcon"].weight[0]),
                random123_seed=self.random123_seed,
                random123_stream_id=int(record["rng_stream_id"]),
                random123_global_index=int(round(sequence_before)),
                rng_sequence_before=sequence_before,
                rng_sequence_after=sequence_after,
                rng_distribution="negexp(1)",
                rng_preview_value=float(pending["rng_preview_value"]),
                release_probability=probability,
                release_success=bool(success),
                released_quantity=1.0 if success else 0.0,
                ampa_state_increment=ampa,
                nmda_state_increment=nmda,
                inhibitory_state_increment=inhibitory,
                pre_synapse_state=before,
                post_synapse_state=after,
            )
            outcome.validate()
            self._outcomes[event_index] = outcome

    def schedule(
        self, start_time_ms: float, actions: Sequence[InputAction]
    ) -> List[Dict[str, Any]]:
        """Schedule callbacks and original NetCon events in causal order."""

        public_actions: List[Dict[str, Any]] = []
        grouped: Dict[float, List[Tuple[int, InputAction]]] = defaultdict(list)
        seen_synapse_times = set()
        synaptic_index = 0
        for action in actions:
            action.validate()
            item = action.to_dict()
            item.pop("release_observed", None)
            item.pop("rng_sequence_before", None)
            public_actions.append(item)
            if action.kind != "synaptic_event":
                continue
            if abs(float(action.weight_multiplier) - 1.0) > 1.0e-12:
                raise NotImplementedError(
                    "v1.1 preserves canonical NetCon weights; protocol weight "
                    "sweeps must select calibrated canonical synapse groups"
                )
            scheduled = float(start_time_ms) + float(action.offset_ms)
            key = (int(action.synapse_id), scheduled)
            if key in seen_synapse_times:
                raise ValueError(
                    "two events for one synapse at the exact same timestamp cannot "
                    "be attributed independently"
                )
            seen_synapse_times.add(key)
            grouped[scheduled].append((synaptic_index, action))
            synaptic_index += 1

        for scheduled_time, events in sorted(grouped.items()):
            before = lambda t=scheduled_time, rows=tuple(events): self._before_group(t, rows)
            after = lambda t=scheduled_time, rows=tuple(events): self._after_group(t, rows)
            self._callback_references.extend((before, after))
            self.cvode.event(float(scheduled_time), before)
            for _, action in events:
                record = self.session.audit.synapse_records[int(action.synapse_id)]
                record["netcon"].event(float(scheduled_time))
            self.cvode.event(float(scheduled_time), after)
        return public_actions

    def outcomes(self) -> List[CausalReleaseOutcome]:
        if self._pending.keys() != self._outcomes.keys():
            missing = sorted(set(self._pending) - set(self._outcomes))
            raise RuntimeError(f"release callbacks were not completed: {missing}")
        return [self._outcomes[index] for index in sorted(self._outcomes)]
