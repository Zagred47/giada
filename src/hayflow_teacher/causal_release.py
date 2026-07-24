"""Causal observation of the canonical Hay probabilistic synapses.

The original NMODL mechanisms and their ``NetCon`` deliveries remain the only
objects that drive the teacher.  Before a one-millisecond macro-step, this
module independently evaluates the presynaptic ``NET_RECEIVE`` equations from
the boundary state and cloned Random123 streams.  The resulting release
decisions are therefore available before membrane integration.  At the next
boundary the shadow synapse states and RNG positions are checked against the
authentic teacher.

This design deliberately avoids same-timestamp Python callbacks: NEURON does
not promise that callbacks and ``NetCon`` deliveries with identical times are
executed in the insertion order needed to bracket ``NET_RECEIVE``.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping, MutableMapping, Sequence, Tuple

from ..hayflow_data import CausalReleaseOutcome, InputAction


_DYNAMIC_POINT_STATE_NAMES = {
    "ProbAMPANMDA2": ("A_AMPA", "B_AMPA", "A_NMDA", "B_NMDA"),
    "ProbUDFsyn2": ("A", "B"),
}

_POINT_PARAMETER_NAMES = {
    "ProbAMPANMDA2": (
        "tau_r_AMPA",
        "tau_d_AMPA",
        "tau_r_NMDA",
        "tau_d_NMDA",
        "Use",
        "Dep",
        "Fac",
        "u0",
    ),
    "ProbUDFsyn2": (
        "tau_r",
        "tau_d",
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

_DECAY_CONSTANTS = {
    "ProbAMPANMDA2": {
        "A_AMPA": "tau_r_AMPA",
        "B_AMPA": "tau_d_AMPA",
        "A_NMDA": "tau_r_NMDA",
        "B_NMDA": "tau_d_NMDA",
    },
    "ProbUDFsyn2": {"A": "tau_r", "B": "tau_d"},
}


class CausalReleaseRecorder:
    """Run a causal shadow front-end and verify it against the teacher."""

    POINT_STATE_ATOL = 1.0e-6
    NETCON_STATE_ATOL = 1.0e-10

    def __init__(
        self,
        session: Any,
        *,
        transition_id: int,
        random123_seed: int,
    ) -> None:
        self.session = session
        self.h = session.h
        self.transition_id = int(transition_id)
        self.random123_seed = int(random123_seed)
        self._outcomes: List[CausalReleaseOutcome] = []
        self._shadows: Dict[int, Dict[str, Any]] = {}
        self._scheduled = False
        self._verified = False
        self.verification_report: Dict[str, Any] = {}

    @staticmethod
    def _weight_values(record: Mapping[str, Any]) -> List[float]:
        names = _NETCON_WEIGHT_NAMES[str(record["class_name"])]
        return [float(record["netcon"].weight[index]) for index in range(len(names))]

    @staticmethod
    def _point_values(record: Mapping[str, Any]) -> Dict[str, float]:
        class_name = str(record["class_name"])
        point = record["point_process"]
        names = (
            *_DYNAMIC_POINT_STATE_NAMES[class_name],
            *_POINT_PARAMETER_NAMES[class_name],
        )
        values = {name: float(getattr(point, name)) for name in names}
        # The normalization factors are ASSIGNED variables, not RANGE
        # variables, in the canonical MOD files and are therefore not exposed
        # through the Python point-process object.  Recompute the exact INITIAL
        # expression without modifying the teacher.
        if class_name == "ProbAMPANMDA2":
            values["factor_AMPA"] = CausalReleaseRecorder._dual_exp_factor(
                values["tau_r_AMPA"], values["tau_d_AMPA"]
            )
            values["factor_NMDA"] = CausalReleaseRecorder._dual_exp_factor(
                values["tau_r_NMDA"], values["tau_d_NMDA"]
            )
        else:
            values["factor"] = CausalReleaseRecorder._dual_exp_factor(
                values["tau_r"], values["tau_d"]
            )
        return values

    @staticmethod
    def _dual_exp_factor(tau_rise_ms: float, tau_decay_ms: float) -> float:
        tau_rise = float(tau_rise_ms)
        tau_decay = float(tau_decay_ms)
        if not 0.0 < tau_rise < tau_decay:
            raise RuntimeError(
                "canonical dual-exponential synapse requires 0 < tau_rise < tau_decay"
            )
        time_to_peak = (
            tau_rise
            * tau_decay
            / (tau_decay - tau_rise)
            * math.log(tau_decay / tau_rise)
        )
        return 1.0 / (
            -math.exp(-time_to_peak / tau_rise)
            + math.exp(-time_to_peak / tau_decay)
        )

    @staticmethod
    def _global_index(rng: Any) -> int:
        value = int(rng.Random123_globalindex())
        if value < 0:
            raise RuntimeError("Random123 global index must be non-negative")
        return value

    def _make_shadow(self, record: Mapping[str, Any], start_time_ms: float) -> Dict[str, Any]:
        stream_id = int(record["rng_stream_id"])
        sequence = float(record["rng"].seq())
        preview = self.h.Random()
        preview.Random123(self.random123_seed, stream_id, 0)
        preview.negexp(1.0)
        preview.seq(sequence)
        return {
            "record": record,
            "class_name": str(record["class_name"]),
            "time_ms": float(start_time_ms),
            "point": self._point_values(record),
            "weights": self._weight_values(record),
            "rng": preview,
            "global_index": self._global_index(record["rng"]),
        }

    @staticmethod
    def _advance_point(shadow: MutableMapping[str, Any], target_time_ms: float) -> None:
        target = float(target_time_ms)
        delta = target - float(shadow["time_ms"])
        if delta < -1.0e-12:
            raise RuntimeError("causal synapse events are not time ordered")
        if delta > 0.0:
            point = shadow["point"]
            for state_name, tau_name in _DECAY_CONSTANTS[shadow["class_name"]].items():
                point[state_name] *= math.exp(-delta / point[tau_name])
        shadow["time_ms"] = target

    @staticmethod
    def _state_view(shadow: Mapping[str, Any]) -> Dict[str, float]:
        class_name = str(shadow["class_name"])
        names = _NETCON_WEIGHT_NAMES[class_name]
        values = {
            f"netcon.{name}": float(value)
            for name, value in zip(names, shadow["weights"])
        }
        values.update(
            {
                f"point_process.{name}": float(value)
                for name, value in shadow["point"].items()
            }
        )
        return values

    @staticmethod
    def _apply_short_term_plasticity(shadow: MutableMapping[str, Any], event_time_ms: float) -> float:
        class_name = str(shadow["class_name"])
        point = shadow["point"]
        weights = shadow["weights"]
        if class_name == "ProbAMPANMDA2":
            weights[1] = weights[0]
            weights[2] = weights[0]
            pv_index, pr_index, u_index, tsyn_index = 3, 4, 5, 6
        elif class_name == "ProbUDFsyn2":
            pv_index, pr_index, u_index, tsyn_index = 1, 2, 3, 4
        else:
            raise RuntimeError(f"unsupported probabilistic synapse {class_name}")

        elapsed = float(event_time_ms) - float(weights[tsyn_index])
        if elapsed < -1.0e-12:
            raise RuntimeError("NET_RECEIVE tsyn lies after the scheduled event")
        if point["Fac"] > 0.0:
            u_value = weights[u_index] * math.exp(-elapsed / point["Fac"])
            u_value += point["Use"] * (1.0 - u_value)
        else:
            u_value = point["Use"]
        pv_available = 1.0 - (1.0 - weights[pv_index]) * math.exp(
            -elapsed / point["Dep"]
        )
        probability = u_value * pv_available
        weights[pv_index] = pv_available - u_value * pv_available
        weights[pr_index] = probability
        weights[u_index] = u_value
        weights[tsyn_index] = float(event_time_ms)
        return float(probability)

    @staticmethod
    def _apply_release(shadow: MutableMapping[str, Any], success: bool) -> Tuple[float, float, float]:
        if not success:
            return 0.0, 0.0, 0.0
        point = shadow["point"]
        weights = shadow["weights"]
        if shadow["class_name"] == "ProbAMPANMDA2":
            ampa = float(weights[1] * point["factor_AMPA"])
            nmda = float(weights[2] * point["factor_NMDA"])
            point["A_AMPA"] += ampa
            point["B_AMPA"] += ampa
            point["A_NMDA"] += nmda
            point["B_NMDA"] += nmda
            return ampa, nmda, 0.0
        inhibitory = float(weights[0] * point["factor"])
        point["A"] += inhibitory
        point["B"] += inhibitory
        return 0.0, 0.0, inhibitory

    def _plan_event(
        self,
        event_index: int,
        action: InputAction,
        scheduled_time_ms: float,
    ) -> None:
        synapse_id = int(action.synapse_id)
        shadow = self._shadows[synapse_id]
        self._advance_point(shadow, scheduled_time_ms)
        pre = self._state_view(shadow)
        sequence_before = float(shadow["rng"].seq())
        probability = self._apply_short_term_plasticity(shadow, scheduled_time_ms)
        draw = float(shadow["rng"].repick())
        sequence_after = float(shadow["rng"].seq())
        success = draw < probability
        ampa, nmda, inhibitory = self._apply_release(shadow, success)
        post = self._state_view(shadow)
        record = shadow["record"]
        outcome = CausalReleaseOutcome(
            transition_id=self.transition_id,
            event_index=int(event_index),
            synapse_id=synapse_id,
            scheduled_time_ms=float(scheduled_time_ms),
            offset_ms=float(action.offset_ms),
            synapse_type=str(record["class_name"]),
            functional_type=str(record["functional_type"]),
            weight=float(shadow["weights"][0]),
            random123_seed=self.random123_seed,
            random123_stream_id=int(record["rng_stream_id"]),
            random123_global_index=int(shadow["global_index"]),
            rng_sequence_before=sequence_before,
            rng_sequence_after=sequence_after,
            rng_distribution="negexp(1)",
            rng_preview_value=draw,
            release_probability=probability,
            release_success=bool(success),
            released_quantity=1.0 if success else 0.0,
            ampa_state_increment=ampa,
            nmda_state_increment=nmda,
            inhibitory_state_increment=inhibitory,
            pre_synapse_state=pre,
            post_synapse_state=post,
        )
        outcome.validate()
        self._outcomes.append(outcome)

    def schedule(
        self, start_time_ms: float, actions: Sequence[InputAction]
    ) -> List[Dict[str, Any]]:
        """Plan causal outcomes, then queue only the original NetCon events."""

        if self._scheduled:
            raise RuntimeError("causal release recorder may be scheduled only once")
        public_actions: List[Dict[str, Any]] = []
        synaptic_events: List[Tuple[int, int, InputAction, float]] = []
        seen_synapse_times = set()
        synaptic_index = 0
        for action_order, action in enumerate(actions):
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
            synapse_id = int(action.synapse_id)
            scheduled = float(start_time_ms) + float(action.offset_ms)
            key = (synapse_id, scheduled)
            if key in seen_synapse_times:
                raise ValueError(
                    "two events for one synapse at the exact same timestamp cannot "
                    "be attributed independently"
                )
            seen_synapse_times.add(key)
            if synapse_id not in self._shadows:
                record = self.session.audit.synapse_records[synapse_id]
                self._shadows[synapse_id] = self._make_shadow(record, start_time_ms)
            synaptic_events.append((synaptic_index, action_order, action, scheduled))
            synaptic_index += 1

        # Evaluate the presynaptic front-end before the membrane macro-step.
        # Input order breaks ties exactly as it does when events are enqueued.
        for event_index, _, action, scheduled in sorted(
            synaptic_events, key=lambda row: (row[3], row[1])
        ):
            self._plan_event(event_index, action, scheduled)

        # Queue the authentic teacher events in the original action order.
        for _, _, action, scheduled in sorted(synaptic_events, key=lambda row: row[1]):
            record = self.session.audit.synapse_records[int(action.synapse_id)]
            record["netcon"].event(float(scheduled))
        self._outcomes.sort(key=lambda row: row.event_index)
        self._scheduled = True
        return public_actions

    def verify_boundary(self, boundary_time_ms: float) -> Dict[str, Any]:
        """Validate shadow states and RNG positions after teacher integration."""

        if not self._scheduled:
            raise RuntimeError("release outcomes must be planned before verification")
        point_errors: List[Dict[str, Any]] = []
        weight_errors: List[Dict[str, Any]] = []
        rng_errors: List[Dict[str, Any]] = []
        maximum_point_error = 0.0
        maximum_weight_error = 0.0
        for synapse_id, shadow in sorted(self._shadows.items()):
            self._advance_point(shadow, boundary_time_ms)
            record = shadow["record"]
            point = record["point_process"]
            for name in _DYNAMIC_POINT_STATE_NAMES[shadow["class_name"]]:
                predicted = float(shadow["point"][name])
                observed = float(getattr(point, name))
                error = abs(predicted - observed)
                maximum_point_error = max(maximum_point_error, error)
                if error > self.POINT_STATE_ATOL:
                    point_errors.append(
                        {
                            "synapse_id": synapse_id,
                            "variable": name,
                            "predicted": predicted,
                            "observed": observed,
                            "absolute_error": error,
                        }
                    )
            observed_weights = self._weight_values(record)
            for name, predicted, observed in zip(
                _NETCON_WEIGHT_NAMES[shadow["class_name"]],
                shadow["weights"],
                observed_weights,
            ):
                error = abs(float(predicted) - float(observed))
                maximum_weight_error = max(maximum_weight_error, error)
                if error > self.NETCON_STATE_ATOL:
                    weight_errors.append(
                        {
                            "synapse_id": synapse_id,
                            "variable": name,
                            "predicted": float(predicted),
                            "observed": float(observed),
                            "absolute_error": error,
                        }
                    )
            predicted_sequence = float(shadow["rng"].seq())
            observed_sequence = float(record["rng"].seq())
            if predicted_sequence != observed_sequence:
                rng_errors.append(
                    {
                        "synapse_id": synapse_id,
                        "predicted": predicted_sequence,
                        "observed": observed_sequence,
                    }
                )
        valid = not point_errors and not weight_errors and not rng_errors
        self.verification_report = {
            "valid": valid,
            "verified_synapse_count": len(self._shadows),
            "maximum_point_state_error": maximum_point_error,
            "point_state_atol": self.POINT_STATE_ATOL,
            "maximum_netcon_state_error": maximum_weight_error,
            "netcon_state_atol": self.NETCON_STATE_ATOL,
            "point_state_mismatches": point_errors[:8],
            "netcon_state_mismatches": weight_errors[:8],
            "rng_sequence_mismatches": rng_errors[:8],
        }
        self._verified = True
        if not valid:
            raise RuntimeError(
                "causal shadow NET_RECEIVE disagrees with the authentic teacher "
                f"at the 1 ms boundary: {self.verification_report}"
            )
        return self.verification_report

    def outcomes(self) -> List[CausalReleaseOutcome]:
        if self._shadows and not self._verified:
            raise RuntimeError("causal release outcomes were not boundary-verified")
        return list(self._outcomes)
