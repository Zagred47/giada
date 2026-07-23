import unittest

from src.hayflow_data import (
    BurnInCriteria,
    InputAction,
    ProtocolTrajectory,
    estimate_dataset_size_bytes,
    stable_split,
    validate_split_isolation,
)
from src.hayflow_teacher.event_extractor import (
    EventDefinition,
    event_ids_by_transition,
    extract_events,
)
from src.hayflow_teacher.diagnostic_dataset import (
    DiagnosticDatasetSession,
    expected_audit_hashes,
)


class DiagnosticContractTest(unittest.TestCase):
    def test_input_offsets_are_half_open_and_ordered(self):
        InputAction("synaptic_event", 0.999, synapse_id=2).validate()
        with self.assertRaisesRegex(ValueError, r"\[0, step_ms\)"):
            InputAction("synaptic_event", 1.0, synapse_id=2).validate()
        with self.assertRaisesRegex(ValueError, "time ordered"):
            ProtocolTrajectory(
                "trajectory",
                "local_synaptic",
                "probe",
                1,
                2,
                "train",
                actions_by_step={
                    0: (
                        InputAction("synaptic_event", 0.8, synapse_id=2),
                        InputAction("synaptic_event", 0.2, synapse_id=2),
                    )
                },
            ).validate()

    def test_split_is_stable_and_rejects_leakage(self):
        self.assertEqual(stable_split(7, "probe"), stable_split(7, "probe"))
        with self.assertRaisesRegex(ValueError, "seed/protocol"):
            validate_split_isolation(
                [
                    {
                        "trajectory_id": "a",
                        "seed": 7,
                        "protocol": "probe",
                        "split": "train",
                    },
                    {
                        "trajectory_id": "b",
                        "seed": 7,
                        "protocol": "probe",
                        "split": "test",
                    },
                ]
            )

    def test_burnin_and_size_contracts_are_explicit(self):
        BurnInCriteria().validate()
        with self.assertRaisesRegex(ValueError, "maximum_duration"):
            BurnInCriteria(consecutive_ms=20, maximum_duration_ms=10).validate()
        estimate = estimate_dataset_size_bytes(10, 100, 20, 41, 642)
        self.assertEqual(
            estimate["estimated_uncompressed_bytes_for_dataset"],
            10 * estimate["estimated_uncompressed_bytes_per_transition"],
        )
        with_current = estimate_dataset_size_bytes(
            10, 100, 20, 41, 642, microtrace_scalar_count=1
        )
        self.assertEqual(
            with_current["estimated_uncompressed_bytes_per_transition"]
            - estimate["estimated_uncompressed_bytes_per_transition"],
            41 * 4,
        )
        self.assertEqual(
            estimate[
                "estimated_uncompressed_boundary_bytes_per_transition"
            ],
            2 * 100 * 8,
        )

    def test_somatic_current_uses_native_iclamp_timing(self):
        class FakeClamp:
            delay = None
            dur = None
            amp = None

        session = object.__new__(DiagnosticDatasetSession)
        session.somatic_clamp = FakeClamp()
        action = InputAction(
            "somatic_current",
            0.125,
            duration_ms=0.75,
            amplitude_na=1.5,
        )
        session._configure_somatic_current(100.0, [action])
        self.assertEqual(session.somatic_clamp.delay, 100.125)
        self.assertEqual(session.somatic_clamp.dur, 0.75)
        self.assertEqual(session.somatic_clamp.amp, 1.5)
        session._disable_somatic_clamp()
        self.assertEqual(session.somatic_clamp.dur, 0.0)
        self.assertEqual(session.somatic_clamp.amp, 0.0)

    def test_somatic_driver_rejects_two_pulses_in_one_macro_step(self):
        session = object.__new__(DiagnosticDatasetSession)
        session.somatic_clamp = type("Clamp", (), {})()
        actions = [
            InputAction(
                "somatic_current", 0.1, duration_ms=0.2, amplitude_na=1.0
            ),
            InputAction(
                "somatic_current", 0.5, duration_ms=0.2, amplitude_na=1.0
            ),
        ]
        with self.assertRaisesRegex(NotImplementedError, "at most one"):
            session._configure_somatic_current(0.0, actions)

    def test_audited_teacher_hash_contract_is_complete(self):
        hashes = expected_audit_hashes()
        self.assertEqual(len(hashes), 22)
        self.assertTrue(
            all(
                len(value) == 64
                and all(character in "0123456789abcdef" for character in value)
                for value in hashes.values()
            )
        )

    def test_random123_restore_includes_key_distribution_and_sequence(self):
        calls = []

        class FakeRandom:
            def Random123(self, first, second, third):
                calls.append(("key", first, second, third))

            def negexp(self, mean):
                calls.append(("distribution", mean))

            def seq(self, sequence):
                calls.append(("sequence", sequence))

        session = object.__new__(DiagnosticDatasetSession)
        session.audit = type(
            "Audit", (), {"synapse_rngs": [FakeRandom(), FakeRandom()]}
        )()
        session._configure_rngs(1234, [7.0, 11.0])

        self.assertEqual(
            calls,
            [
                ("key", 1234, 0, 0),
                ("distribution", 1.0),
                ("sequence", 7.0),
                ("key", 1234, 1, 0),
                ("distribution", 1.0),
                ("sequence", 11.0),
            ],
        )

    def test_event_extractor_preserves_full_event_duration(self):
        time = [index * 0.25 for index in range(13)]
        trace = [-70, -70, -30, -20, -10, -20, -30, -55, -70, -70, -70, -70, -70]
        definition = EventDefinition(
            "nmda_spike",
            "nexus",
            12,
            "nexus",
            threshold=-40,
            reset_threshold=-50,
            min_duration_ms=1.0,
        )
        events = extract_events(time, {"nexus": trace}, [definition])
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["onset_ms"], 0.5)
        self.assertEqual(events[0]["offset_ms"], 1.75)
        self.assertEqual(event_ids_by_transition(events, [0.0, 1.0]), [[0], []])

    def test_linked_bap_requires_a_somatic_event(self):
        time = [index * 0.25 for index in range(9)]
        definitions = [
            EventDefinition("somatic_spike", "soma", 0, "soma", 0, -20),
            EventDefinition(
                "backpropagating_ap",
                "trunk",
                1,
                "apical_trunk",
                -20,
                -40,
                requires_kind="somatic_spike",
                maximum_delay_ms=1.0,
            ),
        ]
        traces = {
            "soma": [-70, 10, -70, -70, -70, -70, -70, -70, -70],
            "trunk": [-70, -70, -10, -70, -70, -70, -70, -70, -70],
        }
        events = extract_events(time, traces, definitions)
        self.assertEqual([event["kind"] for event in events], [
            "somatic_spike", "backpropagating_ap"
        ])


if __name__ == "__main__":
    unittest.main()
