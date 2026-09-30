"""Offline contract tests for Task 17d attribution and measurement."""

import unittest

import numpy as np

from src.giada_teacher.roadmap_task17d_native_solver_mechanism import (
    Task17dConfig, _crossings, _downsample_dense, _select_calcium_state,
    _set_policy, _state_metrics, _timing_metrics,
)


class FakeCVode:
    def __init__(self):
        self.absolute = 1e-3
        self.relative = 0.0
        self.scales = {"cai_CaDynamics_E2": 1.0, "v": 1.0}

    def atol(self, value=None):
        if value is not None:
            self.absolute = value
        return self.absolute

    def rtol(self, value=None):
        if value is not None:
            self.relative = value
        return self.relative

    def atolscale(self, name, value=None):
        if value is not None:
            self.scales[name] = float(np.float32(value))
        return self.scales[name]

    def re_init(self):
        pass


class NativeSolverMechanismTests(unittest.TestCase):
    def test_policies_reset_prior_scales_and_target_only_one_state(self):
        config = Task17dConfig()
        config.validate()
        session = type("Session", (), {"cvode": FakeCVode()})()
        calcium = _select_calcium_state(["v", "cai_CaDynamics_E2", "m_Ca_HVA"])
        ca = _set_policy(session, "calcium_scaled", calcium, config)
        self.assertTrue(np.isclose(ca["calcium_scale"], 1e-4, rtol=1e-6))
        self.assertEqual(ca["voltage_scale"], 1.0)
        voltage = _set_policy(session, "voltage_scaled", calcium, config)
        self.assertEqual(voltage["calcium_scale"], 1.0)
        self.assertTrue(np.isclose(voltage["voltage_scale"], 1e-2, rtol=1e-6))
        ultra = _set_policy(session, "ultra", calcium, config)
        self.assertEqual(ultra["calcium_scale"], 1.0)
        self.assertEqual(ultra["voltage_scale"], 1.0)
        self.assertEqual(ultra["atol"], 1e-7)

    def test_policy_rejects_materially_wrong_scale(self):
        class WrongScaleCVode(FakeCVode):
            def atolscale(self, name, value=None):
                if value is not None and name == "cai_CaDynamics_E2":
                    value *= 2
                return super().atolscale(name, value)

        session = type("Session", (), {"cvode": WrongScaleCVode()})()
        with self.assertRaisesRegex(RuntimeError, "policy not applied"):
            _set_policy(session, "calcium_scaled", "cai_CaDynamics_E2", Task17dConfig())

    def test_timing_preserves_raw_voltage_and_reports_delay(self):
        a = np.array([-2.0, -1.0, 1.0, 2.0])
        b = np.array([-2.0, -1.0, -0.1, 1.8])
        self.assertAlmostEqual(_crossings(a, 1.0)[0], 1.5)
        self.assertGreater(_crossings(b, 1.0)[0], _crossings(a, 1.0)[0])
        traces_a = {str(site): {"v": np.tile(a, 21).tolist(),
                                "cai_mM": [0.0001] * 84} for site in (0, 387, 460, 469)}
        traces_b = {str(site): {"v": np.tile(b, 21).tolist(),
                                "cai_mM": [0.0002] * 84} for site in (0, 387, 460, 469)}
        ref, candidate = {"traces": traces_a}, {"traces": traces_b}
        timing = _timing_metrics(ref, candidate, 1.0)
        self.assertGreater(timing["0"]["first_crossing_delay_ms"], 0)
        self.assertNotEqual(timing["0"]["peak_reference_mv"],
                            timing["0"]["peak_candidate_mv"])
        self.assertAlmostEqual(_state_metrics(ref, candidate)["0"]["cai_mM"]["maximum_absolute_error"], 1e-4)
        dense = {"traces": {"0": {"v": list(range(11))}}}
        self.assertEqual(_downsample_dense(dense, 5)["traces"]["0"]["v"], [0, 5, 10])


if __name__ == "__main__":
    unittest.main()
