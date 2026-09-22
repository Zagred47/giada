import math
import unittest

import numpy as np

from src.giada_teacher.cahva_boundary_semantics_reassessment import (
    BoundarySemanticsConfig,
    corrected_formula_rollout,
)
from src.giada_teacher.cahva_closed_loop_microcanary import (
    ActiveClosedLoopCaHVAConfig,
    _voltage_step,
)


class _VoltageSensitiveRates:
    def rates(self, voltage):
        return {
            "m_inf": 1 / (1 + math.exp(-voltage / 20)),
            "h_inf": 0.5,
            "m_tau_ms": 0.1,
            "h_tau_ms": 0.2,
        }


class BoundarySemanticsTests(unittest.TestCase):
    def test_registered_forensic_design(self):
        BoundarySemanticsConfig().validate()

    def test_voltage_precedes_gate_update_and_current_uses_old_state(self):
        config = ActiveClosedLoopCaHVAConfig()
        teacher = np.zeros((2, 6), dtype=np.float64)
        teacher[0] = [0., -30., .4, .7, 0., 120.]
        area = math.pi * config.section_length_um * config.section_diameter_um
        result = corrected_formula_rollout(
            _VoltageSensitiveRates(), teacher, area, config, 4., "active_low"
        )
        expected_voltage = _voltage_step(-30., .4, .7, 4e-5, 0., config)
        expected_m_inf = _VoltageSensitiveRates().rates(expected_voltage)["m_inf"]
        expected_m = expected_m_inf + (.4 - expected_m_inf) * math.exp(-.025 / .1)
        self.assertAlmostEqual(result[1, 1], expected_voltage)
        self.assertAlmostEqual(result[1, 2], expected_m)
        self.assertAlmostEqual(result[1, 4], 4e-5 * .4**2 * .7 * (-30. - 120.))
        self.assertNotAlmostEqual(result[1, 2], .4)


if __name__ == "__main__":
    unittest.main()
