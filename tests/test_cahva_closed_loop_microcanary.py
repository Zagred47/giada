import unittest

from src.giada_teacher.cahva_closed_loop_microcanary import (
    ClosedLoopCaHVAConfig, _injected_density, _pulses, _voltage_step,
)


class ClosedLoopCaHVATests(unittest.TestCase):
    def test_registered_design_and_current_density_conversion(self):
        config = ClosedLoopCaHVAConfig()
        config.validate()
        area = 100.0
        self.assertEqual(_injected_density(0., _pulses("single_pulse"), area), 0.)
        self.assertAlmostEqual(_injected_density(5., _pulses("single_pulse"), area), .003)

    def test_zero_conductance_zero_current_preserves_voltage(self):
        config = ClosedLoopCaHVAConfig()
        self.assertAlmostEqual(_voltage_step(-76., .5, .5, 0., 0., config), -76.)
        self.assertGreater(_voltage_step(-76., .5, .5, 0., .001, config), -76.)


if __name__ == "__main__":
    unittest.main()
