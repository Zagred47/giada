import unittest
from dataclasses import replace

from src.giada_teacher.cahva_closed_loop_microcanary import (
    ActiveClosedLoopCaHVAConfig, ClosedLoopCaHVAConfig,
    _injected_density, _pulses, _voltage_step,
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

    def test_active_design_is_fixed_and_contains_zero_channel_control(self):
        config = ActiveClosedLoopCaHVAConfig()
        config.validate()
        self.assertEqual(
            len(config.initial_voltage_mv) * len(config.gbar_multipliers)
            * len(config.protocol_names), 24
        )
        self.assertEqual(config.gbar_multipliers, (0., 1., 4.))
        self.assertEqual(_pulses("active_medium"), ((2., 14., .025),))
        self.assertEqual(len(_pulses("active_paired")), 2)
        self.assertEqual(_voltage_step(-30., .5, .5, 0., 0., config),
                         _voltage_step(-30., .0, .0, 0., 0., config))
        with self.assertRaises(ValueError):
            replace(config, current_exposure_threshold_ma_cm2=0.).validate()


if __name__ == "__main__":
    unittest.main()
