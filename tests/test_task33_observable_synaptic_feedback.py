import unittest
from pathlib import Path

import numpy as np

from src.giada_teacher.task33_observable_synaptic_feedback import (
    _candidate_pass, _floor_pass, cases, contract, receptor_conductance,
    voltage_step_with_synapses,
)


ROOT = Path(__file__).resolve().parents[1]


class TestTask33Contract(unittest.TestCase):
    def test_registered_parent_and_matrix(self):
        cfg, t32, c3 = contract(ROOT)
        self.assertEqual(len(cases(cfg)), 16)
        self.assertEqual(t32['dt_ms'], cfg['dt_ms'])
        self.assertEqual(len(c3['synapses']), 4)
        self.assertEqual(cfg['synaptic_state_phase'], 'old')
        self.assertEqual(cfg['calibration']['seed'], 83103)
        self.assertEqual(cfg['confirmation']['seeds'], [83317, 83329])
        self.assertTrue(set(cfg['confirmation']['seeds']).isdisjoint(
            {cfg['calibration']['seed'], 81103, 82317, 82329}))
        for schedule in cfg['schedules_ms'].values():
            for times in schedule.values():
                for time_ms in times:
                    self.assertAlmostEqual(time_ms / cfg['dt_ms'],
                                           round(time_ms / cfg['dt_ms']))

    def test_nmda_block_and_zero_input_voltage(self):
        source = np.array([0., 1., 0., 0.])
        self.assertLess(receptor_conductance(source, -75.)[1],
                        receptor_conductance(source, 20.)[1])
        state = np.zeros((1, 18))
        base = {'cm_uf_cm2': 1., 'g_pas_s_cm2': 0., 'e_pas_mv': -65.}
        result = voltage_step_with_synapses(np.array([-65.]), state,
                   np.zeros((1, 11)), np.array([0.]), np.zeros((1, 4)),
                   314.159, base, .1)
        np.testing.assert_allclose(result, [-65.], atol=1e-12)

    def test_all_registered_gates_are_active(self):
        cfg, _, _ = contract(ROOT)
        row = {'pooled_voltage_rmse_mv': 0., 'worst_voltage_rmse_mv': 0.,
               'worst_cai_rmse_mM': 0., 'finite': True,
               'occupancy_violations': 0, 'physical_voltage_violations': 0}
        self.assertTrue(_floor_pass(row, cfg))
        self.assertTrue(_candidate_pass(row, cfg))
        for key in ('pooled_voltage_rmse_mv', 'worst_voltage_rmse_mv', 'worst_cai_rmse_mM'):
            bad = dict(row)
            bad[key] = 100.
            self.assertFalse(_floor_pass(bad, cfg))
            self.assertFalse(_candidate_pass(bad, cfg))


if __name__ == '__main__':
    unittest.main()
