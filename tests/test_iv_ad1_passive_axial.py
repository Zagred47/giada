"""Pure-math preflight for the preregistered passive axial interface."""

import unittest
import math
from pathlib import Path

import numpy as np

from src.giada_teacher import iv_ad1_passive_axial as passive


ROOT = Path(__file__).resolve().parents[1]


class PassiveAxialTests(unittest.TestCase):
    def test_units_and_reciprocal_axial_current(self):
        cfg = passive.config(ROOT)
        geometry = cfg['geometries'][0]
        cap, leak, axial = passive.coefficients(geometry, cfg['passive_membrane'], True)
        self.assertAlmostEqual(cap[0], np.pi*100*1*1e-5)
        self.assertAlmostEqual(axial, 0.005235987755982989)
        self.assertGreater(axial, 0)
        old = np.array([-80., -60.])
        new = passive.passive_step(old, cap, leak, axial, -76., np.array([.002, 0.]), .1)
        current = axial*(new[0]-new[1])
        residual = cap*(new-old)/.1 + leak*(new+76.) + np.array([current, -current])-np.array([.002, 0.])
        self.assertLess(np.max(np.abs(residual)), 1e-13)

    def test_disconnected_and_symmetric_controls(self):
        cfg = passive.config(ROOT)
        cap, leak, axial = passive.coefficients(cfg['geometries'][0], cfg['passive_membrane'], False)
        self.assertEqual(axial, 0)
        after = passive.passive_step(np.array([-80., -60.]), cap, leak, axial, -76., np.array([.002, 0.]), .1)
        child_alone = passive.passive_step(np.array([-80., -60.]), cap, leak, axial, -76., np.zeros(2), .1)
        self.assertAlmostEqual(after[1], child_alone[1])
        cap, leak, axial = passive.coefficients(cfg['geometries'][0], cfg['passive_membrane'], True)
        same = passive.passive_step(np.array([-70., -70.]), cap, leak, axial, -76., np.array([.002, .002]), .1)
        self.assertAlmostEqual(same[0], same[1])

    def test_refinement_approaches_independent_exact_solution(self):
        cfg = passive.config(ROOT)
        geometry, membrane = cfg['geometries'][0], cfg['passive_membrane']
        cap, leak, axial = passive.coefficients(geometry, membrane, True)
        initial = np.array(cfg['paired_initial_mv'])
        pulse = passive.current_for_case('parent_only', cfg['paired_pulse_na'])
        exact = passive.exact_piecewise_endpoint(initial, cap, leak, axial,
            membrane['e_pas_mv'], pulse, cfg['pulse_window_ms'], cfg['duration_ms'])
        errors = []
        for dt in cfg['convergence_dt_ms']:
            prediction = passive.simulate_discrete(initial, cap, leak, axial,
                membrane['e_pas_mv'], pulse, cfg['pulse_window_ms'], cfg['duration_ms'], dt)
            errors.append(float(np.max(np.abs(prediction[-1]-exact))))
        self.assertGreater(errors[0], errors[1])
        self.assertGreater(errors[1], errors[2])
        self.assertLess(errors[2], cfg['gates']['max_convergence_fine_endpoint_error_mv'])

    def test_v2_new_single_cases_have_prespecified_convergence(self):
        cfg = passive.config(ROOT, 'iv_ad1_passive_axial_preregistration_v2.json')
        start, end = cfg['pulse_window_ms']
        for case in cfg['single_cases']:
            with self.subTest(case=case['id']):
                cap, leak = passive.membrane_coefficients(
                    case['length_um'], case['diam_um'], case['cm_uf_cm2'], case['g_pas_s_cm2'])
                value = case['initial_mv']
                if leak:
                    equilibrium = case['e_pas_mv']
                    value = equilibrium + (value-equilibrium)*math.exp(-start*leak/cap)
                    active = equilibrium + case['pulse_na']/leak
                    value = active + (value-active)*math.exp(-(end-start)*leak/cap)
                    value = equilibrium + (value-equilibrium)*math.exp(-(cfg['duration_ms']-end)*leak/cap)
                else:
                    value += case['pulse_na']*(end-start)/cap
                errors = []
                for dt in cfg['convergence_dt_ms']:
                    predicted = passive.simulate_discrete(
                        np.array([case['initial_mv']]*2), np.array([cap]*2),
                        np.array([leak]*2), 0., case['e_pas_mv'],
                        np.array([case['pulse_na'], 0.]), (start, end),
                        cfg['duration_ms'], dt)[-1, 0]
                    errors.append(abs(predicted-value))
                self.assertLessEqual(errors[-1], cfg['gates']['max_single_fine_endpoint_error_mv'])
                if leak:
                    self.assertGreater(errors[0], errors[1])
                    self.assertGreater(errors[1], errors[2])
                else:
                    self.assertLess(max(errors), 1e-9)

    def test_v3_equal_density_symmetric_control_on_new_geometries(self):
        cfg = passive.config(ROOT, 'iv_ad1_passive_axial_preregistration_v3.json')
        for geometry in cfg['geometries']:
            with self.subTest(geometry=geometry['id']):
                cap, leak, axial = passive.coefficients(geometry, cfg['passive_membrane'], True)
                pulse = np.array([cfg['paired_pulse_na'],
                                  cfg['paired_pulse_na'] * cap[1] / cap[0]])
                trajectory = passive.simulate_discrete(
                    np.array([-70., -70.]), cap, leak, axial,
                    cfg['passive_membrane']['e_pas_mv'], pulse,
                    cfg['pulse_window_ms'], cfg['duration_ms'], cfg['dt_ms'])
                self.assertLess(np.max(np.abs(trajectory[:, 0]-trajectory[:, 1])),
                                cfg['gates']['max_symmetric_voltage_difference_mv'])


if __name__ == '__main__':
    unittest.main()
