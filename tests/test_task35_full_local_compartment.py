"""Fast contract and mixed-panel regression tests for Task35."""

import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from src.giada_teacher import ionic_block_teacher_forced as ionic
from src.giada_teacher import task35_full_local_compartment as task35
from src.giada_teacher import task33_observable_synaptic_feedback as task33


ROOT = Path(__file__).resolve().parents[1]


class Task35ContractTests(unittest.TestCase):
    def test_parent_and_factorial_contract(self):
        spec, cfg, t32cfg, c3 = task35.contract(ROOT)
        rows = task35.cases(spec)
        self.assertEqual(len(rows), 32)
        self.assertEqual(set(r['panel'] for r in rows), set(spec['panels']))
        self.assertEqual(len({(r['panel'], r['seed'], r['initial_voltage_mv'], r['protocol']) for r in rows}), 32)
        self.assertEqual(cfg['synaptic_state_phase'], 'old')
        self.assertEqual(len(c3['synapses']), 4)
        self.assertEqual(len(t32cfg['model_seeds']), 3)

    def test_all_panels_distinct_from_canonical(self):
        panels = ionic.panel_multipliers()
        self.assertTrue(np.array_equal(panels['canonical'], np.ones(11)))
        for name in ('calcium_x4', 'sodium_x4', 'potassium_x4'):
            self.assertFalse(np.array_equal(panels[name], panels['canonical']))

    def test_coupled_solver_uses_each_case_own_conductance_panel(self):
        rows = []
        for panel in ('canonical', 'calcium_x4'):
            rows.append({'voltage': np.array([-70., -70.]),
                         'injection': np.zeros(1), 'area_um2': 314.159,
                         'multipliers': ionic.panel_multipliers()[panel]})
        shadows = [{'base_g_us': np.zeros((2, 4))} for _ in rows]
        _, cfg, t32cfg, _ = task35.contract(ROOT)
        with patch.object(task33.ionic, 'exact_step', side_effect=lambda v, c, s, dt: s.copy()), \
             patch.object(task33.t32, '_ca_current', side_effect=lambda v, s, m: np.zeros_like(v)), \
             patch.object(task33.t32, 'calcium_step', side_effect=lambda c, i, config: c), \
             patch.object(task33.t32, '_sk_update'), \
             patch.object(task33, 'voltage_step_with_synapses',
                          side_effect=lambda v, s, m, i, g, a, b, dt: v + m[..., 0]):
            result = task33.coupled_rollout(rows, shadows, cfg, t32cfg, {})
        np.testing.assert_array_equal(result['voltage'][0, 1], [-69., -66.])


if __name__ == '__main__':
    unittest.main()
