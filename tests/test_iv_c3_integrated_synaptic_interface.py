import unittest
from pathlib import Path

from src.giada_teacher.iv_c3_integrated_synaptic_interface import _passes, load_contract


ROOT = Path(__file__).resolve().parents[1]


class TestIVC3Contract(unittest.TestCase):
    def test_parent_hashes_and_factorial_shape(self):
        cfg = load_contract(ROOT)
        self.assertEqual(cfg['roadmap_id'], 'IV-C3')
        self.assertEqual(len(cfg['synapses']), 4)
        self.assertEqual(len(cfg['confirmation_schedules_ms']), 3)
        self.assertEqual(len(cfg['confirmation_seeds']), 4)
        self.assertEqual(len(cfg['holding_voltages_mv']), 3)
        self.assertFalse(set(sum(cfg['calibration_schedule_ms'].values(), [])) &
                         set(sum((sum(s.values(), []) for s in cfg['confirmation_schedules_ms'].values()), [])))

    def test_every_gate_is_required(self):
        gates = load_contract(ROOT)['gates']
        row = {key: 0 for key in (
            'max_plastic_state_error', 'max_rng_sequence_difference',
            'release_mismatch_count', 'max_receptor_state_error',
            'max_conductance_error_us', 'max_current_error_na',
            'max_integrated_charge_error_na_ms',
            'max_native_restart_state_error', 'max_native_restart_current_error_na',
            'max_native_restart_rng_difference', 'max_native_restart_clock_error_ms',
            'max_holding_error_mv')}
        self.assertTrue(_passes(row, gates))
        for key, limit in gates.items():
            candidate = dict(row)
            candidate[key if key != 'max_release_mismatch_count' else 'release_mismatch_count'] = limit + 1
            self.assertFalse(_passes(candidate, gates), key)


if __name__ == '__main__':
    unittest.main()
