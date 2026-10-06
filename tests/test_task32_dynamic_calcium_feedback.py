"""Local preflight for the prospective Task32 implementation (no NEURON)."""

import json
from pathlib import Path
import unittest

import numpy as np

from src.giada_teacher import task32_dynamic_calcium_feedback as task
from src.giada_teacher.iv_b_calcium_prerequisite import calcium_exact


ROOT = Path(__file__).resolve().parents[1]
CFG = json.loads((ROOT / 'experiments/task32_dynamic_calcium_feedback_v2.json').read_text())
BASE = task.t30.config(ROOT, Path(CFG['base_config']).name)


class Task32Preflight(unittest.TestCase):
    def test_disjoint_episode_roles(self):
        design = task.design(CFG, BASE)
        calibration = [r for r in design['rows'] if r['protocol'] in CFG['calibration_protocols']]
        confirmation = [r for r in design['rows'] if r['protocol'] in CFG['confirmation_protocols']]
        self.assertEqual((len(calibration), len(confirmation)), (8, 8))
        self.assertFalse(set(CFG['calibration_protocols']) & set(CFG['confirmation_protocols']))
        self.assertTrue(np.any(design['injection_ma_cm2'][:80] != 0))

    def test_calcium_array_matches_validated_scalar(self):
        cai = np.array([1e-4, 5e-4])
        current = np.array([0., -.005])
        got = task.calcium_step(cai, current, CFG)
        expected = [calcium_exact(a, b, CFG['dt_ms'], CFG['gamma'],
                                  CFG['depth_um'], CFG['decay_ms'])
                    for a, b in zip(cai, current)]
        np.testing.assert_allclose(got, expected, atol=1e-15)

    def test_closed_loop_and_frozen_calcium_differ(self):
        design = task.design(CFG, BASE)
        live = task.rollout(design, CFG, BASE, 'old_state_ica')
        frozen = task.rollout(design, CFG, BASE, 'old_state_ica', frozen_cai=True)
        self.assertEqual(live['voltage'].shape, (1, 801, 16))
        self.assertTrue(np.isfinite(live['gates']).all())
        self.assertGreater(float(np.max(abs(live['calcium']-frozen['calcium']))), 1e-5)
        self.assertGreater(float(np.max(abs(live['voltage']-frozen['voltage']))), 1e-3)


if __name__ == '__main__':
    unittest.main()
