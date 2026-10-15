"""Regression checks for causal views, native interface and resident membrane math."""
import importlib.util
import json
from pathlib import Path
import unittest

import numpy as np

from src.giada_teacher import cc_n1_contract_memory as n1
from src.giada_teacher import ionic_block_teacher_forced as ionic
from src.giada_teacher import task30_autonomous_voltage as t30

ROOT = Path(__file__).resolve().parents[1]


class MemoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spec, cls.cfg, cls.calcium, _, cls.base = n1.contract(ROOT)

    def test_physical_initial_states_and_full_state_reproducibility(self):
        rows, state = n1.memory_design(self.spec)
        self.assertEqual(len(rows), 288)
        self.assertTrue(np.all((state >= 0) & (state <= 1)))
        # Same future inputs but different preparation; memory must not be initialized to equilibrium each step.
        chosen = [j for j, r in enumerate(rows) if r['tail_ms'] == 0 and r['panel'] == 'canonical'
                  and r['preparation_ms'] == 200 and r['future_current_ma_cm2'] == 0
                  and r['preparation_voltage_mv'] in (-85, -25)]
        self.assertEqual(len(chosen), 2)
        self.assertGreater(np.max(np.abs(state[chosen[0]]-state[chosen[1]])), .01)
        short = {**self.spec, 'memory': {**self.spec['memory'], 'rollout_ms': 1.}}
        subset = [rows[i] for i in chosen]
        a = n1.memory_rollout(subset, state[chosen], short, self.cfg, self.calcium, self.base)
        b = n1.memory_rollout(subset, state[chosen], short, self.cfg, self.calcium, self.base)
        np.testing.assert_array_equal(a['voltage'][0], [-65., -65.])
        np.testing.assert_array_equal(a['calcium'][0], [.0001, .0001])
        np.testing.assert_array_equal(a['voltage'], b['voltage'])
        self.assertGreater(np.max(np.abs(a['voltage'][:, 0]-a['voltage'][:, 1])), .001)

    def test_zero_duration_identity_and_long_tail_decay(self):
        m = self.spec['memory']
        expected = t30.initial_states(np.array([-65.]), np.array([.0001]))[0]
        np.testing.assert_allclose(n1.conditioned_state(-25., 0., 0., m), expected, atol=1e-15)
        a = n1.conditioned_state(-25., 200., 0., m)
        b = n1.conditioned_state(-25., 200., 200., m)
        self.assertLess(np.max(np.abs(b-expected)), np.max(np.abs(a-expected)))

    def test_collision_groups_keep_forcing_and_parameters_fixed(self):
        rows, state = n1.memory_design(self.spec)
        result = n1.memory_rollout(rows, state, self.spec, self.cfg, self.calcium, self.base)
        groups = n1.collision_report(rows, result, self.spec)
        self.assertEqual(len(groups), 96)
        for group in groups:
            left, right = group['voltage_extreme_histories']
            self.assertEqual(left['panel'], right['panel'])
            self.assertEqual(left['future_current_ma_cm2'], right['future_current_ma_cm2'])
            self.assertTrue(all(w <= group['tail_ms'] for w in group['matching_recent_windows_ms']))
            self.assertEqual(group['voltage_minimax_lower_bound_mv'], group['voltage_endpoint_separation_mv']/2)

    def test_notebook_code_syntax_and_no_uncontrolled_report_dump(self):
        nb = json.loads((ROOT/'notebooks/cc_n1_contract_memory.ipynb').read_text(encoding='utf-8'))
        for i, cell in enumerate(nb['cells']):
            if cell['cell_type'] == 'code': compile(''.join(cell['source']), f'cell-{i}', 'exec')
        self.assertEqual(nb['metadata']['giada_contract'], self.spec['schema_version'])

    @unittest.skipUnless(importlib.util.find_spec('torch'), 'torch unavailable in lightweight local runtime')
    def test_resident_step_matches_numpy_with_dynamic_calcium_and_synapses(self):
        import torch
        rng = np.random.default_rng(90600)
        v = np.array([-80., -65., -45., -10.])
        ca = np.array([.0001, .0002, .0004, .0008])
        gates = t30.initial_states(v, ca)
        mult = np.stack(list(ionic.panel_multipliers().values())[:4])
        inj = np.array([0., .003, .006, .012])
        syn = rng.uniform(0, .0001, (4, 4))
        arrays = (v, ca, gates, mult, inj, syn)
        ref = n1.numpy_step(*arrays, 100*np.pi, self.cfg, self.calcium, self.base)
        tensors = [torch.tensor(a[None], dtype=torch.float64) for a in arrays]
        out = n1.torch_step(torch, *tensors, 100*np.pi, self.cfg, self.calcium, self.base)
        for a, b in zip(out, ref): np.testing.assert_allclose(a.numpy()[0], b, rtol=1e-10, atol=1e-10)

    @unittest.skipUnless(importlib.util.find_spec('torch'), 'torch unavailable in lightweight local runtime')
    def test_fullgraph_capture_uses_python_metadata_indices(self):
        import torch
        v = np.array([-65., -45.])
        ca = np.array([.0001, .0002])
        arrays = (v, ca, t30.initial_states(v, ca), np.ones((2, 11)),
                  np.array([.003, .006]), np.full((2, 4), .00005))
        tensors = [torch.tensor(a[None], dtype=torch.float32) for a in arrays]
        cfg, calcium, base, area = dict(self.cfg), dict(self.calcium), dict(self.base), float(100*np.pi)
        def fn(*x):
            return n1.torch_step(torch, *x, area, cfg, calcium, base)
        expected = fn(*tensors)
        # FX capture catches symbolic NumPy indices without requiring a local CUDA compiler.
        captured = torch.compile(fn, backend='eager', fullgraph=True)(*tensors)
        for actual, target in zip(captured, expected):
            torch.testing.assert_close(actual, target)


if __name__ == '__main__':
    unittest.main()
