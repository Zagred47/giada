import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch
from src.giada_teacher import task38_joint_transition as t

ROOT = Path(__file__).resolve().parents[1]


class Task38Tests(unittest.TestCase):
    def test_optimizer_and_phase(self):
        self.assertTrue(t.preflight(torch, 'cpu')['valid'])

    def test_readout_matches_numpy_and_blocks(self):
        rng = np.random.default_rng(2)
        v = rng.uniform(-90, 30, (3, 11)).astype('float32')
        x = rng.uniform(0, 1, (3, 11, 18)).astype('float32')
        m = rng.uniform(0, 4, (11, 11)).astype('float32'); m[:, 3] = 0
        e = np.broadcast_to(t.p.ionic.REVERSALS, (11, 11)).astype('float32')
        got = t.readout(torch, *[torch.tensor(a) for a in (v, x, m, e)]).numpy()
        _, ref = t.p.readout(v, x, m, e)
        np.testing.assert_allclose(got, ref, atol=1e-8)
        self.assertTrue(np.all(got[..., 3] == 0))

    def test_chunked_recurrence(self):
        spec = json.loads((ROOT/t.CONFIG).read_text())
        models = [t.model_class(torch)(w, spec['seeds']) for w in spec['architectures']]
        with tempfile.TemporaryDirectory(dir=ROOT) as folder:
            t.recurrence_preflight(Path(folder), spec, models, torch, 'cpu')

    def test_pool_previous_state(self):
        root = ROOT/'experiments/results/task37_playground_v2'
        pool, families = t.pool(root, 'fit', ['steps', 'slow_recovery'], 17, 3)
        self.assertEqual(pool.shape, (34, 71))
        self.assertEqual(np.bincount(families).tolist(), [17, 17])
        self.assertTrue(np.isfinite(pool).all())
        self.assertTrue(np.all((pool[:, 2:38] >= 0) & (pool[:, 2:38] <= 1)))


if __name__ == '__main__': unittest.main()
