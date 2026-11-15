import copy,json,os,sys,tempfile,unittest
from pathlib import Path
import numpy as np
from src.giada_teacher import task48_state_minimality as q
ROOT=Path(__file__).resolve().parents[1]

class StateDiagnostics(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        cls.torch=torch;torch.set_num_threads(1);cls.p=q.plan(ROOT)
    def test_exact_collisions_and_valid_occupancy(self):
        states,pairs=q.collision_bank(self.p)
        self.assertEqual(states.shape,(42,18));self.assertTrue(((states>=0)&(states<=1)).all())
        for row in pairs:
            np.testing.assert_allclose(q.openings(states[row['left']]),q.openings(states[row['right']]),atol=1e-14,rtol=0)
            self.assertFalse(np.array_equal(states[row['left']],states[row['right']]))
    def test_canonical_parity_and_convexity(self):
        torch=self.torch;self.assertTrue(q.parity(torch,'cpu')['valid'])
        g=torch.tensor(np.tile(q.steady(),(3,2,1)),dtype=torch.float64);v=torch.full((3,2),-30.,dtype=torch.float64);ca=torch.full_like(v,5e-5)
        after=q.canonical_step(torch,g,v,ca);self.assertTrue(bool(((after>=0)&(after<=1)).all()))
        states,pairs=q.collision_bank(self.p);g=torch.tensor(states,dtype=torch.float64);v=torch.full((len(g),),-20.,dtype=torch.float64);ca=torch.full_like(v,5e-5)
        after=q.canonical_step(torch,g,v,ca).numpy();self.assertGreater(max(abs(q.openings(after[r['left']])-q.openings(after[r['right']])).max() for r in pairs),1e-5)
    def test_histories_and_no_false_minimality(self):
        v,ca,meta=q.history_inputs(self.p);self.assertEqual(v.shape,(4000,72));np.testing.assert_equal(v[-1],-65.);np.testing.assert_equal(ca[-1],5e-5)
        self.assertEqual(sum(map(len,q.family_indices(self.p).values())),18)
        pairs=q.neighbour_pairs(np.stack([q.steady(),q.steady()+.001,q.steady()+.002]),self.p['families'])
        self.assertEqual({r['view'] for r in pairs},{'full_gates','channel_openings','family_currents'})
    def test_notebook_routes_only48(self):
        sys.path.insert(0,str(ROOT/'scripts'));import task48_kaggle as k
        doc=k.notebook('a'*40)
        for cell in doc['cells']:
            if cell['cell_type']=='code':compile(''.join(cell['source']),'48-notebook','exec')
        text=json.dumps(doc);self.assertIn('run_task48.py',text);self.assertNotIn('run_task46.py',text);self.assertNotIn('run_task41n_recovery.py',text)
        self.assertIn('checkpoint45c.zip.bin',text);self.assertIn('evaluation45d.zip.bin',text);self.assertIn('task48_final',text)
        self.assertNotIn('nrnivmodl',text);self.assertNotIn('native_python',text)
    def test_complete_small_real_frozen_P(self):
        p=copy.deepcopy(self.p);p.update(history_ms=.4,history_count=4,horizons_ms=[.1,.2],future_protocols=['rest'],sensitivity_anchor_count=1)
        source=ROOT/'artifacts/task41n_primary_external/recovery_final.zip';c=ROOT/'artifacts/task45c_cost_input/checkpoint45c.zip.bin';d=ROOT/'artifacts/task45d_evaluation_input/evaluation45d.zip.bin'
        self.assertTrue(all(path.exists() for path in (source,c,d)))
        with tempfile.TemporaryDirectory(dir=ROOT/'artifacts') as tmp,self.torch.inference_mode():
            out=Path(tmp)/'out';out.mkdir();r=q.run(self.torch,ROOT,source,c,d,out,'cpu',p)
            self.assertTrue(r['valid'] and r['complete']);self.assertEqual(r['controls']['weights_hash_before'],p['weights_sha256'])
            self.assertFalse(r['speedup_claimed']);self.assertFalse(r['compressed_model_qualified'])
            rows=json.loads((out/'diagnostic_rows.json').read_text());self.assertIn('stacked_horizon_observability',{r['kind'] for r in rows})
            self.assertEqual(len(list(out.glob('panel_*.npz'))),3)
if __name__=='__main__':unittest.main()
