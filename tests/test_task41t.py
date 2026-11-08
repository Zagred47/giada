import json,os,sys,unittest
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
import prepare_task41t as t

class Task41t(unittest.TestCase):
    def test_rectangular_solve(self):
        w=torch.tensor([[1.,0.],[0.,2.],[0.,0.]],dtype=torch.float64)
        b=np.array([[1.,2.,3.],[.2,-.3,.1]]);y=np.array([.1,.3])
        actual=t.projected_step(w,b,y).numpy();c=b@w.numpy()
        expected=w.numpy()@np.linalg.solve(np.eye(2)+c.T@c/2,c.T@y/2)
        np.testing.assert_allclose(actual,expected,atol=1e-13)
        np.testing.assert_equal(t.projected_step(torch.zeros((3,0),dtype=torch.float64),b,y).numpy(),np.zeros(3))

    def test_history_pair_same_future_and_restoration(self):
        original=t.r.n.stress_data;spec=t.r.n.contract(ROOT)[0];spec=dict(spec,stress_seeds=[9417101])
        with t.memory_histories(True):
            values=t.r.n.stress_data(spec)
            for i in range(0,len(values),2):
                np.testing.assert_equal(values[i]['v'],values[i+1]['v'])
                np.testing.assert_equal(values[i]['ca'],values[i+1]['ca'])
                self.assertGreater(float(abs(values[i]['initial']-values[i+1]['initial']).max()),.01)
        self.assertIs(t.r.n.stress_data,original)

    def test_protocol(self):
        a=json.loads((ROOT/'experiments/task41t_preregistration_v1.json').read_text());b=json.loads((ROOT/'experiments/task41s_preregistration_v1.json').read_text())
        self.assertEqual(a['quality'],b['quality']);self.assertEqual(len(a['arms']),13)
        self.assertIn('recomposed',a['arms'])
        self.assertFalse(set(a['fit_path_seeds']+a['guard_path_seeds']+a['development_seeds'])&set(b['fit_path_seeds']+b['development_seeds']))

    def test_preflight_and_frozen_weights(self):
        folder=ROOT/'experiments/results/task41t_protocol_v1'
        audit=json.loads((folder/'numerical_preflight.json').read_text());self.assertTrue(audit['valid'])
        self.assertEqual(len(audit['full_recurrence_directional_checks']),8)
        self.assertTrue(all(x['valid'] for x in audit['full_recurrence_directional_checks']))
        if not (folder/'fit_manifest.json').exists():self.skipTest('Fits running')
        manifest=json.loads((folder/'fit_manifest.json').read_text());os.environ['GIADA_TASK41N_FROZEN']=str(ROOT/'artifacts/task41n_frozen.zip.bin')
        base=t.r.n.load_models(torch,ROOT,'cpu')[0]['dual']
        with np.load(folder/'heads_incumbent.npz') as a:incumbent=dict(a)
        for name,entry in manifest['models'].items():
            with np.load(ROOT/entry['path']) as a:weights=dict(a)
            net=t.r.materialize(torch,base,weights)
            self.assertEqual(t.r.n.weights_hash(net),entry['weights_sha256'])
            self.assertEqual(sum(p.numel() for p in net.parameters()),sum(p.numel() for p in base.parameters()))
            self.assertTrue(all(torch.isfinite(p).all() for p in net.parameters()))
            gates=(12,13) if name.startswith('k_') else (8,9) if name.startswith('nap_') else ()
            allowed={f'h{kind}_{gate}' for gate in gates for kind in (0,1)}
            for key in weights:
                if key not in allowed:np.testing.assert_equal(weights[key],incumbent[key])

    def test_notebook(self):
        import task41t_kaggle as k
        doc=k.notebook('a'*40,'b'*64);code=''.join(doc['cells'][1]['source']);compile(code,'41t','exec')
        self.assertIn('scripts/run_task41t.py',code);self.assertIn('task41t_final',code)
        self.assertEqual(doc['metadata']['kernelspec']['name'],'python3')

if __name__=='__main__':unittest.main()
