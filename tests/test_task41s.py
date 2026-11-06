import json,os,sys,unittest
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import prepare_task41s as s
from src.giada_teacher import task41s_direct_timing as run

class Task41s(unittest.TestCase):
    def test_protocol_unchanged_quality_and_selection(self):
        plan=json.loads((ROOT/'experiments/task41s_preregistration_v1.json').read_text())
        previous=json.loads((ROOT/'experiments/task41r_preregistration_v1.json').read_text())
        self.assertEqual(plan['quality'],previous['quality'])
        self.assertEqual(len(plan['arms']),10)
        self.assertFalse(set(plan['fit_path_seeds']+plan['development_seeds'])&set(previous['fit_path_seeds']+previous['development_seeds']))
        self.assertNotIn('joint_control',plan['selectable'])
        self.assertNotIn('probe_activation',plan['selectable'])
        self.assertIn('incumbent',plan['selectable'])

    def test_step_only_requested_seed_and_heads(self):
        weights={f'h0_{g}':np.zeros((3,2),dtype='float32') for g in (8,9)}
        actual=s.step_weights(weights,weights,[(0,8)],1,torch.tensor([1.,2.]),.5)
        np.testing.assert_equal(actual['h0_8'],[[0,0],[.5,1],[0,0]])
        np.testing.assert_equal(actual['h0_9'],weights['h0_9'])
        np.testing.assert_equal(weights['h0_8'],np.zeros((3,2)))

    def test_manifest_and_gate_probes(self):
        path=ROOT/'experiments/results/task41s_protocol_v1/fit_manifest.json'
        if not path.exists():self.skipTest('Preparation running')
        manifest=json.loads(path.read_text());self.assertTrue(manifest['valid'])
        os.environ['GIADA_TASK41N_FROZEN']=str(ROOT/'artifacts/task41n_frozen.zip.bin')
        base=run.n.load_models(torch,ROOT,'cpu')[0]['dual']
        weights={}
        for name,entry in manifest['models'].items():
            with np.load(ROOT/entry['path']) as a:weights[name]=dict(a)
            net=run.materialize(torch,base,weights[name])
            self.assertEqual(run.n.weights_hash(net),entry['weights_sha256'])
            self.assertEqual(sum(p.numel() for p in net.parameters()),sum(p.numel() for p in base.parameters()))
        for label,gate in [('activation',12),('inactivation',13)]:
            for key in weights['incumbent']:
                expected=weights['joint_control'][key] if key in (f'h0_{gate}',f'h1_{gate}') else weights['incumbent'][key]
                np.testing.assert_equal(weights['probe_'+label][key],expected)

    def test_notebook(self):
        import task41s_kaggle as k
        doc=k.notebook('a'*40,'b'*64);code=''.join(doc['cells'][1]['source'])
        compile(code,'41s','exec')
        self.assertEqual(doc['metadata']['kernelspec']['name'],'python3')
        self.assertIn('scripts/run_task41s.py',code)
        self.assertIn('task41s_final',code)
        self.assertNotIn('scripts/run_task41r.py',code)

    def test_tail_probe_on_real_archived_arrays(self):
        import tempfile,zipfile
        source=ROOT/'artifacts/task41r_kaggle_external/task41r_final.zip'
        if not source.exists():self.skipTest('Archived input absent')
        spec=run.n.contract(ROOT)[0]
        with tempfile.TemporaryDirectory(dir=ROOT/'artifacts') as tmp:
            folder=Path(tmp)
            with zipfile.ZipFile(source) as z:
                for name in ('fresh_stress_inputs.npz','stress_dual.npz'):
                    (folder/name).write_bytes(z.read('opened41n_timing_frozen_k/'+name))
            run.tail_probe(folder,spec)
            value=json.loads((folder/'tail_sensitivity_probe.json').read_text())
            self.assertFalse(value['evaluator_changed'])
            self.assertTrue(value['rows'])
            self.assertTrue(all(np.isfinite(row['slope']) for row in value['rows']))

if __name__=='__main__':unittest.main()
