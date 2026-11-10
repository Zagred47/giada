import json,os,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from src.giada_teacher import task45_controlled_coverage as a
ROOT=Path(__file__).resolve().parents[1]

class Coverage(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        cls.torch=torch;torch.set_num_threads(1)
        os.environ['GIADA_TASK41N_FROZEN']=str(ROOT/'artifacts/task41n_frozen.zip.bin')
        cls.p,cls.cfg,cls.ca,cls.base=a.spec(ROOT)
        cls.net=a.n.load_models(torch,ROOT,'cpu')[0]['dual']
        with np.load(ROOT/cls.p['freeze']['heads_path']) as z:cls.net=a.x.repair.materialize(torch,cls.net,dict(z))
    def test_contract_design_and_geometry(self):
        self.assertEqual(self.p['quality'],a.n.contract(ROOT)[0]['quality'])
        self.assertFalse(self.p['conditional_refit']['authorized']);self.assertEqual(self.p['training_updates'],0)
        for geo in self.p['geometries']:
            rows,inj,mult=a.design(geo,self.p);count=len(geo['parents'])
            self.assertEqual(inj.shape,(10000,9,count));self.assertEqual(mult.shape,(9,count,11))
            self.assertTrue((inj<0).any());self.assertTrue((inj>0).any())
            self.assertFalse(inj[6500:].any())
            self.assertEqual(len(rows),9)
            area,lap=a.x.matrix(geo,1.)
            self.assertTrue((area>0).all());np.testing.assert_allclose(lap,lap.T,atol=0)
            np.testing.assert_allclose(lap.sum(0),0,atol=1e-15)
            self.assertTrue(all(p<i for i,p in enumerate(geo['parents']) if i))
    def test_context_and_frozen_endpoint(self):
        before=a.y.spec,a.y.ARMS,a.y.previous.design
        with self.torch.inference_mode(),a.configured():
            self.assertEqual(a.y.ARMS,('none','all'))
            for geo in self.p['geometries']:
                result=a.y.endpoint_preflight(self.torch,ROOT,self.net,geo,self.p,self.cfg,self.ca,self.base,'cpu')
                self.assertTrue(all(r['passed'] for r in result));self.assertEqual({r['batch_copies'] for r in result},{1,2})
        self.assertEqual(before,(a.y.spec,a.y.ARMS,a.y.previous.design))
        self.assertEqual(a.n.weights_hash(self.net),self.p['freeze']['logical_weights_sha256'])
    def test_crop_and_refit_decision(self):
        data=dict(voltage=np.zeros((11,3,1,9,4)),current=np.zeros((10,3,1,9,4,11)))
        cropped=a.crop(data,5);self.assertEqual(cropped['voltage'].shape[0],6);self.assertEqual(cropped['current'].shape[0],5)
        good=dict(arm='all',judgment_eligible=True,finite=True,current_passed=True,compartment_passed=True)
        control=dict(good,arm='none')
        self.assertTrue(a.decision([control,good],[],True)['frozen_coverage_passed'])
        bad=dict(good,current_passed=False)
        self.assertTrue(a.decision([control,good],[bad],True)['refit_recommended'])
        blocked=dict(bad,judgment_eligible=False)
        self.assertFalse(a.decision([control,blocked],[],True)['refit_recommended'])
        self.assertFalse(a.decision([control,bad],[],False)['refit_recommended'])
        invalid_regime=dict(control,compartment_passed=False)
        self.assertFalse(a.decision([invalid_regime,bad],[],True)['refit_recommended'])
    def test_short_full_evaluation_with_prefixes(self):
        torch=self.torch;p=dict(self.p,duration_ms=12.,prefix_horizons_ms=[4.,8.,12.],geometries=[self.p['geometries'][0]])
        with tempfile.TemporaryDirectory(dir=ROOT/'artifacts') as tmp,torch.inference_mode():
            out=Path(tmp)/'result';out.mkdir();(out/'native_floor.json').write_text(json.dumps(dict(valid=True)))
            with a.configured():repeated,_=a.y.rollout(torch,None,p['geometries'][0],p,self.cfg,self.ca,self.base,'cpu',a.ARMS)
            np.savez_compressed(out/'native_chain4.npz',**{k:repeated[k][:,0,0] for k in ('voltage','calcium','gates')})
            with patch.object(a,'spec',return_value=(p,self.cfg,self.ca,self.base)),patch.object(a.x.repair.p.r,'verify_source'),patch.object(a.n,'load_models',return_value=({'dual':self.net},None)),patch.dict(os.environ):
                report=a.evaluate(torch,ROOT,Path(tmp)/'unused-fixture',out,'cpu')
            self.assertTrue(report['complete'] and report['candidate_weights_unchanged'])
            self.assertEqual(report['arms']['all']['autonomous']['total'],27)
            self.assertFalse(report['coverage_decision']['refit_performed'])
            prefixes=json.loads((out/'prefix_results.json').read_text());self.assertEqual({r['horizon_ms'] for r in prefixes},{4.,8.})
            self.assertEqual(len(prefixes),216)
            for name in ('coverage_probes.json','paired_contrasts.json','coupling_contrasts.json'):self.assertTrue((out/name).is_file())
    def test_notebook(self):
        sys.path.insert(0,str(ROOT/'scripts'))
        import task45_kaggle as k
        doc=k.notebook('a'*40,'b'*64)
        for cell in doc['cells']:
            if cell['cell_type']=='code':compile(''.join(cell['source']),'45-notebook','exec')
        text=json.dumps(doc);self.assertIn('run_task45.py',text);self.assertNotIn('run_task44.py',text)
        self.assertIn('neuron==8.2.7',text)
        self.assertNotIn('optimizer', (ROOT/'scripts/run_task45.py').read_text())
    def test_supervisor_only_retrieves_existing_finished_run(self):
        import types
        sys.path.insert(0,str(ROOT/'scripts'))
        import task45_kaggle as k
        api=types.SimpleNamespace(authenticate=lambda:None,kernels_status=lambda ref:types.SimpleNamespace(status='complete',failure_message=None))
        with tempfile.TemporaryDirectory(dir=ROOT/'artifacts') as tmp:
            folder=Path(tmp);(folder/'kaggle_launch.json').write_text(json.dumps(dict(ref='/code/unit-test/existing',version=1)))
            with patch.object(k.launcher,'FOLDER',folder),patch.object(k.launcher,'main') as retrieve,patch.object(sys,'argv',['unit-test','supervise']),patch.object(k.time,'sleep') as sleep:
                k.supervise(api)
                retrieve.assert_called_once_with();sleep.assert_not_called()
                self.assertEqual(sys.argv[-1],'retrieve')
            self.assertEqual(json.loads((folder/'status.json').read_text())['status'],'complete')
if __name__=='__main__':unittest.main()
