import os,json,sys,unittest
from pathlib import Path
import numpy as np
from src.giada_teacher import task44_composition_localization as y
ROOT=Path(__file__).resolve().parents[1]

class Localization(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        cls.torch=torch;torch.set_num_threads(1)
        os.environ['GIADA_TASK41N_FROZEN']=str(ROOT/'artifacts/task41n_frozen.zip.bin')
        cls.p,cls.cfg,cls.ca,cls.base=y.spec(ROOT)
        cls.net=y.n.load_models(torch,ROOT,'cpu')[0]['dual']
        with np.load(ROOT/cls.p['freeze']['heads_path']) as z:cls.net=y.x.repair.materialize(torch,cls.net,dict(z))
    def test_masks_and_contract(self):
        self.assertEqual(self.p['quality'],y.n.contract(ROOT)[0]['quality'])
        self.assertEqual(y.n.weights_hash(self.net),self.p['freeze']['logical_weights_sha256'])
        m=y.channel_masks(4)
        self.assertFalse(m[0].any());self.assertTrue(m[-1].all())
        self.assertTrue(m[1,0].all());self.assertFalse(m[1,1:].any())
        self.assertTrue(m[2,-1].all());self.assertFalse(m[2,:-1].any())
        np.testing.assert_array_equal(m[3:7].sum(0),1)
    def test_legacy_endpoint_bitwise_parity(self):
        with self.torch.inference_mode():
            result=y.endpoint_preflight(self.torch,ROOT,self.net,y.x.geometry(4),self.p,self.cfg,self.ca,self.base,'cpu')
            self.assertTrue(all(r['passed'] for r in result))
    def test_evaluate_short_integration(self):
        import tempfile
        from unittest.mock import patch
        torch=self.torch;p=dict(self.p,duration_ms=.3,geometries=[y.x.geometry(4)])
        with tempfile.TemporaryDirectory(dir=ROOT/'artifacts') as tmp,torch.inference_mode():
            out=Path(tmp)/'result';out.mkdir()
            (out/'native_floor.json').write_text(json.dumps(dict(valid=True)))
            repeated,_=y.rollout(torch,None,p['geometries'][0],p,self.cfg,self.ca,self.base,'cpu',y.ARMS)
            np.savez_compressed(out/'native_chain4.npz',**{k:repeated[k][:,0,0] for k in ('voltage','calcium','gates')})
            with patch.object(y,'spec',return_value=(p,self.cfg,self.ca,self.base)),patch.object(y.x.repair.p.r,'verify_source'),patch.object(y.n,'load_models',return_value=({'dual':self.net},None)),patch.dict(os.environ):
                report=y.evaluate(torch,ROOT,Path(tmp)/'unused-test-fixture',out,'cpu')
            self.assertTrue(report['complete'] and report['candidate_weights_unchanged'] and report['all_parameters_finite'])
            self.assertEqual(report['arms']['all']['autonomous']['total'],18)
            for filename in ('results.json','paired_contrasts.json','coupling_contrasts.json','family_interactions.json','final_report.json'):
                self.assertTrue((out/filename).is_file())
    def test_spatial_state_isolation_and_rollout(self):
        torch=self.torch;p=dict(self.p,duration_ms=12.);geo=y.x.geometry(4)
        with torch.inference_mode():
            repeated,_=y.rollout(torch,None,geo,p,self.cfg,self.ca,self.base,'cpu',y.ARMS)
            ref={k:v[:,:,:1] for k,v in repeated.items()}
            auto,_=y.rollout(torch,self.net,geo,p,self.cfg,self.ca,self.base,'cpu',y.ARMS)
            imposed,_=y.rollout(torch,self.net,geo,p,self.cfg,self.ca,self.base,'cpu',y.ARMS,ref)
            for key in ref:
                np.testing.assert_array_equal(ref[key][:,:,0],auto[key][:,:,0])
                np.testing.assert_array_equal(ref[key][:,:,0],imposed[key][:,:,0])
            mask=self.torch.tensor(y.channel_masks(4,('site_proximal',)),dtype=torch.bool)[None].expand(3,-1,-1,-1).reshape(3,4,11)
            local=y.local_core(torch,self.net,mask,self.cfg,self.ca)
            formula=y.local_core(torch,None,mask,self.cfg,self.ca)
            v=torch.tensor([[-55.,-45.,-65.,-60.]]*3);ca=torch.full_like(v,.0002)
            states=torch.full((3,4,18),.2);mult=torch.ones((3,4,11))
            a=local(v,ca,states,mult);b=formula(v,ca,states,mult)
            for first,second in zip(a,b):torch.testing.assert_close(first[:,1:],second[:,1:],rtol=0,atol=0)
        self.assertEqual(y.n.weights_hash(self.net),self.p['freeze']['logical_weights_sha256'])
    def test_score_and_paired_contrasts(self):
        torch=self.torch;p=dict(self.p,duration_ms=.3);geo=y.x.geometry(4)
        with torch.inference_mode():
            repeated,_=y.rollout(torch,None,geo,p,self.cfg,self.ca,self.base,'cpu',y.ARMS)
            ref={k:v[:,:,:1] for k,v in repeated.items()}
            pred,_=y.rollout(torch,self.net,geo,p,self.cfg,self.ca,self.base,'cpu',y.ARMS,ref)
        native={k:ref[k][:,0,0] for k in ('voltage','calcium','gates')}
        rows=[]
        for mode in ('imposed','autonomous'):rows.extend(y.score(pred,ref,native,geo,y.previous.design(geo,p)[0],y.ARMS,mode,p,self.cfg,True))
        self.assertEqual(len(rows),288)
        self.assertTrue(all(r['judgment_eligible'] and r['finite'] for r in rows))
        self.assertEqual(len(y.paired_contrasts(rows)),144)
        self.assertEqual(len(y.coupling_contrasts(rows)),192)
        interactions=y.family_interactions(pred,ref,geo,'imposed',p)
        self.assertEqual(len(interactions),72)
        self.assertTrue(all(np.isfinite(r['nonadditive_residual_rmse']) for r in interactions))
    def test_notebook_and_worker(self):
        sys.path.insert(0,str(ROOT/'scripts'))
        import task44_kaggle as k
        doc=k.notebook('a'*40,'b'*64)
        for cell in doc['cells']:
            if cell['cell_type']=='code':compile(''.join(cell['source']),'44-notebook','exec')
        text=json.dumps(doc);self.assertIn('run_task44.py',text);self.assertNotIn('run_task43.py',text)
        self.assertIn('neuron==8.2.7',text)
        self.assertEqual(k.launcher.SLUG,'alessandrobelli/giada-task44-composition-localization')
if __name__=='__main__':unittest.main()
