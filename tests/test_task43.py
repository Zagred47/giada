import json,os,sys,unittest
from pathlib import Path
import numpy as np
from src.giada_teacher import task43_frozen_transfer as y
ROOT=Path(__file__).resolve().parents[1]

class FrozenTransfer(unittest.TestCase):
    def test_contract_and_design(self):
        config,cfg,ca,base=y.spec(ROOT)
        self.assertEqual(config['quality'],y.x.spec(ROOT)[0]['quality'])
        self.assertEqual(config['training_updates'],0)
        for geo in config['geometries']:
            rows,inj,mult=y.design(geo,config)
            self.assertEqual(inj.shape,(2000,12,len(geo['parents'])))
            self.assertEqual(len(rows),12)
            self.assertGreater(np.ptp(mult[0,0]),0)
            self.assertFalse(np.array_equal(inj[:,:,0],inj[:,:,-1]))
    def test_restored_context_and_matrix(self):
        original=y.x.spec;geo=y.x.geometry(5,True)
        area,lap=y.x.matrix(geo,1)
        p=np.random.default_rng(43005).permutation(5)
        with y.configured(ROOT,True):
            a,l=y.x.matrix(geo,1)
            np.testing.assert_array_equal(a,area[p]);np.testing.assert_array_equal(l,lap[np.ix_(p,p)])
        self.assertIs(y.x.spec,original)
    def test_frozen_cpu_permutation_and_imposed(self):
        import torch
        torch.set_num_threads(1)
        os.environ['GIADA_TASK41N_FROZEN']=str(ROOT/'artifacts/task41n_frozen.zip.bin')
        config,cfg,ca,base=y.spec(ROOT);config=dict(config,duration_ms=12.)
        geo=y.x.geometry(3,True)
        net=y.x.n.load_models(torch,ROOT,'cpu')[0]['dual']
        with np.load(ROOT/config['freeze']['heads_path']) as z:net=y.x.repair.materialize(torch,net,dict(z))
        self.assertEqual(y.x.n.weights_hash(net),config['freeze']['logical_weights_sha256'])
        with torch.inference_mode():
            outputs=[]
            for permuted in (False,True):
                with y.configured(ROOT,permuted):
                    formula,_=y.x.rollout(torch,None,geo,config,cfg,ca,base,'cpu')
                    autonomous,_=y.x.rollout(torch,net,geo,config,cfg,ca,base,'cpu')
                    imposed,_=y.x.rollout(torch,net,geo,config,cfg,ca,base,'cpu',formula)
                    outputs.append((formula,autonomous,imposed))
            for a,b in zip(*outputs):
                for key in a:
                    limit=y.permutation_limit(config,key)
                    np.testing.assert_allclose(a[key],b[key],rtol=0,atol=limit)
            self.assertGreater(np.ptp(outputs[0][0]['voltage'][-1,0,0]),.01)
        self.assertEqual(y.x.n.weights_hash(net),config['freeze']['logical_weights_sha256'])
    def test_notebook(self):
        sys.path.insert(0,str(ROOT/'scripts'))
        import task43_kaggle as k
        doc=k.notebook('a'*40,'b'*64)
        for cell in doc['cells']:
            if cell['cell_type']=='code':compile(''.join(cell['source']),'43','exec')
        text=json.dumps(doc)
        self.assertIn('run_task43.py',text);self.assertNotIn('run_task42.py',text)
        self.assertIn('neuron==8.2.7',text)
    def test_two_orders_use_independent_verified_caches(self):
        import tempfile
        from unittest.mock import patch
        with tempfile.TemporaryDirectory(dir=ROOT/'artifacts') as tmp:
            out=Path(tmp)/'result';out.mkdir();(out/'native_floor.json').write_text('{}')
            def evaluate(torch,root,source,folder,device):
                (folder.parent/'verified_source42').mkdir(exist_ok=False)
                (folder/'results.json').write_text(json.dumps([dict(mode='41r_autonomous',per_node=[dict(channels=[dict(channel='K_Pst',event=dict(status='missing_event'))])])]))
                for geo in y.spec(ROOT)[0]['geometries']:
                    np.savez_compressed(folder/('traces_'+geo['id']+'.npz'),formula_voltage=np.zeros((2,3,12,len(geo['parents']))))
                return dict(all_parameters_finite=True,candidate_weights_unchanged=True)
            with patch.object(y.x,'evaluate',evaluate):report=y.evaluate(None,ROOT,None,out,'cpu')
            self.assertTrue(report['permutation_passed'])
            self.assertEqual(report['event_support_by_order_mode_channel']['canonical/41r_autonomous/K_Pst']['missing_event'],1)
            self.assertFalse((out/'verified_source42').exists())
            for label in ('canonical','permuted'):self.assertTrue((out.parent/('verified_source43_'+label)).is_dir())
if __name__=='__main__':unittest.main()
