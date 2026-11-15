import copy,gzip,json,sys,tempfile,unittest
from pathlib import Path
import numpy as np
from src.giada_teacher import task49_causal_initialization as q
ROOT=Path(__file__).resolve().parents[1]

class CausalInitialization(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        cls.torch=torch;torch.set_num_threads(1);cls.p=q.plan(ROOT)
    def test_roles_and_active_cuts(self):
        v,c,meta=q.histories(self.p);self.assertEqual(v.shape,(4000,84));self.assertEqual(len({m['history'] for m in meta}),84)
        self.assertEqual({m['input_seed'] for m in meta},{9491103,9491127,9491141})
        mask=q.active_mask(self.p,3000);self.assertFalse(mask[:,0].any());self.assertTrue(mask[-1,1:].all());self.assertFalse(q.active_mask(self.p,4000).any())
        self.assertFalse(q.active_mask(self.p,0)[0].any())
    def test_fit_only_codec_and_full_dimension_identity(self):
        v,c,meta=q.histories(self.p);rng=np.random.default_rng(1);states=rng.uniform(.1,.9,(3,len(meta),18))
        codec=q.codec_fit(states,meta,self.p);changed=states.copy();changed[:,[m['role']!='fit' for m in meta]]=999
        self.assertEqual(codec,q.codec_fit(changed,meta,self.p))
        full={k:len(row['gates']) for k,row in codec.items()};decoded,_=q.codec_decode(states,codec,full);np.testing.assert_allclose(decoded,states,atol=1e-14)
        for widths in self.p['codec_widths'].values():
            decoded,latent=q.codec_decode(states,codec,widths);self.assertEqual(latent.shape[-1],sum(widths.values()));self.assertTrue(((decoded>=0)&(decoded<=1)).all())
    def test_notebook49_only(self):
        sys.path.insert(0,str(ROOT/'scripts'));import task49_kaggle as k
        doc=k.notebook('a'*40);text=json.dumps(doc)
        for cell in doc['cells']:
            if cell['cell_type']=='code':compile(''.join(cell['source']),'49','exec')
        self.assertIn('run_task49.py',text);self.assertIn('task49_final',text);self.assertNotIn('run_task48.py',text);self.assertNotIn('run_task46.py',text)
    def test_future_mutation_cannot_change_earlier_warmup(self):
        torch=self.torch;p=copy.deepcopy(self.p);p.update(cuts_ms=[.1,.2,.4]);v=np.full((4,2),-65.);ca=np.full_like(v,5e-5)
        class Dummy:
            def raw(self,v,ca):return v[...,None].expand(*v.shape,18)*.01,torch.zeros((*v.shape,18))
            def __call__(self,v,ca,state):return q.coefficients_update(torch,*self.raw(v,ca),state)
        with torch.inference_mode():
            first=q.condition(torch,Dummy(),v,ca,[{},{}],p,'cpu')[0]
            v[1:]=20.;ca[1:]=.001;second=q.condition(torch,Dummy(),v,ca,[{},{}],p,'cpu')[0]
        np.testing.assert_array_equal(first['warmup'][0],second['warmup'][0]);np.testing.assert_array_equal(first['P'][0],second['P'][0])
    def test_cost_paths(self):
        p=copy.deepcopy(self.p);p.update(warmup_ms=[0,.1]);p['cost'].update(repetitions=1,batch_histories=2)
        _,_,meta=q.histories(self.p);states=np.random.default_rng(2).uniform(0,1,(3,len(meta),18));codec=q.codec_fit(states,meta,p)
        with self.torch.inference_mode():result=q.cost(self.torch,lambda v,c,g:g,codec,p,'cpu')
        self.assertEqual(set(result['samples_ms']),{'warmup_0','warmup_0.1','family_one','family_medium','physical_identity'})
        self.assertTrue(all(s[0]>=0 for s in result['samples_ms'].values()))
    def test_full_small_real_P_pipeline(self):
        p=copy.deepcopy(self.p);p.update(history_ms=.4,cuts_ms=[.1,.2,.4],horizons_ms=[.1,.2],future_protocols=['rest']);p['cost'].update(repetitions=1,batch_histories=2)
        for v in p['roles'].values():v['count']=6
        # Keep public policy names intact but omit long replay from this CPU mini-run.
        from unittest.mock import patch
        with tempfile.TemporaryDirectory(dir=ROOT/'artifacts') as tmp,self.torch.inference_mode(),patch.object(q,'cost',return_value=dict(valid=True,test_only=True)):
            out=Path(tmp)/'out';out.mkdir()
            r=q.run(self.torch,ROOT,ROOT/'artifacts/task41n_primary_external/recovery_final.zip',ROOT/'artifacts/task45c_cost_input/checkpoint45c.zip.bin',ROOT/'artifacts/task45d_evaluation_input/evaluation45d.zip.bin',out,'cpu',p)
            self.assertTrue(r['valid'] and r['restart_bitwise_identity'] and r['complete_prefix_warmup_bitwise_identity'])
            with gzip.open(out/'metrics_rest.json.gz','rt') as stream:rows=json.load(stream)
            self.assertEqual({r['arm'] for r in rows if 'arm' in r},set(p['arms']))
            self.assertEqual(r['P_weights_hash_before'],r['P_weights_hash_after'])
            self.assertEqual(json.loads((out/'executed_protocol.json').read_text())['task'],'49')
if __name__=='__main__':unittest.main()
