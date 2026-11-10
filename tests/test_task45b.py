import json,os,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from src.giada_teacher import task45b_rate_coverage as b
ROOT=Path(__file__).resolve().parents[1]

class RateCoverage(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        cls.torch=torch;torch.set_num_threads(1)
        os.environ['GIADA_TASK41N_FROZEN']=str(ROOT/'artifacts/task41n_frozen.zip.bin')
        cls.p,cls.cfg,cls.ca,cls.base=b.spec(ROOT)
        cls.old=b.n.load_models(torch,ROOT,'cpu')[0]['dual']
        with np.load(ROOT/cls.p['freeze']['heads_path']) as z:cls.frozen=b.x.repair.materialize(torch,cls.old,dict(z))
    def test_inputs_and_no_threshold_change(self):
        fit,dev=b.fit_inputs(self.p)
        self.assertEqual(fit.shape,(2,2048));self.assertEqual(dev.shape,(512,))
        self.assertEqual(int((fit[1]<-100).sum()),1024)
        self.assertFalse(set(fit.flatten())&set(dev))
        self.assertEqual(self.p['quality'],b.a.spec(ROOT)[0]['quality'])
        self.assertEqual(self.p['freeze']['logical_weights_sha256'],b.n.weights_hash(self.frozen))
    def test_three_head_materialization_and_adapter(self):
        torch=self.torch;ih=b.head_weights(torch,self.frozen)+.0001
        models={arm:b.materialize(torch,self.frozen,ih,self.old,bit) for arm,bit in zip(b.ARMS[1:],b.BITS)}
        self.assertEqual(b.n.weights_hash(models['f000']),b.n.weights_hash(self.frozen))
        adapter=b.batched_model(torch,self.frozen,models)
        with torch.inference_mode():self.assertTrue(all(r['passed'] for r in b.adapter_preflight(torch,adapter,models,36,self.p,'cpu')))
        v=torch.linspace(-130,45,15).repeat(9)[None].expand(3,-1);ca=torch.full_like(v,.0002)
        with torch.inference_mode():
            full=adapter.raw(v,ca);frozen=self.frozen.raw(v,ca)
            for ai,arm in enumerate(b.ARMS):
                if arm=='none':continue
                vv=v[:,15*ai:15*(ai+1)];cc=ca[:,15*ai:15*(ai+1)]
                wanted=models[arm].raw(vv,cc)
                for got,expected in zip(full,wanted):torch.testing.assert_close(got[:,15*ai:15*(ai+1)],expected,rtol=0,atol=1e-5)
                for got,expected in zip(full,frozen):
                    if arm=='f000':self.assertTrue(torch.equal(got[:,15*ai:15*(ai+1)],expected[:,15*ai:15*(ai+1)]))
        for arm,model in models.items():
            changed=[]
            for key,value in model.state_dict().items():
                original=self.frozen.state_dict()[key]
                if not torch.equal(value,original):changed.append(key)
            self.assertTrue(set(changed)<= {'parent.c','parent.d'})
        self.assertEqual(b.n.weights_hash(self.frozen),self.p['freeze']['logical_weights_sha256'])
    def test_design_context_and_all_masks(self):
        p=dict(self.p,panel='safe_hyper',calibrated_amplitudes=[-.001,-.003,-.005]);geo=p['geometries'][0]
        rows,inj,m=b.design(geo,p);self.assertEqual(inj.shape,(10000,9,4))
        np.testing.assert_array_equal(inj[500,0:3,0],[-.001,-.003,-.005])
        np.testing.assert_array_equal(inj[500,:3,0],inj[500,3:6,0])
        before=b.y.spec,b.y.previous.design,b.y.channel_masks
        with b.configured(p):
            mask=b.y.channel_masks(4,b.ARMS);self.assertFalse(mask[0].any());self.assertTrue(mask[1:].all())
            with self.torch.inference_mode():result=b.y.endpoint_preflight(self.torch,ROOT,self.frozen,geo,p,self.cfg,self.ca,self.base,'cpu')
            self.assertTrue(all(r['passed'] for r in result))
        self.assertEqual(before,(b.y.spec,b.y.previous.design,b.y.channel_masks))
    def test_calibration_only_teacher_and_safe_side(self):
        p=json.loads(json.dumps(self.p));p['duration_ms']=210.
        def fake(h,geo,row,inj,mult,*unused):return dict(voltage=np.full((3,4),-65+10000*inj.min()))
        with patch.object(b.x,'native_episode',side_effect=fake):r=b.calibrate(None,p,self.cfg,self.ca,self.base)
        self.assertTrue(r['teacher_only']);self.assertFalse(r['candidate_outcomes_accessed'])
        self.assertEqual(len(r['amplitudes_ma_cm2']),3)
        for amp,target in zip(r['amplitudes_ma_cm2'],[-90,-110,-130]):self.assertGreaterEqual(-65+10000*amp,target)
    def test_short_training_and_path_oracles(self):
        torch=self.torch;p=json.loads(json.dumps(self.p));p['fit'].update(points=128,dev_points_per_bin=16,checkpoints=[2,4]);p['atomic']['horizons_ms']=[.2,.4]
        with tempfile.TemporaryDirectory(dir=ROOT/'artifacts') as tmp:
            out=Path(tmp);weights,freeze=b.train(torch,self.frozen,p,out,'cpu')
            self.assertTrue(torch.isfinite(weights).all());self.assertFalse(freeze['selection_uses_coupled_outcomes'])
            self.assertEqual(len(freeze['all_candidates']),4)
            self.assertTrue((out/'training_4.pt').is_file())
            short=dict(p,duration_ms=.4,panel='original45');geo=p['geometries'][0]
            models={arm:b.materialize(torch,self.frozen,weights,self.old,bit) for arm,bit in zip(b.ARMS[1:],b.BITS)}
            with torch.inference_mode(),b.configured(short):
                repeated,_=b.y.rollout(torch,None,geo,short,self.cfg,self.ca,self.base,'cpu',b.ARMS)
                ref={k:v[:,:,:1].copy() for k,v in repeated.items()}
                adapter=b.batched_model(torch,self.frozen,models)
                pred,_=b.y.rollout(torch,adapter,geo,short,self.cfg,self.ca,self.base,'cpu',b.ARMS,ref)
                for key in ref:np.testing.assert_array_equal(pred[key][:,:,0],ref[key][:,:,0])
                probe=b.atomic_paths(torch,self.frozen,self.old,ref,p,out,'unit','cpu')
            both=[r for r in probe['controls'] if r['control']=='oracle_both']
            self.assertTrue(both);self.assertLess(max(np.max(r['free_gate_endpoint_abs']) for r in both),1e-12)
            b.save_targeted(out/'targeted.npz',pred)
            with np.load(out/'targeted.npz') as z:self.assertEqual(z['gates_Nap_h_Ih_m'].shape[-1],2)
    def test_notebook_compiles_and_has_real_worker(self):
        sys.path.insert(0,str(ROOT/'scripts'));import task45b_kaggle as k
        doc=k.notebook('a'*40,'b'*64)
        for cell in doc['cells']:
            if cell['cell_type']=='code':compile(''.join(cell['source']),'45b-notebook','exec')
        text=json.dumps(doc);self.assertIn('run_task45b.py',text);self.assertNotIn('run_task45.py',text)
        self.assertIn('neuron==8.2.7',text);self.assertIn('repair_freeze.json',text)
        self.assertEqual(k.launcher.SLUG,'alessandrobelli/giada-task45b-rate-coverage')
    def test_full_short_pipeline_preserves_all_arms(self):
        torch=self.torch;p=json.loads(json.dumps(self.p));p.update(duration_ms=.4,prefix_horizons_ms=[.2,.4],geometries=[p['geometries'][0]],panels=['original45'])
        p['atomic']['horizons_ms']=[.2,.4];weights=b.head_weights(torch,self.frozen)
        with tempfile.TemporaryDirectory(dir=ROOT/'artifacts') as tmp:
            out=Path(tmp);dest=out/'original45';dest.mkdir()
            (out/'calibration_freeze.json').write_text(json.dumps(dict(valid=True,teacher_only=True,candidate_outcomes_accessed=False)))
            (dest/'native_floor.json').write_text(json.dumps(dict(valid=True)))
            panel=dict(p,panel='original45')
            with torch.inference_mode(),b.configured(panel):repeated,_=b.y.rollout(torch,None,p['geometries'][0],panel,self.cfg,self.ca,self.base,'cpu',b.ARMS)
            np.savez_compressed(dest/'native_chain4.npz',**{k:repeated[k][:,0,0] for k in ('voltage','calcium','gates')})
            stress=dict(passed=66,total=72,failed=[['known',i] for i in range(6)])
            with patch.object(b,'spec',return_value=(p,self.cfg,self.ca,self.base)),patch.object(b,'panel_configs',return_value=[panel]),patch.object(b.x.repair.p.r,'verify_source'),patch.object(b.n,'load_models',return_value=({'dual':self.old},None)),patch.object(b,'train',return_value=(weights,dict(valid=True))),patch.object(b.x.repair,'stress',side_effect=lambda *args:json.loads(json.dumps(stress))),patch.dict(os.environ):
                report=b.evaluate(torch,ROOT,out/'fixture_source',out,'cpu')
            self.assertTrue(report['complete'] and report['frozen41r_unchanged'])
            self.assertEqual(set(report['panels']['original45']['0.4']),set(b.ARMS[1:]))
            self.assertTrue((out/'constant_clamp_probes.json').is_file())
            self.assertTrue((dest/'timing_chain4_autonomous.json').is_file())
            self.assertEqual(b.n.weights_hash(self.frozen),self.p['freeze']['logical_weights_sha256'])

if __name__=='__main__':unittest.main()
