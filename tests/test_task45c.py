import copy,json,os,sys,tempfile,unittest
from pathlib import Path
import numpy as np
from unittest.mock import patch
from src.giada_teacher import task45c_preserving_kinetics as c
ROOT=Path(__file__).resolve().parents[1]

class PreservingKinetics(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        cls.torch=torch;torch.set_num_threads(1)
        os.environ['GIADA_TASK41N_FROZEN']=str(ROOT/'artifacts/task41n_frozen.zip.bin')
        cls.p,cls.cfg,cls.ca,cls.base=c.spec(ROOT)
        old=c.n.load_models(torch,ROOT,'cpu')[0]['dual'];cls.old=old
        with np.load(ROOT/cls.p['reference41r']['heads_path']) as z:cls.ref41=c.x.repair.materialize(torch,old,dict(z))
        with np.load(ROOT/cls.p['baseline']['heads_path']) as z:cls.baseline=c.b.materialize(torch,cls.ref41,torch.tensor(z['weights']),old,(0,1,1))
    def test_contract_and_histories(self):
        self.assertEqual(c.n.weights_hash(self.baseline),self.p['baseline']['logical_weights_sha256'])
        self.assertEqual(self.p['quality'],c.b.spec(ROOT)[0]['quality'])
        data=c.histories(self.p,'fit');self.assertEqual(data['voltage'].shape,(10000,24))
        self.assertLessEqual(data['voltage'].min(),-130);self.assertEqual(data['truth'].shape,(10000,24,18))
        self.assertTrue(np.isfinite(data['truth']).all());self.assertTrue(np.all((data['truth']>=0)&(data['truth']<=1)))
        self.assertEqual(len({r['input_seed'] for r in data['meta']}),3)
    def test_causal_jacobian_finite_difference(self):
        rng=np.random.default_rng(4503);T,P,H=10,2,4
        data=dict(features=rng.normal(size=(T,3,P,H)),raw=rng.normal(size=(T,3,P,18,2)),initial=rng.uniform(.2,.8,(P,18)))
        for joint in (False,True):
            delta=rng.normal(size=(3,H*(2 if joint else 1)))*.01;ticks=[0,4,9]
            state,jac=c.sensitivities(data,9,joint,delta,ticks)
            for j in range(delta.shape[1]):
                change=np.zeros_like(delta);change[:,j]=1e-6
                plus=c.sensitivities(data,9,joint,delta+change,ticks)[0];minus=c.sensitivities(data,9,joint,delta-change,ticks)[0]
                np.testing.assert_allclose((plus-minus)/2e-6,jac[...,j],rtol=2e-5,atol=1e-8)
    def test_head_isolation_and_packed_maps(self):
        torch=self.torch;models={'C0':self.baseline,'R41':self.ref41,'R111':self.baseline}
        for name,(gate,joint,_,_) in c.DEFINITIONS.items():
            weights=c.readout_weights(torch,self.baseline,gate,joint)+.0001;model=c.materialize(torch,self.baseline,gate,joint,weights);models[name]=model
            changed=[k for k,v in model.state_dict().items() if not torch.equal(v,self.baseline.state_dict()[k])]
            self.assertTrue(set(changed)<= {'parent.c','parent.d'})
            for col in (7,24):self.assertTrue(torch.equal(model.parent.c[:,:,col],self.baseline.parent.c[:,:,col]))
        models['J']=self.baseline
        with torch.inference_mode():
            adapter=c.batched_model(torch,self.baseline,models);self.assertTrue(all(r['passed'] for r in c.packed_parity(torch,adapter,models,36,self.p,'cpu')))
    def test_nomination_rejects_new_failure_not_just_count(self):
        row=dict(seed=17,case=0,horizon_ms=100.,passed=True,targets={'9':dict(passed=True,severity=.5)})
        bad=dict(row,passed=False,targets={'9':dict(passed=True,severity=.1)})
        self.assertEqual(c.nominate([('C0',0,[row]),('N00',1,[bad])],[row],9,['C0','N00']),('C0',0))
    def test_tiny_full_fit_preserves_real_treatment_arms(self):
        torch=self.torch;p=copy.deepcopy(self.p);p['fitting'].update(duration_ms=8.,points=128,sample_ticks=8);p['fitting_horizons_ms']=[1.,8.]
        with tempfile.TemporaryDirectory(dir=ROOT/'artifacts') as tmp:
            out=Path(tmp);models,freeze=c.fit(torch,self.baseline,p,out,'cpu')
            self.assertEqual(set(models),set(c.DEFINITIONS)|{'J'})
            self.assertTrue(freeze['all_parameters_finite']);self.assertFalse(freeze['selection_used_coupled'])
            for name in c.DEFINITIONS:self.assertEqual(freeze['per_arm_choices'][name]['source'],name)
            self.assertTrue((out/'N11_4.pt').is_file());self.assertTrue((out/'nomination_freeze.json').is_file())
        self.assertEqual(c.n.weights_hash(self.baseline),self.p['baseline']['logical_weights_sha256'])
    def test_notebook_compiles_and_uses_both_verified_inputs(self):
        sys.path.insert(0,str(ROOT/'scripts'));import task45c_kaggle as k
        doc=k.notebook('a'*40,'b'*64)
        for cell in doc['cells']:
            if cell['cell_type']=='code':compile(''.join(cell['source']),'45c-notebook','exec')
        text=json.dumps(doc);self.assertIn('run_task45c.py',text);self.assertIn('task45b_final.zip',text)
        self.assertNotIn('run_task41n_recovery.py',text);self.assertNotIn('nrnivmodl',text)
        self.assertIn('str(fixtures[0])',text)
    def test_short_complete_evaluator(self):
        torch=self.torch;p=copy.deepcopy(self.p);p.update(duration_ms=.4,prefix_horizons_ms=[.2,.4],geometries=[p['geometries'][0]],panels=['original45'])
        panel=dict(p,panel='original45');geo=p['geometries'][0]
        treatment={k:copy.deepcopy(self.baseline) for k in list(c.DEFINITIONS)+['J']}
        with tempfile.TemporaryDirectory(dir=ROOT/'artifacts') as tmp,torch.inference_mode():
            out=Path(tmp)/'out';out.mkdir();fixtures=out.parent/'verified_native45c';(fixtures/'original45').mkdir(parents=True)
            (fixtures/'calibration_freeze.json').write_text('{}');(fixtures/'original45/native_floor.json').write_text(json.dumps(dict(valid=True)))
            torch.save(self.baseline.state_dict(),fixtures/'f111.pt')
            with c.configured(panel):repeated,_=c.y.rollout(torch,None,geo,panel,self.cfg,self.ca,self.base,'cpu',c.ARMS)
            np.savez_compressed(fixtures/'original45/native_chain4.npz',**{k:repeated[k][:,0,0] for k in ('voltage','calcium','gates')})
            cache=out.parent/'verified_source45c';cache.mkdir();np.savez_compressed(cache/'native_shadow_traces.npz',voltage=np.zeros((2,2)))
            def fake_stress(*args):return dict(passed=64 if args[4]=='stress_R111' else 66,total=72,failed=[['known',i] for i in range(6)])
            with patch.object(c,'spec',return_value=(p,self.cfg,self.ca,self.base)),patch.object(c,'verify_fixtures'),patch.object(c,'initial_oracle_probes'),patch.object(c.x.repair.p.r,'verify_source'),patch.object(c.n,'load_models',return_value=({'dual':self.old},self.old)),patch.object(c,'fit',return_value=(treatment,dict(valid=True))),patch.object(c.b,'panel_configs',return_value=[panel]),patch.object(c.x.repair,'stress',side_effect=fake_stress),patch.object(c.n,'benchmark',return_value=[]),patch.dict(os.environ):
                report=c.evaluate(torch,ROOT,out/'unused',out/'fixtures',out,'cpu')
            self.assertTrue(report['complete'] and report['baseline_unchanged'])
            self.assertEqual(len(report['panels']['original45']['0.4']),12)
            self.assertTrue(report['all_parameters_finite'] and report['all_traces_finite'])

if __name__=='__main__':unittest.main()
