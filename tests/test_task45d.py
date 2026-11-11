import copy,json,os,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from tests import test_task45c as fixtures
from src.giada_teacher import task45d_learnability_execution as d
ROOT=fixtures.ROOT
class Task45d(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixtures.PreservingKinetics.setUpClass();f=fixtures.PreservingKinetics;cls.torch=f.torch;cls.base=f.baseline;cls.p=d.spec(ROOT)[0]
    def tiny(self):
        p=copy.deepcopy(self.p);p['fitting'].update(duration_ms=.4,points=128,checkpoints=[1,2],sample_ticks=4);p['fitting_horizons_ms']=[.2,.4]
        p['matrix'].update(atomic_checkpoints=[1,2],batch=16,held_ticks=[1,10]);return p
    def test_contract(self):
        self.assertEqual(self.p['quality'],d.c.spec(ROOT)[0]['quality']);self.assertEqual(self.p['matrix']['atomic_checkpoints'],[250,1000,4000])
        self.assertEqual(tuple(self.p['coupled_arms']),d.ARMS);self.assertEqual(self.p['fitting']['checkpoints'],[1,4,16,64])
    def test_notebook_inputs_and_timers(self):
        import sys
        sys.path.insert(0,str(ROOT/'scripts'));import task45d_kaggle as k
        document=k.notebook('a'*40,'b'*64)
        text=''.join(document['cells'][1]['source']);compile(text,'45d-notebook','exec')
        self.assertIn('run_task45d.py',text);self.assertIn('str(native[0]),str(checkpoints[0])',text);self.assertIn('threading.Timer(10800',text)
        self.assertIn('12000',str(k.k.main.__code__.co_consts));self.assertNotIn('training_updates_performed=0',text)
        self.assertIn(420,k.k.main.__code__.co_consts)
    def test_feature_training_and_real_materialization(self):
        torch=self.torch;p=self.tiny();data=d.c.histories(p,'fit');po=d.pool(torch,self.base,p,data,0,'cpu')
        matched=d.pool(torch,self.base,p,data,1,'cpu');self.assertGreater(matched['v'].max(),50);self.assertLess(matched['v'].min(),-130)
        for adapt in (False,True):
            view=d.training_view(torch,self.base,po,p,adapt,'cpu');t={k:torch.tensor(po[k],dtype=torch.float32 if k in ('v','ca') else torch.float64) for k in ('v','ca','u','z','raw','label')};t['initial']=torch.full((128,18),.5,dtype=torch.float64)
            loss=d.train_loss(torch,view,t,torch.arange(16),p);self.assertTrue(bool(torch.isfinite(loss).all()));loss.sum().backward();self.assertTrue(bool(torch.isfinite(view.theta.grad).all()))
            self.assertEqual(view.model.parent.a.grad is not None,adapt);model=view.materialize();self.assertTrue(all(bool(torch.isfinite(q).all()) for q in model.parameters()))
            self.assertEqual(d.n.weights_hash(model),d.n.weights_hash(self.base))
    def test_full_tiny_fit_all_real_arms(self):
        torch=self.torch;p=self.tiny()
        with tempfile.TemporaryDirectory(dir=ROOT/'artifacts') as tmp,patch.dict(os.environ,GIADA_TASK45C_CHECKPOINTS=str(ROOT/'artifacts/task45c_cost_input/checkpoint45c.zip.bin')):
            models,freeze=d.fit(torch,self.base,p,Path(tmp),'cpu')
            self.assertEqual(set(models),set(('C1','A00','A01','A10','A11','Ncap','Nline','T0','T1','J')))
            self.assertTrue(freeze['valid']);self.assertFalse(freeze['selection_used_coupled']);self.assertTrue((Path(tmp)/'A11_2.pt').exists())
            packed=d.adapter(torch,self.base,{**models,'C0':self.base,'R41':self.base,'R111':self.base});count=3
            v=torch.linspace(-100,50,count)[None].expand(3,-1).repeat(1,len(d.ARMS));ca=torch.full_like(v,1e-4);got=packed.raw(v,ca)
            for ai,name in enumerate(d.ARMS):
                net=self.base if name=='none' or name in ('C0','R41','R111') else models[name];want=net.raw(v[:,ai*count:(ai+1)*count],ca[:,ai*count:(ai+1)*count])
                for a,b in zip(want,got):self.assertTrue(torch.isfinite(b).all())
                # Same acceptance as original packed/materialized map test;
                # exact-execution claims are tested separately, never inferred.
                wa=.37+torch.sigmoid(want[1])*(torch.sigmoid(want[0])-.37);sl=slice(ai*count,(ai+1)*count);ga=.37+torch.sigmoid(got[1][:,sl])*(torch.sigmoid(got[0][:,sl])-.37)
                self.assertLessEqual(float(abs(wa-ga).max()),self.p['quality']['occupancy_atol'])
    def test_short_complete_evaluator(self):
        torch=self.torch;p=self.tiny();p.update(duration_ms=.4,prefix_horizons_ms=[.2,.4],geometries=[p['geometries'][0]],panels=['original45']);panel=dict(p,panel='original45');geo=p['geometries'][0]
        treatment={k:copy.deepcopy(self.base) for k in ('C1','A00','A01','A10','A11','Ncap','Nline','T0','T1','J')}
        with tempfile.TemporaryDirectory(dir=ROOT/'artifacts') as tmp,torch.inference_mode():
            out=Path(tmp)/'out';out.mkdir();native=out.parent/'verified_native45c';(native/'original45').mkdir(parents=True);(native/'calibration_freeze.json').write_text('{}');(native/'original45/native_floor.json').write_text(json.dumps(dict(valid=True)));torch.save(self.base.state_dict(),native/'f111.pt')
            with d.configured(panel):repeated,_=d.c.y.rollout(torch,None,geo,panel,fixtures.PreservingKinetics.cfg,fixtures.PreservingKinetics.ca,fixtures.PreservingKinetics.base,'cpu',d.ARMS)
            np.savez_compressed(native/'original45/native_chain4.npz',**{k:repeated[k][:,0,0] for k in ('voltage','calcium','gates')})
            cache=out.parent/'verified_source45c';cache.mkdir();np.savez_compressed(cache/'native_shadow_traces.npz',voltage=np.zeros((2,2)))
            def stress(*args):return dict(passed=64 if args[4]=='stress_R111' else 66,total=72,failed=[['known',i] for i in range(6)])
            with patch.object(d,'spec',return_value=(p,fixtures.PreservingKinetics.cfg,fixtures.PreservingKinetics.ca,fixtures.PreservingKinetics.base)),patch.object(d,'control_check',return_value=dict(mocked_for_short_fixture=True)),patch.object(d.c,'verify_fixtures'),patch.object(d.c,'initial_oracle_probes'),patch.object(d.c.x.repair.p.r,'verify_source'),patch.object(d.n,'load_models',return_value=({'dual':fixtures.PreservingKinetics.old},fixtures.PreservingKinetics.old)),patch.object(d,'fit',return_value=(treatment,dict(valid=True))),patch.object(d.c.b,'panel_configs',return_value=[panel]),patch.object(d.c.x.repair,'stress',side_effect=stress),patch.object(d.n,'benchmark',return_value=[]),patch.dict(os.environ):
                report=d.evaluate(torch,ROOT,out/'unused',out/'unusednative',ROOT/'artifacts/task45c_cost_input/checkpoint45c.zip.bin',out,'cpu')
            self.assertTrue(report['complete'] and report['all_parameters_finite']);self.assertEqual(report['task'],'45d');self.assertEqual(len(report['panels']['original45']['0.4']),13)
    def test_control_reproduction_rejects_changed_counts(self):
        import zipfile
        path=ROOT/'artifacts/task45c_cost_input/checkpoint45c.zip.bin'
        with zipfile.ZipFile(path) as z:old=json.loads(z.read('accuracy_summary.json'))['panels']
        report=dict(panels=copy.deepcopy(old))
        for panel in report['panels'].values():
            for rows in panel.values():rows['C1']=copy.deepcopy(rows['J'])
        self.assertTrue(d.control_check(report,path)['valid'])
        report['panels']['original45']['1000.0']['C0']['autonomous']['passed']-=1
        with self.assertRaises(AssertionError):d.control_check(report,path)
if __name__=='__main__':unittest.main()
