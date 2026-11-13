import copy,json,os,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from tests import test_task45c as fixtures
ROOT=fixtures.ROOT
from src.giada_teacher import task45e_anchored_repair as e
class AnchoredRepair(unittest.TestCase):
    @classmethod
    def setUpClass(cls):fixtures.PreservingKinetics.setUpClass();cls.torch=fixtures.PreservingKinetics.torch;cls.base=fixtures.PreservingKinetics.baseline
    def test_contract(self):
        p=e.spec(ROOT)[0];self.assertEqual(p['quality'],e.d.spec(ROOT)[0]['quality']);self.assertEqual(tuple(p['coupled_arms']),e.ARMS)
        self.assertNotEqual(p['fitting']['dev_seeds'],e.d.spec(ROOT)[0]['fitting']['dev_seeds'])
    def test_isolation_rejects_untargeted_change(self):
        torch=self.torch;weights=e.c.readout_weights(torch,self.base,9,False)+.0001;model=e.c.materialize(torch,self.base,9,False,weights)
        self.assertTrue(e.assert_preserved(torch,model,self.base,9,False)['valid'])
        with torch.no_grad():model.parent.d[:,3].add_(.0001)
        with self.assertRaises(AssertionError):e.assert_preserved(torch,model,self.base,9,False)
    def test_notebook(self):
        sys.path.insert(0,str(ROOT/'scripts'));import task45e_kaggle as k
        text=''.join(k.notebook('a'*40,'b'*64)['cells'][1]['source']);compile(text,'45e-notebook','exec')
        self.assertIn('run_task45e.py',text);self.assertIn('str(checkpoints[0]),str(saved[0]),str(out)',text)
        self.assertIn('threading.Timer(10800',text);self.assertNotIn('training_performed=False',text)
    def test_real_tiny_fit_and_transplant(self):
        p=copy.deepcopy(e.spec(ROOT)[0]);p['fitting'].update(duration_ms=.4,points=128,sample_ticks=4,checkpoints=[1]);p['fitting_horizons_ms']=[.2,.4]
        with tempfile.TemporaryDirectory(dir=ROOT/'artifacts') as tmp,patch.dict(os.environ,GIADA_TASK45C_CHECKPOINTS=str(ROOT/'artifacts/task45c_cost_input/checkpoint45c.zip.bin'),GIADA_TASK45D_SAVED=str(ROOT/'artifacts/task45d_evaluation_input/evaluation45d.zip.bin')):
            # Old1s audit is independently tested from archived rows; keep this unit fixture bounded.
            with patch.object(e.d,'spec',return_value=(p,fixtures.PreservingKinetics.cfg,fixtures.PreservingKinetics.ca,fixtures.PreservingKinetics.base)):
                models,f=e.fit(self.torch,self.base,p,Path(tmp),'cpu')
            self.assertEqual(set(models),{'C1','P','N00','N01','N10','N11','Iz','Iu','J'})
            self.assertTrue(f['valid']);self.assertFalse(f['selection_used_coupled']);self.assertTrue(all(q['valid'] for q in f['lineage_checks']))
            self.assertTrue(all(q['untargeted_tensors_bitwise_equal'] for q in f['lineage_checks']))
    def test_short_complete_evaluator(self):
        import numpy as np
        torch=self.torch;p=copy.deepcopy(e.spec(ROOT)[0]);p.update(duration_ms=.4,prefix_horizons_ms=[.2,.4],geometries=[p['geometries'][0]],panels=['original45'])
        panel=dict(p,panel='original45');geo=p['geometries'][0];fx=fixtures.PreservingKinetics
        treatment={k:copy.deepcopy(self.base) for k in ('C1','P','N00','N01','N10','N11','Iz','Iu','J')}
        with tempfile.TemporaryDirectory(dir=ROOT/'artifacts') as tmp,torch.inference_mode():
            out=Path(tmp)/'out';out.mkdir();native=out.parent/'verified_native45c';(native/'original45').mkdir(parents=True)
            (native/'calibration_freeze.json').write_text('{}');(native/'original45/native_floor.json').write_text(json.dumps(dict(valid=True)));torch.save(self.base.state_dict(),native/'f111.pt')
            with e.configured(panel):repeated,_=e.c.y.rollout(torch,None,geo,panel,fx.cfg,fx.ca,fx.base,'cpu',e.ARMS)
            np.savez_compressed(native/'original45/native_chain4.npz',**{k:repeated[k][:,0,0] for k in ('voltage','calcium','gates')})
            cache=out.parent/'verified_source45c';cache.mkdir();np.savez_compressed(cache/'native_shadow_traces.npz',voltage=np.zeros((2,2)))
            def stress(*args):return dict(passed=64 if args[4]=='stress_R111' else 66,total=72,failed=[['known',i] for i in range(6)])
            with patch.object(e,'spec',return_value=(p,fx.cfg,fx.ca,fx.base)),patch.object(e.d,'control_check',return_value=dict(mocked_for_short_fixture=True)),patch.object(e.c,'verify_fixtures'),patch.object(e.c,'initial_oracle_probes'),patch.object(e.c.x.repair.p.r,'verify_source'),patch.object(e.n,'load_models',return_value=({'dual':fx.old},fx.old)),patch.object(e,'fit',return_value=(treatment,dict(valid=True))),patch.object(e.c.b,'panel_configs',return_value=[panel]),patch.object(e.c.x.repair,'stress',side_effect=stress),patch.object(e.d,'ORIGINAL_BENCHMARK',return_value=[]),patch.dict(os.environ):
                report=e.evaluate(torch,ROOT,out/'unused',out/'nativeunused',ROOT/'artifacts/task45c_cost_input/checkpoint45c.zip.bin',ROOT/'artifacts/task45d_evaluation_input/evaluation45d.zip.bin',out,'cpu')
            self.assertTrue(report['complete'] and report['all_parameters_finite']);self.assertEqual(report['task'],'45e')
if __name__=='__main__':unittest.main()
