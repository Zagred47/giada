import ast,copy,hashlib,io,json,os,sys,tempfile,unittest,zipfile
from pathlib import Path
from unittest.mock import patch
import numpy as np
from tests import test_task45c as fixtures
from src.giada_teacher import task45d_evaluation_recovery as r
ROOT=fixtures.ROOT
class EvaluationRecovery(unittest.TestCase):
    def test_no_training_or_selection_calls_and_notebook(self):
        tree=ast.parse(Path(r.__file__).read_text());calls={n.func.attr for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)}
        self.assertFalse(calls&{'fit','choose','nominate','train_loss','backward','step'})
        sys.path.insert(0,str(ROOT/'scripts'));import task45d_evaluation_kaggle as k
        text=''.join(k.notebook('a'*40,'b'*64)['cells'][1]['source']);compile(text,'evaluation-recovery','exec')
        self.assertIn('run_task45d_evaluation_recovery.py',text);self.assertIn('str(saved[0]),str(checkpoints[0])',text);self.assertIn('selection_recomputed=False',text);self.assertNotIn('selection_recomputed=(out/',text)
    def test_skip_completed_block_and_load_saved_freeze(self):
        fixtures.PreservingKinetics.setUpClass();f=fixtures.PreservingKinetics;t=f.torch;p=copy.deepcopy(r.d.spec(ROOT)[0]);p.update(duration_ms=.4,prefix_horizons_ms=[.2,.4],geometries=[p['geometries'][0]],panels=['original45']);panel=dict(p,panel='original45');geo=p['geometries'][0]
        models={k:copy.deepcopy(f.baseline) for k in r.d.ARMS[1:]};cfg,ca,base=f.cfg,f.ca,f.base
        with tempfile.TemporaryDirectory(dir=ROOT/'artifacts') as tmp,t.inference_mode():
            tmp=Path(tmp);out=tmp/'out';out.mkdir();native=tmp/'verified_native45c';(native/'original45').mkdir(parents=True);(native/'original45/native_floor.json').write_text('{"valid":true}')
            with r.d.configured(panel):pred,_=r.c.y.rollout(t,None,geo,panel,cfg,ca,base,'cpu',r.d.ARMS)
            ref={k:v[:,:,:1] for k,v in pred.items()};physical=r.c.b.design(geo,panel)[0]
            np.savez_compressed(native/'original45/native_chain4.npz',**{k:pred[k][:,0,0] for k in ('voltage','calcium','gates')})
            saved=[]
            for h in (.2,.4):
                ticks=round(h/.1);values=r.c.y.score(r.c.b.a.crop(pred,ticks),r.c.b.a.crop(ref,ticks),{k:v[:ticks+1,0,0] for k,v in pred.items() if k in ('voltage','calcium','gates')},geo,physical,r.d.ARMS,'imposed',panel,cfg,True)
                for q in values:q.update(panel='original45',horizon_ms=h,reference_valid=True,regression_vs_f011=False)
                saved.extend(values)
            packed=r.d.adapter(t,f.baseline,models);archive=tmp/'saved.zip';members={}
            freeze=dict(weights_sha256={k:r.n.weights_hash(v) for k,v in models.items()},chosen=dict(arm='C1',checkpoint=0))
            with zipfile.ZipFile(archive,'w') as z:
                def add(name,data):z.writestr(name,data);members[name]=hashlib.sha256(data).hexdigest()
                for name in models:
                    stream=io.BytesIO();t.save(models[name].state_dict(),stream);add(name+'.pt',stream.getvalue())
                for name,data in {'nomination_freeze.json':freeze,'calibration_freeze.json':{},'execution.json':{'original45/chain4/imposed':{}},'packed_execution_preflight.json':packed.execution_preflight,'GPU_training_transition_preflight.json':{'valid':True},'original45/results.json':saved}.items():add(name,json.dumps(data).encode())
            plan=dict(sha256=r.c.sha(archive),members=members,completed_blocks=['original45/chain4/imposed'],total_blocks=2)
            cache=tmp/'verified_source45c';cache.mkdir();np.savez_compressed(cache/'native_shadow_traces.npz',voltage=np.zeros((2,2)))
            seen=[];rollout=r.c.y.rollout
            def track(*args,**kwargs):
                if args[1] is not None:seen.append('imposed' if len(args)>9 and args[9] is not None else 'autonomous')
                return rollout(*args,**kwargs)
            def stress(*args):return dict(passed=64 if args[4]=='stress_R111' else 66,total=72,failed=[['known',i] for i in range(6)])
            with patch.object(r,'load_plan',return_value=plan),patch.object(r.d,'spec',return_value=(p,cfg,ca,base)),patch.object(r.c.x.repair.p.r,'verify_source'),patch.object(r.c,'verify_fixtures'),patch.object(r.n,'load_models',return_value=({'dual':f.old},f.old)),patch.object(r.c.b,'panel_configs',return_value=[panel]),patch.object(r.c.y,'rollout',side_effect=track),patch.object(r.c.x.repair,'stress',side_effect=stress),patch.object(r.d,'ORIGINAL_BENCHMARK',return_value=[{'finite':True}]*6),patch.object(r.d,'control_check',return_value={'mocked_for_short_fixture':True}),patch.object(r.d,'fit',side_effect=AssertionError('No fit allowed')),patch.object(r.d,'choose',side_effect=AssertionError('No selection allowed')),patch.dict(os.environ):
                report=r.recover(t,ROOT,tmp/'unused',tmp/'unused2',archive,ROOT/'artifacts/task45c_cost_input/checkpoint45c.zip.bin',out,'cpu')
            self.assertEqual(seen,['autonomous']);self.assertTrue(report['complete'] and report['nomination_bytes_unchanged']);self.assertFalse(report['training_performed'] or report['selection_recomputed'])
            self.assertEqual(report['reused_blocks'],['original45/chain4/imposed'])
if __name__=='__main__':unittest.main()
