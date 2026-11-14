import copy,json,subprocess,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from tests import test_task45c as fixtures
from src.giada_teacher import task47_independent_confirmation as q
a=q.a;ROOT=fixtures.ROOT
class Confirmation47(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixtures.PreservingKinetics.setUpClass();cls.f=fixtures.PreservingKinetics;cls.torch=cls.f.torch;cls.p,cls.cfg,cls.ca,cls.base=q.spec(ROOT)
    def test_prospective_disjoint_and_paired(self):
        old=json.loads((ROOT/'experiments/task46_preregistration_v1.json').read_text());self.assertEqual(old['freeze'],self.p['freeze']);self.assertEqual(old['quality'],self.p['quality'])
        for geo in self.p['geometries']:
            self.assertNotIn(geo['parents'],[g['parents'] for g in old['geometries']]);rows,inj,mult=q.design(geo,self.p)
            self.assertEqual(len(rows),24);np.testing.assert_array_equal(inj,q.design(geo,self.p)[1])
            for i,row in enumerate(rows):
                if row['block']=='none':continue
                j=next(j for j,r in enumerate(rows) if r['block']=='none' and all(r[k]==row[k] for k in ('coupling','form','site_label','density')))
                np.testing.assert_array_equal(inj[:,i],inj[:,j]);blocked=[2,3,4] if row['block']=='sodium' else [5]
                self.assertFalse(mult[i,:,blocked].any());keep=[k for k in range(11) if k not in blocked];np.testing.assert_array_equal(mult[i][:,keep],mult[j][:,keep])
    def test_real_short_paired_worker(self):
        p=copy.deepcopy(self.p);p.update(duration_ms=.4,prefix_horizons_ms=[.2,.4],geometries=[p['geometries'][0]])
        t=self.torch
        with tempfile.TemporaryDirectory(dir=ROOT/'artifacts') as tmp,t.inference_mode():
            out=Path(tmp);cache=out/'cache';cache.mkdir();np.savez_compressed(cache/'native_shadow_traces.npz',voltage=np.zeros((2,2)))
            with q.configured(),patch.object(a,'spec',return_value=(p,self.cfg,self.ca,self.base)):
                with patch.object(a.x.repair.p.r,'verify_source'),patch.object(a.n,'load_models',return_value=({'dual':self.f.old},self.f.old)):
                    models,_,_=a.load_frozen(t,ROOT,out/'unused',ROOT/'artifacts/task45c_cost_input/checkpoint45c.zip.bin',ROOT/'artifacts/task45d_evaluation_input/evaluation45d.zip.bin',out,'cpu')
                with a.configured(ROOT):
                    ref,_=a.y.rollout(t,None,p['geometries'][0],p,self.cfg,self.ca,self.base,'cpu',a.ARMS)
                np.savez_compressed(out/f'native_{p["geometries"][0]["id"]}.npz',**{k:ref[k][:,0,0] for k in ('voltage','calcium','gates')});(out/'native_floor.json').write_text('{"valid":true}')
                with patch.object(a,'load_frozen',return_value=(models,self.f.old,cache)),patch.object(a.d,'ORIGINAL_BENCHMARK',return_value=[]),patch.object(a,'component_benchmark',return_value=[]):a.evaluate(t,ROOT,None,None,None,out,'cpu')
                sys.path.insert(0,str(ROOT/'scripts'));import run_task46 as worker
                for phase in ('driver','preflight','native','gpu'):a.n.p.save_json(out/f'executed_protocol_{phase}.json',worker.protocol_identity(p))
                with patch.object(q,'spec',return_value=(p,self.cfg,self.ca,self.base)):q.finalize(out)
                r=json.loads((out/'final_report.json').read_text());self.assertEqual(r['task'],'47');self.assertTrue(r['all_traces_finite']);self.assertTrue(r['executed_protocol_verified']);self.assertFalse(r['task48_authorized'])
    def test_notebook_and_phase_cache_fix(self):
        sys.path.insert(0,str(ROOT/'scripts'));import task47_kaggle as k
        code=''.join(k.notebook('a'*40)['cells'][1]['source']);compile(code,'47-notebook','exec')
        self.assertIn('run_task47.py',code);self.assertIn('task47_final',code);self.assertNotIn('run_task46.py',code);self.assertIn('GIADA_NATIVE_PYTHON',code)
    def test_real_child_routing_and_parent_identity_checks(self):
        sys.path.insert(0,str(ROOT/'scripts'));import run_task46 as worker
        original_run=subprocess.run;seen=[]
        with tempfile.TemporaryDirectory(dir=ROOT/'artifacts') as tmp:
            out=Path(tmp)/'out';entry=ROOT/'scripts/run_task47.py'
            def child(command,**kwargs):
                self.assertEqual(Path(command[2]).resolve(),entry.resolve());seen.append(command[-1])
                result=original_run([*command[:-1],'identity'],**kwargs)
                identity=json.loads((out/'executed_protocol_identity.json').read_text());self.assertEqual(identity['task'],'47');self.assertEqual([g['nodes'] for g in identity['geometries']],[5,9])
                a.n.p.save_json(out/f'executed_protocol_{command[-1]}.json',identity);return result
            with q.configured(),patch.object(sys,'argv',['worker','unused','unused','unused',str(out)]),patch.object(worker.subprocess,'run',side_effect=child):worker.main(entrypoint=entry)
            self.assertEqual(seen,['preflight','native','gpu'])
    def test_rejects_completed_but_wrong_protocol(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'artifacts') as tmp:
            out=Path(tmp);a.n.p.save_json(out/'final_report.json',dict(valid=True,complete=True,task='46'))
            with self.assertRaisesRegex(RuntimeError,'protocol identity audit failed'):q.finalize(out)
            r=json.loads((out/'final_report.json').read_text());self.assertFalse(r['valid']);self.assertFalse(r['complete']);self.assertTrue(r['technical_failure'])
if __name__=='__main__':unittest.main()
