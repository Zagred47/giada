import copy,json,os,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from tests import test_task45c as fixtures
from src.giada_teacher import task46_spatial_cost as a
ROOT=fixtures.ROOT
class Spatial46(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixtures.PreservingKinetics.setUpClass();f=fixtures.PreservingKinetics;cls.torch=f.torch;cls.f=f;cls.p,cls.cfg,cls.ca,cls.base=a.spec(ROOT)
    def test_design_contrasts_and_block_isolation(self):
        p=self.p;geo=p['geometries'][0];rows,inj,mult=a.design(geo,p)
        self.assertEqual(len(rows),24);self.assertEqual(inj.shape,(10000,24,len(geo['parents'])))
        self.assertEqual(p['quality'],a.n.contract(ROOT)[0]['quality']);self.assertFalse(p['selection_performed']);self.assertEqual(p['training_updates'],0)
        for index,row in enumerate(rows):
            if row['block']=='none':continue
            j=next(i for i,q in enumerate(rows) if all(q[k]==row[k] for k in ('coupling','form','site_label','density')) and q['block']=='none')
            np.testing.assert_array_equal(inj[:,index],inj[:,j]);columns=[2,3,4] if row['block']=='sodium' else [5]
            self.assertFalse(mult[index,:,columns].any());keep=[k for k in range(11) if k not in columns];np.testing.assert_array_equal(mult[index][:,keep],mult[j][:,keep])
        self.assertEqual({q['coupling'] for q in rows},{0.,.25})
    def test_frozen_recipe_and_short_complete_pipeline(self):
        t=self.torch;p=copy.deepcopy(self.p);p.update(duration_ms=.4,prefix_horizons_ms=[.2,.4],geometries=[p['geometries'][0]]);p['cost'].update(ticks=1,repetitions=1)
        with tempfile.TemporaryDirectory(dir=ROOT/'artifacts') as tmp,t.inference_mode(),patch.dict(os.environ):
            out=Path(tmp)/'out';out.mkdir();cache=out.parent/'verified_source46';cache.mkdir();np.savez_compressed(cache/'native_shadow_traces.npz',voltage=np.zeros((2,2)))
            with patch.object(a,'spec',return_value=(p,self.cfg,self.ca,self.base)),patch.object(a.x.repair.p.r,'verify_source'),patch.object(a.n,'load_models',return_value=({'dual':self.f.old},self.f.old)):
                models,_,_=a.load_frozen(t,ROOT,out/'unused',ROOT/'artifacts/task45c_cost_input/checkpoint45c.zip.bin',ROOT/'artifacts/task45d_evaluation_input/evaluation45d.zip.bin',out,'cpu')
            self.assertEqual(a.n.weights_hash(models['P']),self.p['freeze']['logical_weights_sha256'])
            with a.configured(ROOT),patch.object(a,'spec',return_value=(p,self.cfg,self.ca,self.base)):
                repeated,_=a.y.rollout(t,None,p['geometries'][0],p,self.cfg,self.ca,self.base,'cpu',a.ARMS)
                np.savez_compressed(out/f'native_{p["geometries"][0]["id"]}.npz',**{k:repeated[k][:,0,0] for k in ('voltage','calcium','gates')})
                component=a.component_benchmark(t,models,p,self.cfg,self.ca,self.base,out,'cpu');self.assertTrue(all(q['axial_component_bitwise_equal'] for q in component[0]['parity']))
            (out/'native_floor.json').write_text('{"valid":true}')
            with patch.object(a,'spec',return_value=(p,self.cfg,self.ca,self.base)),patch.object(a,'load_frozen',return_value=(models,self.f.old,cache)),patch.object(a.d,'ORIGINAL_BENCHMARK',return_value=[{}]*6),patch.object(a,'component_benchmark',return_value=component):
                report=a.evaluate(t,ROOT,out/'unused',out/'unused2',out/'unused3',out,'cpu')
            self.assertTrue(report['complete'] and report['candidate_weights_unchanged'] and report['all_traces_finite']);self.assertFalse(report['task45e_passed'] or report['task47_authorized'])
            self.assertEqual(report['summary']['0.4']['P']['autonomous']['total'],72)
    def test_notebook_inputs_and_native_isolation(self):
        sys.path.insert(0,str(ROOT/'scripts'));import task46_kaggle as k
        text=''.join(k.notebook('a'*40,'b'*64)['cells'][1]['source']);compile(text,'46-notebook','exec')
        self.assertIn('run_task46.py',text);self.assertIn('neuron==8.2.7',text);self.assertIn('evaluation45d.zip.bin',text);self.assertIn('checkpoint45c.zip.bin',text)
        self.assertIn('str(checkpoints[0]),str(saved[0]),str(out)',text);self.assertIn('threading.Timer(10800',text);self.assertNotIn('dataset_create_new',Path(k.__file__).read_text())
        worker=(ROOT/'scripts/run_task46.py').read_text();self.assertIn("('preflight','native','gpu')",worker);self.assertIn('subprocess.run',worker)
    def test_preflight_and_gpu_source_cache_are_distinct(self):
        sys.path.insert(0,str(ROOT/'scripts'));import run_task46 as worker
        t=self.torch
        with tempfile.TemporaryDirectory(dir=ROOT/'artifacts') as tmp,patch.dict(os.environ):
            out=Path(tmp)/'result';out.mkdir();scratch=worker.preflight_output(out);seen=[]
            def verify(root,source,cache):
                cache.mkdir(exist_ok=False);seen.append(cache)
            with patch.object(a.x.repair.p.r,'verify_source',side_effect=verify),patch.object(a.n,'load_models',return_value=({'dual':self.f.old},self.f.old)):
                hashes=[]
                for destination in (scratch,out):
                    models,_,_=a.load_frozen(t,ROOT,out/'unused',ROOT/'artifacts/task45c_cost_input/checkpoint45c.zip.bin',ROOT/'artifacts/task45d_evaluation_input/evaluation45d.zip.bin',destination,'cpu');hashes.append(a.n.weights_hash(models['P']))
                with self.assertRaises(FileExistsError):a.load_frozen(t,ROOT,out/'unused',ROOT/'artifacts/task45c_cost_input/checkpoint45c.zip.bin',ROOT/'artifacts/task45d_evaluation_input/evaluation45d.zip.bin',out,'cpu')
            self.assertNotEqual(seen[0],seen[1]);self.assertEqual(hashes,[self.p['freeze']['logical_weights_sha256']]*2)
if __name__=='__main__':unittest.main()
