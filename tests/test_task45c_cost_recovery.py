"""Recovery wiring and real-checkpoint CPU dry run; not GPU timing evidence."""
import importlib.util,json,sys,tempfile,unittest,copy
from pathlib import Path
from unittest.mock import patch
import numpy as np
from tests.test_task45c import PreservingKinetics,ROOT

class Recovery(unittest.TestCase):
    def test_real_checkpoints_no_fit_or_selection(self):
        PreservingKinetics.setUpClass();fixture=PreservingKinetics;torch=fixture.torch
        spec=importlib.util.spec_from_file_location('cost_recovery',ROOT/'scripts/run_task45c_cost_recovery.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
        original_load=torch.load;prior,cfg,ca,base=m.a.n.contract(ROOT);prior=copy.deepcopy(prior);prior['torch']=torch.__version__;prior['gpu']='CPU mocked GPU identity'
        with tempfile.TemporaryDirectory(dir=ROOT/'artifacts') as tmp:
            out=Path(tmp)/'result';cache=out.parent/'verified_source45c';cache.mkdir();np.savez_compressed(cache/'native_shadow_traces.npz',voltage=np.zeros((2,2)))
            def bench(*args):
                self.assertTrue(torch.is_inference_mode_enabled());self.assertEqual(len(args[1]),12)
                return [dict(all_outputs_finite=True) for _ in range(6)]
            def load(*args,**kw):kw['map_location']='cpu';return original_load(*args,**kw)
            with patch.object(m.a.n,'contract',return_value=(prior,cfg,ca,base)),patch.object(m.a.n,'load_models',return_value=({'dual':fixture.old},fixture.old)),patch.object(m.a.x.repair.p.r,'verify_source'),patch.object(m.a.n,'benchmark',side_effect=bench),patch.object(m.a,'fit',side_effect=AssertionError('No fitting allowed')),patch.object(torch.cuda,'is_available',return_value=True),patch.object(torch.cuda,'get_device_name',return_value=prior['gpu']),patch.object(torch,'load',side_effect=load),patch.object(sys,'argv',['worker','unused',str(ROOT/'artifacts/task45c_cost_input/checkpoint45c.zip.bin'),str(out)]):
                m.main()
            r=json.loads((out/'final_report.json').read_text());self.assertTrue(r['complete'] and r['weights_unchanged']);self.assertFalse(r['training_performed'] or r['selection_recomputed'])
            self.assertEqual(r['panels']['original45']['1000.0']['J']['autonomous']['passed'],68)

    def test_console_notebook(self):
        sys.path.insert(0,str(ROOT/'scripts'));import task45c_cost_transport_v2 as k
        for c in k.notebook('a'*40,'b'*64)['cells']:
            if c['cell_type']=='code':
                s=''.join(c['source']);compile(s,'recovery','exec');self.assertIn('checkpoint45c.zip.bin',s);self.assertNotIn('task45c_final.zip',s)
if __name__=='__main__':unittest.main()
