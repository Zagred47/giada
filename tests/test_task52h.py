import copy,json,tempfile,unittest
from pathlib import Path
import torch
from src.giada_teacher import task52h_ih_preservation as h
ROOT=Path(__file__).resolve().parents[1]
class IhPreservation(unittest.TestCase):
    def test_contract(self):
        p=h.plan(ROOT);self.assertEqual([(a['targeted_Ih'],a['worst_guard']) for a in p['arms']],[(False,False),(True,False),(False,True),(True,True)])
        old=h.g.plan(ROOT)
        for key in ('source_indices','seeds','roles','checkpoints','learning_rate','updates','selection','families'):self.assertEqual(p[key],old[key])
        self.assertTrue(all(a['guard'] and not a['project'] for a in p['arms']))
    def test_fit_probe_restores_weights_even_if_blocked(self):
        torch.set_num_threads(1);p=h.plan(ROOT);p['history_ms']=8;p['support_protocol']['count_per_role']=8;p['feasibility_probe']['chunk_ticks']=16
        net,cal,_=h.f.load_model(torch,p,ROOT/'artifacts/task52e_input/models52e.zip.bin','cpu');data=h.f.support_data(torch,p,'cpu')
        before={k:v.clone() for k,v in net.state_dict().items()}
        with tempfile.TemporaryDirectory(dir=ROOT/'artifacts') as temp:
            h.probe(torch,p,net,data,cal,Path(temp));r=json.loads((Path(temp)/'Ih_feasibility_preflight.json').read_text());self.assertTrue(r['fit_only'] and r['weights_restored']);self.assertEqual(len(r['trials']),4)
        self.assertTrue(all(torch.equal(v,before[k]) for k,v in net.state_dict().items()))
if __name__=='__main__':unittest.main()
