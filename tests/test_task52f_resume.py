import copy,json,tempfile,unittest
from pathlib import Path
import torch
from src.giada_teacher import task52f_encoder_credit as q
ROOT=Path(__file__).resolve().parents[1]
class Resume(unittest.TestCase):
    def test_split_matches_continuous(self):
        torch.set_num_threads(1);p=copy.deepcopy(q.plan(ROOT));p.update(history_ms=2,updates=2,checkpoints=[1,2],batch_size=2,chunk_ticks=4,bptt_ticks=2);p['encoder_calibration']['updates']=2;p['encoder_calibration']['batch_size']=4
        for i,arm in enumerate(p['arms']):arm['bptt']=2 if i<2 else 4
        p['roles']={r:dict(seed=700+i,count=2) for i,r in enumerate(('fit','development'))};p['support_protocol']['count_per_role']=2;p['coupled_training'].update(duration_ms=2,batch_size=2,burnin_ticks=2,horizon_before300=4,horizon_after300=4)
        model,cal,_=q.load_model(torch,p,ROOT/'artifacts/task52e_input/models52e.zip.bin','cpu')
        with torch.no_grad():ordinary=q.c.b.data(torch,p,'cpu');data=q.merge_data(torch,ordinary,q.support_data(torch,p,'cpu'));aut=q.d.coupled_data(torch,p,ROOT,'cpu')
        fit=[i for i,m in enumerate(data['meta']) if m['role']=='fit'];cal=dict(cal,maxop=data['op'][:,fit].amax((0,1)).clamp_min(1e-30))
        with tempfile.TemporaryDirectory(dir=ROOT/'artifacts') as tmp:
            full=Path(tmp)/'full';split=Path(tmp)/'split';full.mkdir();split.mkdir()
            q.train(torch,p,model,data,aut,cal,full,len(ordinary['meta']))
            model,_,_=q.load_model(torch,p,ROOT/'artifacts/task52e_input/models52e.zip.bin','cpu');one=copy.deepcopy(p);one['updates']=1;one['checkpoints']=[1];q.train(torch,one,model,data,aut,cal,split,len(ordinary['meta']))
            (split/'resume_request.json').write_text(json.dumps({'checkpoint_update':1}))
            model,_,_=q.load_model(torch,p,ROOT/'artifacts/task52e_input/models52e.zip.bin','cpu');q.train(torch,p,model,data,aut,cal,split,len(ordinary['meta']))
            for filename in ('checkpoint_2.pt','selected.pt'):
                x=torch.load(full/filename,weights_only=True);y=torch.load(split/filename,weights_only=True)
                self.assertTrue(all(torch.equal(v,y[k]) for k,v in x.items()),filename)
            self.assertEqual(json.loads((full/'selection_freeze.json').read_text()),json.loads((split/'selection_freeze.json').read_text()))
if __name__=='__main__':unittest.main()
