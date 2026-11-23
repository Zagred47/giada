import copy,json,tempfile,unittest
from pathlib import Path
import torch
from src.giada_teacher import task52g_gradient_preservation as g
ROOT=Path(__file__).resolve().parents[1]

def config():
    p=copy.deepcopy(g.plan(ROOT));p.update(history_ms=2,updates=2,checkpoints=[1,2],batch_size=2,chunk_ticks=4,bptt_ticks=2)
    p['preservation']['block_ticks']=2
    for a in p['arms']:a['bptt']=4
    p['roles']={r:dict(seed=700+i,count=2) for i,r in enumerate(('fit','development'))};p['support_protocol']['count_per_role']=2;p['coupled_training'].update(duration_ms=2,batch_size=2,burnin_ticks=2,horizon_before300=4,horizon_after300=4);p['cost'].update(batch=2,ticks=2,repetitions=1)
    return p

class Preservation(unittest.TestCase):
    def test_projection(self):
        c=[torch.tensor([[1.,0.],[1.,0.],[0.,0.]])];v=[torch.tensor([[-2.,1.],[-2.,1.],[1.,1.]])]
        merged,d=g.merge_gradients(torch,c,v,torch.tensor([True,False,True]))
        self.assertTrue(torch.equal(merged[0],torch.tensor([[1.,1.],[-1.,1.],[1.,1.]])))
        self.assertGreaterEqual(d['after'][0].item(),0);self.assertTrue(torch.isfinite(merged[0]).all())
    def test_control_matches_original_teacher(self):
        torch.set_num_threads(1);p=config()
        for a in p['arms']:a['guard']=False
        net,cal,_=g.f.load_model(torch,p,ROOT/'artifacts/task52e_input/models52e.zip.bin','cpu');other=copy.deepcopy(net)
        data=g.f.c.b.data(torch,p,'cpu');cal=dict(cal,maxop=data['op'].amax((0,1)).clamp_min(1e-30));ids=torch.zeros(12,2,dtype=torch.long);starts=torch.zeros_like(ids)
        g.f.teacher(torch,p,net,data,ids,starts,cal,torch.ones(12),True)
        ref=g.reference_errors(torch,p,other,data,ids,starts);g.teacher(torch,p,other,data,ids,starts,cal,ref)
        for (name,a),(_,b) in zip(net.named_parameters(),other.named_parameters()):
            if a.grad is not None:self.assertTrue(torch.allclose(a.grad,b.grad,atol=1e-6,rtol=1e-5),name)
    def test_pipeline(self):
        torch.set_num_threads(1);p=config();cp=copy.deepcopy(g.f.parent.plan(ROOT));cp.update(duration_ms=2,horizons_ms=[1,2],protocols=['burst'])
        with tempfile.TemporaryDirectory(dir=ROOT/'artifacts') as tmp:
            out=Path(tmp)/'out';out.mkdir();g.run(torch,ROOT,ROOT/'artifacts/task41n_primary_external/recovery_final.zip',ROOT/'artifacts/task45c_cost_input/checkpoint45c.zip.bin',ROOT/'artifacts/task45d_evaluation_input/evaluation45d.zip.bin',ROOT/'artifacts/task52e_input/models52e.zip.bin',out,'cpu',p,cp)
            report=json.loads((out/'final_report.json').read_text());self.assertEqual(report['task'],'52g');self.assertTrue(report['complete']);self.assertEqual(report['encoder_calibration_updates'],0)
            rows=json.loads((out/'learning_curve.json').read_text());self.assertEqual(len(rows),2);self.assertEqual(len(rows[-1]['conflict_fraction']),12)
    def test_unmodified_one_step_matches_long_credit(self):
        torch.set_num_threads(1);p=config();p['updates']=1;p['checkpoints']=[1]
        for arm in p['arms']:arm['guard']=False;arm['project']=False
        net,cal,_=g.f.load_model(torch,p,ROOT/'artifacts/task52e_input/models52e.zip.bin','cpu');other=copy.deepcopy(net)
        ordinary=g.f.c.b.data(torch,p,'cpu');data=g.f.merge_data(torch,ordinary,g.f.support_data(torch,p,'cpu'));aut=g.f.d.coupled_data(torch,p,ROOT,'cpu');cal=dict(cal,maxop=data['op'].amax((0,1)).clamp_min(1e-30))
        with tempfile.TemporaryDirectory(dir=ROOT/'artifacts') as tmp:
            left=Path(tmp)/'left';right=Path(tmp)/'right';left.mkdir();right.mkdir()
            g.f.train(torch,p,net,data,aut,cal,left,len(ordinary['meta']));g.train(torch,p,other,data,aut,cal,right,len(ordinary['meta']))
            a=torch.load(left/'checkpoint_1.pt',weights_only=True);b=torch.load(right/'checkpoint_1.pt',weights_only=True)
            for name in a:self.assertTrue(torch.allclose(a[name],b[name],atol=1e-6,rtol=1e-5),name)
    def test_contract(self):
        p=g.plan(ROOT);self.assertEqual(len(p['arms']),4);self.assertFalse(p['thresholds_changed']);self.assertEqual(p['user_cap'].split(';')[0],'Last allowed subtask52j')
        old=g.f.plan(ROOT);self.assertEqual(p['selection'],old['selection']);self.assertEqual(p['roles'],old['roles']);self.assertEqual(p['coupled_training'],old['coupled_training'])
    def test_resume_preserves_updates_and_original_anchor(self):
        torch.set_num_threads(1);p=config()
        initial,cal,_=g.f.load_model(torch,p,ROOT/'artifacts/task52e_input/models52e.zip.bin','cpu')
        data0=g.f.c.b.data(torch,p,'cpu');data=g.f.merge_data(torch,data0,g.f.support_data(torch,p,'cpu'));aut=g.f.d.coupled_data(torch,p,ROOT,'cpu');cal=dict(cal,maxop=data['op'].amax((0,1)).clamp_min(1e-30))
        with tempfile.TemporaryDirectory(dir=ROOT/'artifacts') as tmp:
            full=Path(tmp)/'full';split=Path(tmp)/'split';full.mkdir();split.mkdir()
            g.train(torch,p,copy.deepcopy(initial),data,aut,cal,full,len(data0['meta']))
            short=copy.deepcopy(p);short['updates']=1;short['checkpoints']=[1]
            g.train(torch,short,copy.deepcopy(initial),data,aut,cal,split,len(data0['meta']))
            # Exercise the legacy checkpoint lacking diagnostic accumulators.
            state=torch.load(split/'resumable_training.pt',weights_only=True);state.pop('diagnostic_sums');torch.save(state,split/'resumable_training.pt')
            (split/'resume_request.json').write_text(json.dumps(dict(checkpoint_update=1)))
            g.train(torch,p,copy.deepcopy(initial),data,aut,cal,split,len(data0['meta']))
            left=torch.load(full/'checkpoint_2.pt',weights_only=True);right=torch.load(split/'checkpoint_2.pt',weights_only=True)
            for key in left:self.assertTrue(torch.equal(left[key],right[key]),key)
            self.assertEqual(json.loads((full/'selection_freeze.json').read_text()),json.loads((split/'selection_freeze.json').read_text()))
if __name__=='__main__':unittest.main()
