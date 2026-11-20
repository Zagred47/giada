import copy,json,sys,tempfile,unittest
from pathlib import Path
from src.giada_teacher import task52b_decoder_memory as q
ROOT=Path(__file__).resolve().parents[1]
class Matrix(unittest.TestCase):
    def test_paired(self):
        import torch
        p=q.plan(ROOT);torch.set_num_threads(1);mean=torch.linspace(.001,.5,11);net=q.model_class(torch)(p,mean,'cpu')
        g=torch.full((12,2,18),.5);op=net.opening(net.initialize(g))
        self.assertLess(float((op-mean).abs().max()),1e-6)
        self.assertEqual(net.lm.sum(-1).tolist(),[[10.]]*12)
        self.assertTrue(torch.equal(net.w[:3],net.w[3:6]))
        self.assertFalse(torch.equal(net.c[:3],net.c[6:9]))
    def test_pipeline(self):
        import torch
        torch.set_num_threads(1);p=copy.deepcopy(q.plan(ROOT));p.update(history_ms=2.,updates=2,checkpoints=[1,2],batch_size=2,chunk_ticks=4,bptt_ticks=2,diagnostic_ticks=10)
        p['roles']={r:dict(seed=400+i,count=2) for i,r in enumerate(('fit','development','fresh'))}
        p['oracle_initialization'].update(iterations=2,window_ticks=2,cases=2)
        cp=copy.deepcopy(q.parent.plan(ROOT));cp.update(duration_ms=2.,horizons_ms=[1,2],protocols=['burst']);p['cost'].update(batch=2,ticks=2,repetitions=1)
        with tempfile.TemporaryDirectory(dir=ROOT/'artifacts') as tmp:
            out=Path(tmp)/'out';out.mkdir();q.run(torch,ROOT,ROOT/'artifacts/task41n_primary_external/recovery_final.zip',ROOT/'artifacts/task45c_cost_input/checkpoint45c.zip.bin',ROOT/'artifacts/task45d_evaluation_input/evaluation45d.zip.bin',ROOT/'artifacts/task52_input/models52.zip.bin',out,'cpu',p,cp)
            report=json.loads((out/'final_report.json').read_text());self.assertTrue(report['valid']);self.assertEqual(len(report['summary']),5);self.assertTrue(report['postfreeze_weights_unchanged'])
            self.assertTrue(json.loads((out/'initialization_diagnostics.json').read_text())['weights_unchanged'])
    def test_notebook(self):
        sys.path.insert(0,str(ROOT/'scripts'));import task52b_kaggle as k
        doc=k.notebook('a'*40);text=json.dumps(doc);self.assertIn('run_task52b.py',text);self.assertIn('models52.zip.bin',text)
        self.assertIn("worker_report.get('training_updates')",text)
        for c in doc['cells']:
            if c['cell_type']=='code':compile(''.join(c['source']),'52b','exec')
    def test_contract(self):
        p=q.plan(ROOT);self.assertEqual(len(p['arms']),4);self.assertFalse(p['thresholds_changed']);self.assertFalse(p['abstraction_changed'])
        self.assertEqual({tuple(x['widths']) for x in p['arms']},{(2,3,3,1,1)})
if __name__=='__main__':unittest.main()
