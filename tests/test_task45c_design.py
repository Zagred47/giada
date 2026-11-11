import copy,hashlib,io,json,os,unittest,zipfile
from pathlib import Path
import numpy as np
from src.giada_teacher import task45b_rate_coverage as b
ROOT=Path(__file__).resolve().parents[1]

class Design45c(unittest.TestCase):
    def test_audit_and_named_gate_correction(self):
        a=json.loads((ROOT/'experiments/results/task45b_audited_v1/audit.json').read_text())
        self.assertTrue(a['valid'] and a['all_checkpoints_finite'])
        self.assertEqual(len(a['checkpoints']),11)
        self.assertEqual({r['dominant_gate'] for r in a['corrected_gate_failures']},{'Ca_LVAst.h'})
        self.assertEqual(len(a['corrected_gate_failures']),6)
        for r in a['corrected_gate_failures']:self.assertGreater(r['Ca_LVAst_h_error'],r['NaTs2_t_h_error'])
        self.assertTrue(a['original45_reproduction']['passed'])
        r=json.loads((ROOT/'artifacts/task45b_kaggle_external/external_backup_receipt.json').read_text())['report']
        for panel,values in a['panels'].items():
            for row in values['summaries']:
                expected=r['panels'][panel][str(float(row['horizon_ms']))][row['arm']][row['mode']]
                self.assertEqual(row['passed'],expected['passed']);self.assertEqual(row['total'],expected['eligible'])
    def test_f011_small_freeze_reconstructs_actual_checkpoint(self):
        import torch
        torch.set_num_threads(1);os.environ['GIADA_TASK41N_FROZEN']=str(ROOT/'artifacts/task41n_frozen.zip.bin')
        p=b.spec(ROOT)[0];base=b.n.load_models(torch,ROOT,'cpu')[0]['dual']
        with np.load(ROOT/p['freeze']['heads_path']) as z:frozen=b.x.repair.materialize(torch,base,dict(z))
        folder=ROOT/'experiments/results/task45b_audited_v1';f=json.loads((folder/'f011_freeze.json').read_text());path=ROOT/f['heads_path']
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),f['heads_sha256'])
        with np.load(path) as z:w=torch.tensor(z['weights'])
        net=b.materialize(torch,frozen,w,base,(0,1,1))
        self.assertEqual(b.n.weights_hash(net),f['logical_weights_sha256'])
        self.assertEqual(b.n.weights_hash(frozen),p['freeze']['logical_weights_sha256'])
    def test_design_is_explicitly_not_a_run(self):
        d=json.loads((ROOT/'experiments/task45c_design_v1.json').read_text());old=b.spec(ROOT)[0]
        self.assertEqual(d['quality'],old['quality']);self.assertEqual(d['compartment_gates'],old['compartment_gates'])
        self.assertEqual(d['status'],'DESIGN_PREPARED_NOT_EXECUTABLE')
        self.assertFalse(d['remote_execution_authorized'] or d['training_performed'] or d['independent47_accessed'])
        self.assertEqual(len(d['arms']),10)
        self.assertEqual(d['corrected_target']['primary'],'Ca_LVAst.h')
        self.assertEqual(d['corrected_target']['negative_control'],'NaTs2_t.h')
        self.assertEqual({r['id'] for r in d['arms']},{'C0','N00','N01','N10','N11','I0','I1','L0','L1','J'})
    def test_initial_probes_use_correct_gate_and_intermediate_node(self):
        p=json.loads((ROOT/'experiments/results/task45b_audited_v1/initial45c_probes.json').read_text())
        self.assertFalse(p['training_performed'] or p['thresholds_changed'])
        self.assertEqual({r['index'] for r in p['rates_at_minus130'] if r['channel']=='Ca_LVAst'},{3})
        self.assertEqual({r['index'] for r in p['rates_at_minus130'] if r['channel']=='NaTs2_t'},{7})
        t=p['remaining_Ih_crossing'];self.assertEqual(t['node'],1);self.assertEqual(t['arm'],'f011')
        self.assertAlmostEqual(t['observed_signed_shift_ms'],-.9)
        self.assertAlmostEqual(t['linear_predicted_shift_ms'],-.934318555,places=6)
        self.assertEqual(t['unchanged_threshold_ms'],.3)

if __name__=='__main__':unittest.main()
