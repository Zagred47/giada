"""Paired case-level comparison against frozen39h in the same evaluator."""
import argparse,json,hashlib,zipfile
from pathlib import Path
from collections import Counter
ROOT=Path(__file__).resolve().parents[1]
def main(folder):
    rows=json.loads((folder/'recurrence_evaluation.json').read_text());final=json.loads((folder/'final_report.json').read_text())
    baseline='joint_wfrozen39h_H8';report={'valid':True,'task':'39i','baseline':baseline,'thresholds_changed':False,'arms':{}}
    key=lambda r:(r['role'],r['family'],r['case'],r['seed'])
    reference={key(r):r for r in rows if r['model']==baseline}
    original=json.loads((ROOT/'experiments/results/task39h_colab_v1/recurrence_evaluation.json').read_text())
    historical={key(r):r for r in original if r['model']=='joint_wH8'}
    report['baseline_backend_pass_disagreements']=[list(k) for k,r in reference.items() if historical[k]['metrics']['passed']!=r['metrics']['passed']]
    for model in sorted({r['model'] for r in rows if r['model'].startswith('joint_wI')}):
        out={}
        for role in ('development','stress_diagnostic'):
            selected=[r for r in rows if r['model']==model and r['role']==role]
            gains=[r for r in selected if r['metrics']['passed'] and not reference[key(r)]['metrics']['passed']]
            losses=[r for r in selected if not r['metrics']['passed'] and reference[key(r)]['metrics']['passed']]
            failed=[r for r in selected if not r['metrics']['passed']]
            out[role]={'passed':sum(r['metrics']['passed'] for r in selected),'total':len(selected),'gains':[list(key(r)) for r in gains],'losses':[list(key(r)) for r in losses],'failures':[]}
            for r in failed:
                channels=[c for c in r['metrics']['channels'] if not c['passed']]
                out[role]['failures'].append(dict(family=r['family'],case=r['case'],seed=r['seed'],gate=r['metrics']['gate'],channels=channels))
        report['arms'][model]=out
    report['baseline']={role:sum(r['metrics']['passed'] for r in reference.values() if r['role']==role) for role in ('development','stress_diagnostic')}
    path=folder/'comparison_with_39h.json';path.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    compact={k:{role:{kk:vv for kk,vv in val.items() if kk!='failures'} for role,val in v.items()} for k,v in report['arms'].items()}
    print(json.dumps(dict(valid=True,baseline=report['baseline'],backend_pass_disagreements=len(report['baseline_backend_pass_disagreements']),arms=compact)))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('folder',type=Path);main(p.parse_args().folder)
