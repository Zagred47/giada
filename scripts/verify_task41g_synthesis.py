"""Cross-check final summaries against independently reduced raw panel audit."""
import json
from pathlib import Path
R=Path(__file__).resolve().parents[1];O=R/'experiments/results/task41g_audited_v1';D=R/'artifacts/task41g_primary_external'
def main():
    final=json.loads((O/'final_report_recovered.json').read_text());a=json.loads((D/'partial_scientific_audit.json').read_text())
    checks=0
    for name,s in final['summaries'].items():
        actual=a['summaries'][name]
        for panel in ('41','41c'):
            for mode in ('imposed','autonomous'):
                x=s['panels'][panel][mode];y=actual[panel+'/'+mode]
                assert (x['trajectory'],x['total'])==(y['trajectories']['passed'],y['trajectories']['total'])
                assert (x['responses'],x['response_total'])==(y['responses']['passed'],y['responses']['total']);checks+=2
            assert s['stress'][panel]==actual['stress'+panel];checks+=1
        for mode in ('imposed','autonomous'):
            x=s['historical'][mode];y=actual['historical/'+mode]
            assert x['trajectory']==y['trajectories']['passed'] and x['responses']==y['responses']['passed'];checks+=1
    for name,count in [('M_constant_point_broad',5),('M_anneal_sensitive_tail',3),('F_nominee',7),('E_Q0',5)]:
        assert sum(len(a['failures'][name][p]) for p in ('41c/autonomous','stress41','stress41c'))==count;checks+=1
    syn=json.loads((O/'synthesis.json').read_text());assert not syn['next_experiment']['launched'] and not syn['thresholds_changed'];checks+=2
    result=dict(valid=True,checks=checks,scope='Final report counts independently cross-checked against raw-panel reduction; not fresh simulation',thresholds_changed=False)
    (O/'synthesis_verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result))
if __name__=='__main__':main()
