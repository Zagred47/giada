"""Audit only byte-verified locally available 41g reports; no final GO claim."""
import json,zipfile
from pathlib import Path
from audit_task41d_result import failed
ROOT=Path(__file__).resolve().parents[1]
D=ROOT/'artifacts/task41g_primary_external'
def count(rows):return {'passed':sum(r['metrics']['passed'] for r in rows),'total':len(rows)}
def ids(rows):return {(r.get('case'),r.get('seed'),r.get('family'),r.get('input_seed')) for r in rows}
def delta(a,b):
    x,y=ids(a),ids(b)
    return {'resolved':sorted(x-y,key=str),'introduced':sorted(y-x,key=str),'remaining':sorted(x&y,key=str)}
receipt=json.loads((D/'external_backup_receipt.json').read_text())
with zipfile.ZipFile(receipt['local']) as z:
    read=lambda n:json.loads(z.read(n))
    out={'scope':'Verified intermediate backup; no final report or speed result available','summaries':{},'failures':{},'deltas_vs_41f_nominee':{},'missing':[]}
    for panel in ('41','41c'):
        c=read('opened'+panel+'/causal_results.json')
        for key,v in c.items():
            model,mode=key.split('/')
            summary=out['summaries'].setdefault(model,{})
            summary[panel+'/'+mode]={'trajectories':{'passed':v['passed_cases'],'total':v['total']},'responses':count(v['interventions']),'voltage_rmse_mv':v['pooled_voltage_rmse_mv']}
            out['failures'].setdefault(model,{})[panel+'/'+mode]=failed(v['interventions'])
        del c
        stress=read('stress'+panel+'.json')
        for model in out['summaries']:
            rows=[r for r in stress['rows'] if r['model']==model]
            out['summaries'][model]['stress'+panel]=count(rows)
            out['failures'][model]['stress'+panel]=failed(rows)
    hist=read('historical_regression/causal_results.json')
    for key,v in hist.items():
        model,mode=key.split('/')
        out['summaries'].setdefault(model,{})['historical/'+mode]={'trajectories':{'passed':v['passed_cases'],'total':v['total']},'responses':count(v['interventions'])}
        out['failures'].setdefault(model,{})['historical/'+mode]=failed(v['interventions'])
    for model in out['summaries']:
        out['deltas_vs_41f_nominee'][model]={panel:delta(out['failures']['F_nominee'][panel],rows) for panel,rows in out['failures'][model].items() if panel in out['failures']['F_nominee']}
    if 'historical_regression/recurrence_evaluation.json' in z.namelist():
        rr=read('historical_regression/recurrence_evaluation.json')
        for model,s in out['summaries'].items():
            for role in ('development','stress_diagnostic'):
                rows=[r for r in rr if r['model']=='joint_w'+model and r['role']==role]
                if rows:s['recurrence/'+role]=count(rows)
    out['backup_source']=receipt
    for n in ['final_report.json','cost.json','execution_matrix.json','precision_execution_cost.json','recurrence_evaluation.json','paired_contrasts.json']:
        if n not in z.namelist():out['missing'].append(n)
    out['freeze']=read('selection_freeze.json')
    for name in ['precision_factorial','per_head_rollback']:
        data=read('frozen_audit/'+name+'.json')
        (D/('partial_'+name+'.json')).write_text(json.dumps(data))
    (D/'partial_scientific_audit.json').write_text(json.dumps(out,indent=2))
    for model,s in out['summaries'].items():
        print(model,json.dumps({k:v.get('responses',v) for k,v in s.items() if 'autonomous' in k or 'stress' in k}))
    print('missing',out['missing'])
