"""Read-only detailed failure and causal contrast audit after final recovery."""
import json
from pathlib import Path
import zipfile
from audit_task41d_result import failed

ROOT=Path(__file__).resolve().parents[1]
folder=ROOT/'artifacts/task41e_reserve_external'
receipt=json.loads((folder/'external_backup_receipt.json').read_text())
base=folder/Path(receipt['local'].replace('\\','/')).name
with zipfile.ZipFile(base) as b,zipfile.ZipFile(folder/'recovery_delta.zip') as d:
    names=set(d.namelist())
    load=lambda n:json.loads((d if n in names else b).read(n))
    report=load('final_report.json');dev=load('development_gpu.json')
    details={n:{'development':failed(dev[n]),'stress':{},'responses':{}} for n in report['summaries']}
    for panel in ('41','41c'):
        stress=load('stress'+panel+'.json')
        causal=load('opened'+panel+'/causal_results.json')
        for name in details:
            details[name]['stress'][panel]=failed(stress['rows'],name)
            details[name]['responses'][panel]=failed(causal[name+'/autonomous']['interventions'])
        del causal
    regressions={}
    for name in details:
        regressions[name]={}
        for kind in ('stress','responses'):
            for panel in ('41','41c'):
                identity=lambda row:(row.get('case'),row.get('seed'),row.get('family'))
                parent={identity(r) for r in details['P110'][kind][panel]}
                now={identity(r) for r in details[name][kind][panel]}
                regressions[name][kind+'/'+panel]={'resolved':list(parent-now),'introduced':list(now-parent),
                                                 'remaining':list(now&parent)}
    probes={n:load(n) for n in set(b.namelist())|names if n.startswith('coupling_probe/') and n.endswith('.json')}
    out=dict(decision=report['decision'],nominee=report['nominee'],failure_details=details,
             regressions_vs_P110=regressions,contrasts=load('paired_contrasts.json'),
             selection=load('selection_freeze.json'),coupling_probes=probes)
path=folder/'final_detailed_audit.json'
path.write_text(json.dumps(out,indent=2))
print(json.dumps({'audit':str(path),'models':len(details),'coupling_probes':list(probes),
                  'contrasts':out['contrasts']}),flush=True)
