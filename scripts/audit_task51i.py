import gzip,hashlib,io,json,zipfile
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
summary={}
for task in ('51h','51i'):
    folder=ROOT/f'artifacts/task{task}_kaggle_external'
    archive=folder/f'task{task}_final.zip'
    receipt=json.loads((folder/'kaggle_receipt.json').read_text())
    with archive.open('rb') as handle:digest=hashlib.file_digest(handle,'sha256').hexdigest()
    assert digest==receipt['sha256'] and archive.stat().st_size==receipt['bytes']
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        def read(name):return json.loads(z.read(name)) if name in z.namelist() else json.loads(gzip.decompress(z.read(name+'.gz')))
        ev=read('evaluation.json');report=read('final_report.json');freeze=read('selection_freeze.json')
        arms={}
        for arm in sorted({r['arm'] for r in ev}):
            rows=[r for r in ev if r['arm']==arm];item={}
            for group in ('reachable','synthetic'):
                rr=[r for r in rows if r['group']==group and 'decade_bin' not in r]
                item[group]={metric:float(np.mean([r[metric] for r in rr])) for metric in ('opening_rmse','log_rmse')}
            rr=[r for r in rows if r['group']=='reachable_trajectory_current']
            item['trajectory_passes']=[sum(r['passed_amplitude_only']) for r in rr]
            nr=np.concatenate([r['nrmse'] for r in rr]);nm=np.concatenate([r['nmax'] for r in rr])
            item['channel_passes_of72']=((nr<=.01)&(nm<=.05)).sum(0).tolist()
            arms[arm]=item
        finite={}
        for name in z.namelist():
            if name.startswith('checkpoint_') or name=='selected.pt':
                state=torch.load(io.BytesIO(z.read(name)),map_location='cpu',weights_only=True)
                finite[name]=all(bool(torch.isfinite(v).all()) for v in state.values())
        assert all(finite.values())
        result=dict(archive_sha256=digest,checkpoint_finite=finite,arms=arms,freeze=freeze,report=report)
        if task=='51i':
            result['curve']=read('learning_curve.json')
            final=read('final_evaluation.json')
            result['final_arms']={}
            for arm in arms:
                rr=[r for r in final if r['arm']==arm]
                agg=[r for r in rr if r['group']=='reachable' and 'decade_bin' not in r]
                tr=[r for r in rr if r['group']=='reachable_trajectory_current']
                result['final_arms'][arm]=dict(opening_rmse=float(np.mean([r['opening_rmse'] for r in agg])),log_rmse=float(np.mean([r['log_rmse'] for r in agg])),trajectory_passes=[sum(r['passed_amplitude_only']) for r in tr])
            exact=read('exact_checkpoint_evaluation.json')
            result['exact_checkpoint']={arm:{label:{group:float(np.mean([r['exact_raw_log_rmse'] for r in exact if r['arm']==arm and r['checkpoint']==label and r['group']==group])) for group in ('reachable','synthetic')} for label in ('initial','final','selected')} for arm in arms}
        summary[task]=result
out=ROOT/'experiments/results/task51i_audit_v1';out.mkdir(exist_ok=True)
(out/'comparison.json').write_text(json.dumps(summary,indent=2)+'\n')
for task,r in summary.items():
    print(task,'FINITE',r['checkpoint_finite']);print(json.dumps(r['arms'],indent=2))
    if 'exact_checkpoint' in r:print('EXACT',json.dumps(r['exact_checkpoint'],indent=2))
