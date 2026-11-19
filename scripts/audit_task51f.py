import gzip, hashlib, io, json, zipfile
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
summaries = {}
for task in ('51e', '51f'):
    folder = ROOT / f'artifacts/task{task}_kaggle_external'
    archive = folder / f'task{task}_final.zip'
    receipt = json.loads((folder / 'kaggle_receipt.json').read_text())
    with archive.open('rb') as handle:
        digest = hashlib.file_digest(handle, 'sha256').hexdigest()
    assert digest == receipt['sha256']
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        def read(name):
            if name in z.namelist(): return json.loads(z.read(name))
            return json.loads(gzip.decompress(z.read(name+'.gz')))
        report = read('final_report.json')
        evaluation = read('evaluation.json')
        keys = sorted({r['representation']+'|'+r['loss'] for r in evaluation})
        arms = {}
        for key in keys:
            rows = [r for r in evaluation if r['representation']+'|'+r['loss']==key]
            item = {}
            for group in ('reachable', 'synthetic'):
                base = [r for r in rows if r['group']==group and 'decade_bin' not in r]
                item[group] = {m:float(np.mean([r[m] for r in base])) for m in ('opening_rmse','log_rmse')}
            tr = [r for r in rows if r['group']=='reachable_trajectory_current']
            item['trajectory_passes'] = [sum(r['passed_amplitude_only']) for r in tr]
            nr = np.concatenate([r['nrmse'] for r in tr])
            nm = np.concatenate([r['nmax'] for r in tr])
            item['channel_passes_of72'] = ((nr<=.01)&(nm<=.05)).sum(0).tolist()
            arms[key] = item
        finite = {}
        for name in z.namelist():
            if name.startswith('checkpoint_') or name=='selected.pt':
                state = torch.load(io.BytesIO(z.read(name)), map_location='cpu', weights_only=True)
                finite[name] = all(bool(torch.isfinite(v).all()) for v in state.values())
        assert all(finite.values())
        result = dict(archive_sha256=digest,report=report,checkpoint_finite=finite,arms=arms,cost=read('cost.json'))
        if task=='51f':
            result['solver']=read('solver_controls.json')['results']
            exact=read('exact_log_evaluation.json')
            result['exact_log']={arm:{g:float(np.mean([r['exact_raw_log_rmse'] for r in exact if r['arm']==arm and r['group']==g])) for g in ('reachable','synthetic')} for arm in sorted({r['arm'] for r in exact})}
        summaries[task]=result
out=ROOT/'experiments/results/task51f_audit_v1'
out.mkdir(exist_ok=True)
(out/'comparison.json').write_text(json.dumps(summaries,indent=2)+'\n')
for task,s in summaries.items():
    print(task,'finite',s['checkpoint_finite'])
    print(json.dumps(s['arms'],indent=2))
    if 'solver' in s: print('SOLVER',s['solver'],'EXACT',s['exact_log'])
