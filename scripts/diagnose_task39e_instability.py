"""Read preserved immutable checkpoints; no training/selection on development."""
import io
import json
from pathlib import Path
import zipfile
import torch
from src.giada_teacher import task39e_tail_support_heads as e

def main():
    root=Path(__file__).resolve().parents[1];out=root/'experiments/results/task39e_instability_v1';out.mkdir(exist_ok=True)
    source=root/'artifacts/task39e_retry1_external/recovery_final.zip';rows=[]
    with zipfile.ZipFile(source) as z:
        for step in (0,3000,10000,30000,60000):
            weights=torch.load(io.BytesIO(z.read(f'checkpoint_{step}.pt')),map_location='cpu',weights_only=True)
            for g in e.groups(torch,e.arms(),[17,29,43],'cpu'):
                for j in range(g['model'].a.shape[0]):
                    a=g['definitions'][j//3];seed=[17,29,43][j%3];stats={}
                    for k,v in weights[g['key']].items():
                        if k=='gbar':continue
                        x=v[j];finite=x[torch.isfinite(x)]
                        stats[k]=dict(nonfinite=int((~torch.isfinite(x)).sum()),maximum_finite_abs=float(finite.abs().max()) if len(finite) else None)
                    rows.append(dict(step=step,cohort=g['key'],arm=a['name'],seed=seed,parameters=stats,finite=not any(s['nonfinite'] for s in stats.values())))
        (out/'failure_report.json').write_bytes(z.read('failure_report.json'))
    (out/'checkpoint_audit.json').write_text(json.dumps(dict(rows=rows,source=str(source),thresholds_changed=False),indent=2))
    for step in (0,3000,10000,30000,60000):
        print(json.dumps(dict(step=step,failed=[(r['arm'],r['seed']) for r in rows if r['step']==step and not r['finite']],max_finite=max(s['maximum_finite_abs'] or 0 for r in rows if r['step']==step for s in r['parameters'].values()))),flush=True)
if __name__=='__main__':main()
