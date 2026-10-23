"""Inspect saved case-level failures while the remaining frozen audit runs."""
import hashlib,json,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
D=ROOT/'artifacts/task41h_frozen_evaluation_primary_external'
receipt=json.loads((D/'external_backup_receipt.json').read_text());path=Path(receipt['local'])
with path.open('rb') as stream:assert hashlib.file_digest(stream,'sha256').hexdigest()==receipt['remote']['sha256']
def failed(rows):
    result=[]
    for row in rows:
        metrics=row['metrics']
        if metrics['passed']:continue
        result.append({k:v for k,v in row.items() if k not in ('metrics','model')}|dict(channels=[r for r in metrics['channels'] if not r['passed']],gate=metrics.get('gate')))
    return result
with zipfile.ZipFile(path) as z:
    n=json.loads(z.read('selection_freeze.json'))['nominee']
    panel=json.loads(z.read('opened41c/causal_results.json'));stress=json.loads(z.read('stress41c.json'))
    result=dict(complete=False,verified_source_sha256=receipt['remote']['sha256'],nominee=n,models={})
    for name in (n,'G_nominee','G_tail_diagnostic'):
        result['models'][name]=dict(responses=failed(panel[name+'/autonomous']['interventions']),stress=failed([r for r in stress['rows'] if r['model']==name]))
target=ROOT/'experiments/results/task41h_recovery_v1/saved_failures_partial.json'
target.write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2),flush=True)
