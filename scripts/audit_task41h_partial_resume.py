"""Report only externally verified panels; never infer missing evaluations."""
import hashlib,io,json,sys,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from recover_task41h_evaluation import finite
D=ROOT/'artifacts/task41h_frozen_resume'
receipt=json.loads((D/'package_receipt.json').read_text());path=Path(receipt['archive'])
with path.open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==receipt['sha256']
with zipfile.ZipFile(path) as z:
    assert z.testzip() is None
    read=lambda n:json.loads(z.read(n))
    manifest=read('frozen_resume_manifest.json')
    for name,row in manifest['members'].items():assert hashlib.sha256(z.read(name)).hexdigest()==row['sha256'],name
    freeze=read('selection_freeze.json');parity=read('selection_parity.json')
    weights=z.read('frozen_checkpoint.pt');assert hashlib.sha256(weights).hexdigest()==freeze['checkpoint_sha256']
    assert finite(torch.load(io.BytesIO(weights),map_location='cpu',weights_only=True))
    reports=read('opened41/causal_results.json');stress=read('stress41.json')
    comparison={}
    for name in (freeze['nominee'],'G_nominee','G_tail_diagnostic'):
        comparison[name]={mode:dict(trajectories=reports[name+'/'+mode]['passed_cases'],total=reports[name+'/'+mode]['total'],
            responses=sum(r['metrics']['passed'] for r in reports[name+'/'+mode]['interventions']),response_total=len(reports[name+'/'+mode]['interventions'])) for mode in ('imposed','autonomous')}
        rows=[r for r in stress['rows'] if r['model']==name]
        comparison[name]['stress41']=dict(passed=sum(r['metrics']['passed'] for r in rows),total=len(rows))
    result=dict(valid=True,complete=False,training_updates_performed=0,thresholds_changed=False,
        freeze_sha256=freeze['freeze_sha256'],frozen_parameters_finite=True,selection_parity=parity,
        external_package=receipt,verified_comparison_on_opened41_only=comparison,
        missing=['opened41c complete externally verified reports','stress41c','historical regression','recurrence regression','cost','execution matrix','selected numerical diagnostics','final report'],
        backend_blocker={'primary':'T4 assignment Service Unavailable','reserve':'T4 assignment Service Unavailable'},
        task42_authorized=False)
target=ROOT/'experiments/results/task41h_recovery_v1/partial_resume_audit.json'
target.write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k not in ('external_package','selection_parity')},indent=2))
