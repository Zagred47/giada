"""Resume evaluations from an already verified freeze; no ranking or fitting."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse,hashlib,json,sys,traceback,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from src.giada_teacher import task41h_path_repair as h
from src.giada_teacher import task41h_evaluation as ev
from recover_task41h_evaluation import finite,install_backup_barriers

def verify_freeze(folder):
    manifest=json.loads((folder/'frozen_resume_manifest.json').read_text())
    for name,row in manifest['members'].items():
        assert h.p.digest(folder/name)==row['sha256'],name
    freeze=json.loads((folder/'selection_freeze.json').read_text())
    payload={k:v for k,v in freeze.items() if k!='freeze_sha256'}
    assert hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()==freeze['freeze_sha256']==manifest['freeze_sha256']
    assert h.p.digest(folder/'frozen_checkpoint.pt')==freeze['checkpoint_sha256']
    assert not freeze['opened_regressions_used_for_selection']
    assert json.loads((folder/'selection_parity.json').read_text())['all60_rank_vectors_bitwise_equal']
    return manifest,freeze

def install_completed_panel_cache(output):
    """Only complete panels are reused; incomplete panels use original code."""
    original_suite=ev.ee.old.suite;original_stress=ev.ee.old.stress
    pending_labels=[];reused=[]
    def suite(torch,folder,models,*args):
        label=folder.name.removeprefix('opened');pending_labels.append(label)
        path=folder/'causal_results.json'
        if path.exists():
            cached=json.loads(path.read_text())
            wanted={name+'/'+mode for name in models for mode in ('imposed','autonomous')}
            if set(cached)==wanted:
                for mode in ('imposed','autonomous'):
                    floor=json.loads((folder/f'floor_{mode}.json').read_text())
                    assert all(r['passed'] for r in floor['cases']+floor['interventions'])
                assert all(r['passed'] for r in json.loads((folder/'native_phase_floor.json').read_text()))
                reused.append(dict(panel=folder.name,rows=len(cached)))
                print(f'[GIADA Task41h] reusing verified {folder.name}: {len(cached)} model/mode reports',flush=True)
                return cached
        return original_suite(torch,folder,models,*args)
    def stress(torch,models,*args):
        if not pending_labels:return original_stress(torch,models,*args)
        label=pending_labels.pop(0);path=output/f'stress{label}.json'
        if path.exists():
            cached=json.loads(path.read_text())
            assert {r['model'] for r in cached['rows']}==set(models)
            assert all(r['passed'] for r in cached['formula_floor'])
            reused.append(dict(panel='stress'+label,rows=len(cached['rows'])))
            print(f'[GIADA Task41h] reusing verified stress{label}',flush=True)
            return cached
        return original_stress(torch,models,*args)
    ev.ee.old.suite=suite;ev.ee.old.stress=stress
    return reused

def run(archive,output,expected_sha):
    assert h.p.digest(archive)==expected_sha
    assert not output.exists() or not any(output.iterdir())
    output.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for n in z.namelist():assert (output/n).resolve().is_relative_to(output.resolve())
        z.extractall(output)
    manifest,freeze=verify_freeze(output)
    spec,cfg,calcium,base=h.contract(ROOT)
    assert json.loads((output/'preregistration.json').read_text())==spec
    seal=manifest['original_seal']
    assert torch.cuda.is_available() and torch.__version__==seal['torch_version']
    assert torch.cuda.get_device_name()==seal['GPU']
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    def forbidden(*args,**kwargs):raise RuntimeError('Training is forbidden in frozen evaluation resume')
    h.run=forbidden;h.g.Adam.step=forbidden;torch.optim.Optimizer.step=forbidden
    controls=h.frozen_controls(torch,ROOT,spec,'cuda');trained={}
    states=torch.load(output/'frozen_checkpoint.pt',map_location='cpu',weights_only=True)
    assert finite(states)
    for name,state in states.items():
        definition=next(a for a in h.arms() if a['name']==name.removeprefix('aggregate_'))
        net=h.make_model(torch,controls['E_Q0'],[definition],'cuda')
        net.load_state_dict(state);net.eval()
        for parameter in net.parameters():parameter.requires_grad_(False)
        trained[name]=net
    assert freeze['nominee'] in trained
    reused=install_completed_panel_cache(output)
    install_backup_barriers(output)
    # No training checkpoints are loaded: only the immutable frozen model.
    with torch.inference_mode():report=ev.evaluate(torch,ROOT,output,controls,trained,spec,cfg,calcium,base,'cuda')
    assert h.p.digest(output/'frozen_checkpoint.pt')==freeze['checkpoint_sha256']
    report.update(evaluation_recovered=True,training_updates_performed=0,thresholds_changed=False,
        selection_recomputed=False,cached_panels_reused=reused,original_source_archive_sha256=manifest['source_archive_sha256'])
    h.p.save_json(output/'final_report.json',report)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('output',type=Path);ap.add_argument('--archive',type=Path,required=True);ap.add_argument('--sha256',required=True);a=ap.parse_args()
    try:run(a.archive,a.output,a.sha256)
    except BaseException as exc:
        a.output.mkdir(parents=True,exist_ok=True)
        (a.output/'resume_failure_report.json').write_text(json.dumps(dict(valid=False,technical_failure=True,error=repr(exc),training_updates_performed=0,traceback=traceback.format_exc()),indent=2))
        raise
