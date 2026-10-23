"""Finish only missing stages from the complete frozen-quality checkpoint."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse,hashlib,json,sys,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from src.giada_teacher import task41h_path_repair as h
from src.giada_teacher import task41h_evaluation as ev
from resume_task41h_frozen_evaluation import verify_freeze
from recover_task41h_evaluation import finite
from task41h_stage_cache import install

def run(archive,output,digest):
    assert h.p.digest(archive)==digest and not output.exists()
    output.mkdir(parents=True)
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for n in z.namelist():assert (output/n).resolve().is_relative_to(output.resolve())
        z.extractall(output)
    manifest,freeze=verify_freeze(output);seal=manifest['original_seal']
    spec,cfg,calcium,base=h.contract(ROOT)
    assert json.loads((output/'preregistration.json').read_text())==spec
    assert torch.cuda.is_available() and torch.cuda.get_device_name()==seal['GPU'] and torch.__version__==seal['torch_version']
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    def forbidden(*a,**k):raise RuntimeError('Training/selection is forbidden during remaining-stage recovery')
    h.run=forbidden;h.g.Adam.step=forbidden;torch.optim.Optimizer.step=forbidden
    controls=h.frozen_controls(torch,ROOT,spec,'cuda');states=torch.load(output/'frozen_checkpoint.pt',map_location='cpu',weights_only=True)
    assert finite(states);trained={}
    for name,state in states.items():
        definition=next(a for a in h.arms() if a['name']==name.removeprefix('aggregate_'))
        model=h.make_model(torch,controls['E_Q0'],[definition],'cuda');model.load_state_dict(state)
        for par in model.parameters():par.requires_grad_(False)
        trained[name]=model.eval()
    # Validate completeness before allowing cached arithmetic to be skipped.
    models={**controls,**trained};dev=json.loads((output/'development_gpu.json').read_text())
    assert set(dev)==set(models) and all(len(rows)==36 for rows in dev.values())
    for label in ('41','41c'):
        rr=json.loads((output/f'opened{label}'/'causal_results.json').read_text())
        assert len(rr)==48 and all(r['total']==528 and len(r['interventions'])==336 for r in rr.values())
    reused=install(h,ev,output)
    old_cost=json.loads((output/'cost.json').read_text()) if (output/'cost.json').exists() else []
    with torch.inference_mode():report=ev.evaluate(torch,ROOT,output,controls,trained,spec,cfg,calcium,base,'cuda')
    from task41h_restore_traces import restore
    with torch.inference_mode():trace_report=restore(torch,h,ev,ROOT,output,controls,trained,spec,cfg,calcium,base,'cuda')
    assert h.p.digest(output/'frozen_checkpoint.pt')==freeze['checkpoint_sha256']
    report.update(evaluation_recovered=True,training_updates_performed=0,selection_recomputed=False,thresholds_changed=False,cached_panels_reused=reused,
        cached_cost_rows=[dict(batch=r['batch_per_seed'],scope=r['scope']) for r in old_cost],
        recovered_source_sha256=digest,trace_restoration=trace_report,evaluation_backend=dict(torch=torch.__version__,gpu=torch.cuda.get_device_name(),platform='Kaggle batch'))
    h.p.save_json(output/'final_report.json',report)
    print('[GIADA Task41h] completed '+report['decision'],flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('output',type=Path);ap.add_argument('--archive',type=Path,required=True);ap.add_argument('--sha256',required=True);a=ap.parse_args()
    try:run(a.archive,a.output,a.sha256)
    except BaseException as exc:
        import traceback
        a.output.mkdir(parents=True,exist_ok=True)
        (a.output/'failure_report.json').write_text(json.dumps(dict(valid=False,technical_failure=True,error=repr(exc),traceback=traceback.format_exc(),training_updates_performed=0,selection_recomputed=False,thresholds_changed=False),indent=2))
        raise
