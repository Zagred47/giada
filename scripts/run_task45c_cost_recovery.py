"""Recover ONLY cost evaluation from immutable45c checkpoints, no fit/selection."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import hashlib,json,sys,zipfile,copy,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.giada_teacher import task45c_preserving_kinetics as a
SOURCE_SHA='04a117d26354747cd54092158d509ea1388f6dcd2d8e0fa4dce11e76cd6b8986'

def main():
    import torch,numpy as np
    source,archive,out=map(Path,sys.argv[1:4]);out.mkdir(exist_ok=False)
    try:
        torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.use_deterministic_algorithms(True)
        p,cfg,ca,base=a.spec(ROOT);prior=a.n.contract(ROOT)[0]
        assert torch.cuda.is_available() and torch.__version__==prior['torch'] and torch.cuda.get_device_name()==prior['gpu']
        assert a.sha(archive)==SOURCE_SHA
        cache=out.parent/'verified_source45c';a.x.repair.p.r.verify_source(ROOT,source,cache)
        os.environ['GIADA_TASK41N_FROZEN']=str(cache/'frozen_source.zip.bin')
        nets,parent=a.n.load_models(torch,ROOT,'cuda');old=nets['dual']
        with np.load(ROOT/p['reference41r']['heads_path']) as z:template=a.x.repair.materialize(torch,old,dict(z))
        models={}
        with zipfile.ZipFile(archive) as z:
            freeze=json.loads(z.read('nomination_freeze.json'));rows=[]
            for arm in a.ARMS[1:]:
                import io
                model=copy.deepcopy(template);model.load_state_dict(torch.load(io.BytesIO(z.read(arm+'.pt')),map_location='cuda',weights_only=True));model.eval()
                for parameter in model.parameters():parameter.requires_grad_(False)
                models[arm]=model
                if arm in freeze['weights_sha256']:assert a.n.weights_hash(model)==freeze['weights_sha256'][arm]
            assert a.n.weights_hash(models['C0'])==p['baseline']['logical_weights_sha256']
            for name in ('nomination_freeze.json','preregistration.json','stress_regressions.json','endpoint_preflight.json','execution.json'):
                if name in z.namelist():(out/name).write_bytes(z.read(name))
            for panel in p['panels']:
                rows.extend(json.loads(z.read(panel+'/results.json')))
        before={k:a.n.weights_hash(v) for k,v in models.items()}
        with np.load(cache/'native_shadow_traces.npz') as z:data=dict(z)
        cost_cfg=a.n.old_native.x.calibration_config(prior,cfg)
        # No autograd graphs may capture inference tensors in the CUDA stepper.
        with torch.inference_mode():
            costs=a.n.benchmark(torch,dict(dual=models['C0'],**{k:v for k,v in models.items() if k!='C0'}),parent,data,(cost_cfg,ca,base),p,out,'cuda')
        assert len(costs)==2*len(p['cost']['batches']) and all(r['all_outputs_finite'] for r in costs)
        assert before=={k:a.n.weights_hash(v) for k,v in models.items()}
        summary={}
        for panel in p['panels']:
            summary[panel]={}
            for h in sorted({r['horizon_ms'] for r in rows if r['panel']==panel}):
                summary[panel][str(h)]={arm:{mode:dict(eligible=sum(r['reference_valid'] for r in pool),passed=sum(r['reference_valid'] and r['current_passed'] and r['compartment_passed'] for r in pool),regressions=sum(r['reference_valid'] and r['regression_vs_f011'] for r in pool)) for mode in ('imposed','autonomous') for pool in [[r for r in rows if r['panel']==panel and r['horizon_ms']==h and r['arm']==arm and r['mode']==mode]]} for arm in a.ARMS[1:]}
        a.n.p.save_json(out/'final_report.json',dict(valid=True,complete=True,task='45c cost recovery',training_performed=False,selection_recomputed=False,thresholds_changed=False,abstraction_changed=False,weights_unchanged=True,all_parameters_finite=all(bool(torch.isfinite(v).all()) for m in models.values() for v in m.parameters()),all_traces_finite=all(r['finite'] for r in rows),source_archive_sha256=SOURCE_SHA,panels=summary,cost_rows=len(costs),release_authorized=False,speedup_claim=False))
    except BaseException as e:
        a.n.p.save_json(out/'failure_report.json',dict(valid=False,error=repr(e),traceback=traceback.format_exc()));raise
if __name__=='__main__':main()
