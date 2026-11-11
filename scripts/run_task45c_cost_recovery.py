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
        transport=json.loads((ROOT/'experiments/task45c_cost_transport_v2.json').read_text())
        assert a.sha(archive)==transport['sha256'] and archive.stat().st_size==transport['bytes']
        cache=out.parent/'verified_source45c';a.x.repair.p.r.verify_source(ROOT,source,cache)
        os.environ['GIADA_TASK41N_FROZEN']=str(cache/'frozen_source.zip.bin')
        nets,parent=a.n.load_models(torch,ROOT,'cuda');old=nets['dual']
        with np.load(ROOT/p['reference41r']['heads_path']) as z:template=a.x.repair.materialize(torch,old,dict(z))
        models={}
        with zipfile.ZipFile(archive) as z:
            assert z.testzip() is None
            for name,digest in transport['members'].items():assert hashlib.sha256(z.read(name)).hexdigest()==digest
            freeze=json.loads(z.read('nomination_freeze.json'));rows=[]
            inherited=json.loads(z.read('accuracy_summary.json'));assert inherited['source_archive_sha256']==SOURCE_SHA
            for arm in a.ARMS[1:]:
                import io
                model=copy.deepcopy(template);model.load_state_dict(torch.load(io.BytesIO(z.read(arm+'.pt')),map_location='cuda',weights_only=True));model.eval()
                for parameter in model.parameters():parameter.requires_grad_(False)
                models[arm]=model
                if arm in freeze['weights_sha256']:assert a.n.weights_hash(model)==freeze['weights_sha256'][arm]
            assert a.n.weights_hash(models['C0'])==p['baseline']['logical_weights_sha256']
            for name in ('nomination_freeze.json','preregistration.json','stress_regressions.json','endpoint_preflight.json','execution.json'):
                if name in z.namelist():(out/name).write_bytes(z.read(name))
        before={k:a.n.weights_hash(v) for k,v in models.items()}
        with np.load(cache/'native_shadow_traces.npz') as z:data=dict(z)
        cost_cfg=a.n.old_native.x.calibration_config(prior,cfg)
        # No autograd graphs may capture inference tensors in the CUDA stepper.
        with torch.inference_mode():
            costs=a.n.benchmark(torch,dict(dual=models['C0'],**{k:v for k,v in models.items() if k!='C0'}),parent,data,(cost_cfg,ca,base),p,out,'cuda')
        assert len(costs)==2*len(p['cost']['batches']) and all(r['all_outputs_finite'] for r in costs)
        assert before=={k:a.n.weights_hash(v) for k,v in models.items()}
        summary=inherited['panels']
        a.n.p.save_json(out/'final_report.json',dict(valid=True,complete=True,task='45c cost recovery',training_performed=False,selection_recomputed=False,thresholds_changed=False,abstraction_changed=False,weights_unchanged=True,all_parameters_finite=all(bool(torch.isfinite(v).all()) for m in models.values() for v in m.parameters()),all_traces_finite=inherited['all_traces_finite'],source_archive_sha256=SOURCE_SHA,panels=summary,cost_rows=len(costs),release_authorized=False,speedup_claim=False))
    except BaseException as e:
        a.n.p.save_json(out/'failure_report.json',dict(valid=False,error=repr(e),traceback=traceback.format_exc()));raise
if __name__=='__main__':main()
