"""Restore the original selection and evaluate; no optimizer or training calls."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse,hashlib,json,shutil,sys,time,traceback,uuid,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from src.giada_teacher import task41h_path_repair as h
from src.giada_teacher import task41h_evaluation as ev

def choose(curves,count):
    best={r:[None]*count for r in ('aggregate','stratified')};steps={r:[None]*count for r in best}
    for row in sorted(curves,key=lambda r:r['step']):
        for rule in best:
            assert len(row['ranks'][rule])==count
            for j,rank in enumerate(row['ranks'][rule]):
                rank=tuple(rank);assert all(np.isfinite(rank))
                if best[rule][j] is None or rank<best[rule][j]:best[rule][j]=rank;steps[rule][j]=row['step']
    assert all(v is not None for values in steps.values() for v in values)
    return best,steps

def finite(v):
    if torch.is_tensor(v):return bool(torch.isfinite(v).all())
    if isinstance(v,dict):return all(finite(x) for x in v.values())
    if isinstance(v,(tuple,list)):return all(finite(x) for x in v)
    return True

def install_backup_barriers(output):
    original=h.p.save_json;last=[0.]
    def barrier(label):
        if not os.environ.get('GIADA_RECOVERY_BARRIERS'):return
        token=label+'_'+uuid.uuid4().hex
        print('[GIADA recovery barrier] '+token,flush=True)
        ack=output.parent/'backup_ack.txt';deadline=time.monotonic()+900
        while not ack.exists() or ack.read_text().strip()!=token:
            if time.monotonic()>deadline:raise RuntimeError('Backup barrier acknowledgement timed out')
            time.sleep(.2)
        last[0]=time.monotonic()
    def save(path,value):
        path=Path(path);tmp=path.with_name(path.name+'.atomic.tmp')
        original(tmp,value);tmp.replace(path)
        if path.name in ('selection_freeze.json','final_report.json') or time.monotonic()-last[0]>=120:
            barrier('saved_'+path.stem)
    h.p.save_json=save
    return barrier

def run(archive,output,seal_path,device='cuda',selection_only=False):
    seal=json.loads(seal_path.read_text());assert seal['training_updates_performed']==0
    assert h.p.digest(archive)==seal['source_archive_sha256']
    assert not output.exists() or not any(output.iterdir()),'Fresh recovery destination required'
    output.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for name in z.namelist():assert (output/name).resolve().is_relative_to(output.resolve())
        z.extractall(output)
    shutil.copy2(output/'process.log',output/'original_training_process.log')
    assert json.loads((output/'revision.json').read_text())['revision']==seal['original_training_revision']
    spec,cfg,ca,base=h.contract(ROOT)
    assert json.loads((output/'preregistration.json').read_text())==spec
    assert h.p.digest(ROOT/h.CONFIG)==seal['protocol_sha256']
    for name,digest in seal['checkpoint_sha256'].items():assert h.p.digest(output/name)==digest,name
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    assert device=='cuda' and torch.cuda.is_available(),'Original CUDA selection backend required'
    assert torch.__version__==seal['torch_version'] and torch.cuda.get_device_name()==seal['GPU']
    barrier=install_backup_barriers(output)
    h.p.save_json(output/'evaluation_recovery.json',dict(valid=True,phase='restored',training_updates_performed=0,
        thresholds_changed=False,source=seal,recovery_code_revision=__import__('subprocess').check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()))
    def forbidden(*a,**k):raise RuntimeError('Training/optimizer step forbidden during evaluation recovery')
    h.run=forbidden;h.g.Adam.step=forbidden;torch.optim.Optimizer.step=forbidden
    checkpoints={}
    for step in spec['checkpoints']:
        value=torch.load(output/f'checkpoint_{step}.pt',map_location='cpu',weights_only=False)
        assert value['step']==step and finite(value)
        checkpoints[step]=value['correction'].clone()
    controls=h.frozen_controls(torch,ROOT,spec,device);parent=controls['E_Q0'];model=h.make_model(torch,parent,h.arms(),device)
    for par in model.parameters():par.requires_grad_(False)
    atomic,raw=h.atomic_data(torch,parent,spec,device)
    with np.load(output/'fit_atomic.npz') as saved:
        assert set(saved.files)==set(raw)
        for key in saved.files:assert np.array_equal(saved[key],raw[key]),key
    ids=torch.arange(spec['batch'],device=device)[None].expand(30,-1)
    with torch.no_grad():norm=h.f.atomic_terms(torch,model.f_correction,model.active,atomic,ids).detach().clamp_min(1e-12)
    del atomic,raw
    selection,_=h.atomic_data(torch,parent,spec,device,selection=True)
    selection_tail=h.path_tail_data(torch,parent,spec,device,count=spec['selection_anchors'],selection=True)
    h.f.ee.differentiable_constants()
    sf,ss=h.f.feedback_data(torch,spec,cfg,ca,base,device,selection=True)
    view=h.FeedbackView(torch,parent,model)
    dev=h.d.development(spec['new_seeds']['development'])
    with np.load(output/'development_inputs.npz') as saved:
        for key in saved.files:assert np.array_equal(saved[key],dev[key]),key
    with torch.no_grad():parent_rows=h.c.shadow_evaluate(torch,[parent],dev,spec['quality'],device)
    curves=json.loads((output/'training_curves.json').read_text());assert [r['step'] for r in curves]==spec['checkpoints'][:-1]
    si=torch.arange(spec['selection_anchors'],device=device)[None].expand(30,-1)
    # Replay step0 to detect backend, seed, normalization or ranking drift.
    for step in (0,spec['checkpoints'][-1]):
        with torch.no_grad():model.f_correction.copy_(checkpoints[step].to(device))
        models=[h.extract(torch,model,k,parent,device) for k in range(10)]
        with torch.no_grad():
            point=h.f.atomic_terms(torch,model.f_correction,model.active,selection,si)/norm
            response=h.g.feedback_terms(torch,model,view,sf,ss,torch.arange(spec['selection_feedback_anchors'],device=device),cfg,ca,base)
            ca0=h.path_loss(torch,model,selection_tail,torch.arange(spec['selection_anchors'],device=device),force_history=0)
            ca1=h.path_loss(torch,model,selection_tail,torch.arange(spec['selection_anchors'],device=device),force_history=1)
            measured=h.c.shadow_evaluate(torch,models,dev,spec['quality'],device)
        ranks={r:[] for r in ('aggregate','stratified')}
        for j,rows in enumerate(measured):
            for rule in ranks:
                value=response[j]+((ca0[j]+ca1[j])/2 if rule=='aggregate' else torch.maximum(ca0[j],ca1[j]))
                ranks[rule].append(h.f.ranks(rows,parent_rows[j%3],value,point[j].sum(),spec['quality']))
        if step==0:
            before=np.asarray([curves[0]['ranks'][r] for r in ranks]);after=np.asarray([ranks[r] for r in ranks])
            assert np.array_equal(before,after),('Original selection rank parity failed',float(np.max(abs(before-after))))
            h.p.save_json(output/'selection_parity.json',dict(valid=True,step=0,all60_rank_vectors_bitwise_equal=True,training_updates_performed=0))
        else:
            curves.append(dict(step=step,ranks=ranks,point=point.tolist(),constant_tail=ca0.tolist(),ramp_tail=ca1.tolist(),evaluation_recovered=True))
            h.p.save_json(output/'training_curves.json',curves);h.p.save_json(output/f'development_{step}.json',measured)
        print(f'[GIADA Task41h recovery] checkpoint selection replay {step}',flush=True)
    best,chosen=choose(curves,30)
    def weights(rule):return torch.stack([checkpoints[chosen[rule][j]][j] for j in range(30)]).to(device)
    primary='stratified';ww=weights(primary)
    trained={a['name']:h.extract(torch,model,k,parent,device,ww) for k,a in enumerate(h.arms())}
    agg=weights('aggregate')
    for k,a in enumerate(h.arms()):
        if chosen['aggregate'][3*k:3*k+3]!=chosen[primary][3*k:3*k+3]:trained['aggregate_'+a['name']]=h.extract(torch,model,k,parent,device,agg)
    assert sorted(k for k in trained if k.startswith('aggregate_'))==seal['observed_aggregate_models']
    nominee=min(range(10),key=lambda k:tuple(max(best[primary][3*k+s][i] for s in range(3)) for i in range(len(best[primary][0])))+(k,))
    torch.save({k:v.state_dict() for k,v in trained.items()},output/'frozen_checkpoint.pt')
    freeze=dict(arms=h.arms(),seeds=spec['seeds'],chosen_steps=chosen[primary],ranks=best[primary],selection_rules=dict(steps=chosen,ranks=best),primary_selection=primary,nominee=h.arms()[nominee]['name'],checkpoint_sha256=h.p.digest(output/'frozen_checkpoint.pt'),opened_regressions_used_for_selection=False,independent_confirmation=False,parent='E_Q0')
    freeze['freeze_sha256']=hashlib.sha256(json.dumps(freeze,sort_keys=True).encode()).hexdigest();h.p.save_json(output/'selection_freeze.json',freeze)
    if selection_only:return
    with torch.inference_mode():report=ev.evaluate(torch,ROOT,output,controls,trained,spec,cfg,ca,base,device)
    for name,digest in seal['checkpoint_sha256'].items():assert h.p.digest(output/name)==digest,name
    assert h.p.digest(output/'frozen_checkpoint.pt')==freeze['checkpoint_sha256']
    report.update(evaluation_recovered=True,training_updates_performed=0,thresholds_changed=False)
    h.p.save_json(output/'final_report.json',report)
    print('[GIADA Task41h recovery] completed '+report['decision'],flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('output',type=Path);ap.add_argument('--archive',type=Path,required=True);ap.add_argument('--seal',type=Path,required=True);ap.add_argument('--selection-only',action='store_true');a=ap.parse_args()
    try:run(a.archive,a.output,a.seal,selection_only=a.selection_only)
    except BaseException as exc:
        a.output.mkdir(parents=True,exist_ok=True)
        (a.output/'failure_report.json').write_text(json.dumps(dict(valid=False,technical_failure=True,training_updates_performed=0,error=repr(exc),traceback=traceback.format_exc()),indent=2))
        traceback.print_exc();raise
