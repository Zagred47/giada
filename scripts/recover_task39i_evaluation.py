"""Evaluation-only recovery; never constructs an optimizer or calls training."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse, hashlib, json, shutil, subprocess, sys, time, traceback, zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from src.giada_teacher import task39i_coupled_relaxation as i

def choose(curves, count):
    best=[None]*count; chosen=[None]*count
    for row in sorted(curves,key=lambda r:r['step']):
        assert len(row['panel'])==count
        for j, panel in enumerate(row['panel']):
            rank=panel['rank']
            if rank is None:continue
            rank=tuple(rank)
            if best[j] is None or rank<best[j]:best[j]=rank;chosen[j]=row['step']
    assert all(x is not None for x in chosen)
    return best,chosen

def run(archive, output, device):
    resumed=output.exists()
    output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    if device=='cuda':torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for n in z.namelist():
            target=(output/n).resolve();assert target.is_relative_to(output.resolve())
        if not resumed:z.extractall(output)
    spec=json.loads((output/'preregistration.json').read_text())
    assert spec==json.loads((ROOT/i.CONFIG).read_text())
    quality=json.loads((ROOT/i.p.CONFIG).read_text());assert i.p.digest(ROOT/i.p.CONFIG)==spec['quality_config_sha256']
    assert json.loads((output/'revision.json').read_text())['revision']=='cf16e9748b027cee9aed3802ffce87ed47fb7e0f'
    assert not (output/'final_report.json').exists()
    if resumed:assert (output/'selection_freeze.json').exists(),'Only frozen evaluation can resume'
    for row in json.loads((ROOT/spec['dataset']/'dataset_manifest.json').read_text()):assert i.p.digest(ROOT/spec['dataset']/row['file'])==row['sha256']
    source_sha=i.p.digest(archive); checkpoint_sha=i.p.digest(output/'checkpoint_120000.pt')
    cohorts=[]
    for arch in ('plain','channel','generic','monotone','transfer'):
        definitions=[a for a in i.arms() if a['architecture']==arch]
        model=i.model_class(torch)(arch,spec['seeds']*len(definitions)).to(device)
        cohorts.append(dict(key=arch,definitions=definitions,model=model))
    def load(step):
        state=torch.load(output/f'checkpoint_{step}.pt',map_location=device,weights_only=True)
        for group in cohorts:
            group['model'].load_state_dict(state[group['key']]);group['model'].eval()
            assert all(bool(torch.isfinite(v).all()) for v in group['model'].parameters())
    curves=json.loads((output/'training_curves.json').read_text());assert [r['step'] for r in curves] in (spec['checkpoints'][:-1],spec['checkpoints'])
    print('[39i recovery] all saved checkpoint ranks recovered; evaluating missing120000 panel',flush=True)
    if curves[-1]['step']!=120000:
        load(120000);normal=json.loads((ROOT/spec['normalizers']).read_text())
        dev,held=i.h.development(ROOT,spec,quality)
        with torch.no_grad():common,parts,ranks,panel=i.e.checkpoint_scores(torch,cohorts,dev,held,normal,spec,device,quality)
        curves.append(dict(step=120000,common_development_score=common.tolist(),panel=panel,evaluation_recovered=True))
        i.p.save_json(output/'training_curves.json',i.e.diagnostic_json(curves))
    count=sum(len(g['definitions'])*3 for g in cohorts);best,chosen=choose(curves,count)
    saved=[None]*count
    for step in sorted(set(chosen)):
        load(step);j=0
        for group in cohorts:
            for k in range(group['model'].a.shape[0]):
                if chosen[j]==step:saved[j]={n:v[k].detach().cpu().clone() for n,v in group['model'].named_parameters()}
                j+=1
    j=0
    with torch.no_grad():
        for group in cohorts:
            for k in range(group['model'].a.shape[0]):
                for name,v in group['model'].named_parameters():v[k].copy_(saved[j][name])
                j+=1
    torch.save({g['key']:g['model'].state_dict() for g in cohorts},output/'frozen_checkpoint.pt')
    definitions=[a for g in cohorts for a in g['definitions']]
    freeze=dict(arms=definitions,seeds=spec['seeds'],chosen_steps=chosen,ranks=best,development_only=True,stress_used_for_selection=False,independent_confirmation_accessed=False,checkpoint_sha256=i.p.digest(output/'frozen_checkpoint.pt'))
    freeze['freeze_sha256']=hashlib.sha256(json.dumps(freeze,sort_keys=True).encode()).hexdigest();i.p.save_json(output/'selection_freeze.json',freeze)
    print('[39i recovery] selection frozen; full evaluation begins',flush=True)
    baseline=json.loads((ROOT/i.old.CONFIG).read_text());compare=json.loads((ROOT/i.h.CONFIG).read_text())
    es=dict(spec,architectures=[a['name'] for a in definitions]+['frozen39h_H8','frozen39g_A0','frozen39g_A6'])
    for k in ('baseline_parent','baseline_checkpoint'):es[k]=baseline[k]
    assert i.p.digest(ROOT/baseline['baseline_parent'])==baseline['baseline_parent_sha256'] and i.p.digest(ROOT/baseline['baseline_checkpoint'])==baseline['baseline_sha256']
    models=i.split_models(torch,cohorts,spec['seeds'],device)+i.load_parents(torch,ROOT,spec,device,torch.float32)+i.h.load_parents(torch,ROOT,compare,device,torch.float32)
    if device=='cpu':
        from task39i_cpu_scanner import install
        install(i.old,torch,models,output,spec['numeric_atol'])
    with torch.no_grad():summary=i.old.evaluate(ROOT,output,es,models,torch,device,quality)
    rows=json.loads((output/'recurrence_evaluation.json').read_text());scores={}
    for a in definitions:
        scores[a['name']]={}
        for seed in spec['seeds']:
            rr=[r for r in rows if r['model']=='joint_w'+a['name'] and r['role']=='development' and r['seed']==seed];values=[i.q.severity(r['metrics']) for r in rr]
            scores[a['name']][str(seed)]=dict(median=float(np.median(values)),worst=max(values),passed=sum(r['metrics']['passed'] for r in rr))
    contrasts=[]
    for axis,control,treatment in i.contrast_pairs():
        ratios=[scores[treatment][str(s)]['median']/max(scores[control][str(s)]['median'],1e-12) for s in spec['seeds']]
        worst=[scores[treatment][str(s)]['worst']/max(scores[control][str(s)]['worst'],1e-12) for s in spec['seeds']]
        contrasts.append(dict(axis=axis,control=control,treatment=treatment,paired_ratios=ratios,worst_ratios=worst,supported=bool(np.median(ratios)<=.8 and max(ratios)<1 and max(worst)<=1.05)))
    i.p.save_json(output/'paired_contrasts.json',dict(scores=scores,contrasts=contrasts))
    invalid={j for row in curves for j,r in enumerate(row['panel']) if r['numerically_invalid']}
    unstable={definitions[j//3]['name'] for j in invalid};passing=[a['name'] for a in definitions if summary['joint_w'+a['name']]['all_development_passed'] and a['name'] not in unstable]
    assert i.p.digest(output/'checkpoint_120000.pt')==checkpoint_sha
    receipt=dict(valid=True,training_updates_performed=0,source_archive_sha256=source_sha,checkpoint120000_sha256=checkpoint_sha,selection='Original saved ranks through90000 plus recomputed120000;earliest lexicographic minimum',thresholds_changed=False,device=device,torch=torch.__version__,gpu_benchmark_recovered=False,original_training_revision='cf16e9748b027cee9aed3802ffce87ed47fb7e0f')
    i.p.save_json(output/'evaluation_recovery.json',receipt)
    final=dict(valid=True,task='39i',decision='CANDIDATE_REQUIRES_AUDIT' if passing else 'NO_GO_TASK40_BLOCKED',passing_arms=passing,numerically_unstable_arms=sorted(unstable),summaries=summary,thresholds_changed=False,independent_confirmation_accessed=False,speedup_claimed=False,limits=spec['limits'],evaluation_recovered=True)
    i.p.save_json(output/'final_report.json',final)
    result=Path(shutil.make_archive(str(output)+'_final','zip',output))
    i.p.save_json(Path(str(result)+'.receipt.json'),dict(valid=True,archive=str(result),sha256=i.p.digest(result),bytes=result.stat().st_size,runtime_independent=True))
    print('[39i recovery] '+json.dumps(final),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('archive',type=Path);parser.add_argument('output',type=Path);parser.add_argument('--device',default='cpu',choices=['cpu','cuda']);args=parser.parse_args()
    try:run(args.archive,args.output,args.device)
    except BaseException:
        traceback.print_exc();raise
