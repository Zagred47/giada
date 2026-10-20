"""Complete preregistered nonselectable diagnostics from immutable checkpoints."""
import json,sys,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from src.giada_teacher import task39i_coupled_relaxation as i
from task39i_cpu_scanner import install

def main(output):
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    spec=json.loads((output/'preregistration.json').read_text());quality=json.loads((ROOT/i.p.CONFIG).read_text())
    primary=(output/'frozen_checkpoint.pt').read_bytes();primary_sha=hashlib.sha256(primary).hexdigest()
    curves=json.loads((output/'training_curves.json').read_text());cohorts=[]
    for arch in ('plain','channel','generic','monotone','transfer'):
        aa=[a for a in i.arms() if a['architecture']==arch];cohorts.append(dict(key=arch,definitions=aa,model=i.model_class(torch)(arch,spec['seeds']*len(aa))))
    definitions=[a for g in cohorts for a in g['definitions']];count=len(definitions)*3
    selected=np.argmin(np.asarray([r['common_development_score'] for r in curves]),axis=0)
    old_steps=[curves[k]['step'] for k in selected];report={}
    baseline=json.loads((ROOT/i.old.CONFIG).read_text());comparison=json.loads((ROOT/i.h.CONFIG).read_text())
    es=dict(spec,architectures=[a['name'] for a in definitions]+['frozen39h_H8','frozen39g_A0','frozen39g_A6'])
    for k in ('baseline_parent','baseline_checkpoint'):es[k]=baseline[k]
    for label,chosen in [('diagnostic_checkpoint_120000',[120000]*count),('nonselectable_old_score_selection',old_steps)]:
        folder=output/label;folder.mkdir(exist_ok=False);saved=[None]*count
        for step in sorted(set(chosen)):
            state=torch.load(output/f'checkpoint_{step}.pt',map_location='cpu',weights_only=True)
            j=0
            for group in cohorts:
                group['model'].load_state_dict(state[group['key']]);group['model'].eval()
                for k in range(group['model'].a.shape[0]):
                    if chosen[j]==step:saved[j]={n:v[k].detach().clone() for n,v in group['model'].named_parameters()}
                    j+=1
        j=0
        with torch.no_grad():
            for group in cohorts:
                for k in range(group['model'].a.shape[0]):
                    for name,v in group['model'].named_parameters():v[k].copy_(saved[j][name])
                    j+=1
        models=i.split_models(torch,cohorts,spec['seeds'],'cpu')+i.load_parents(torch,ROOT,spec,'cpu',torch.float32)+i.h.load_parents(torch,ROOT,comparison,'cpu',torch.float32)
        install(i.old,torch,models,folder,spec['numeric_atol'])
        print('[39i recovery diagnostic] '+label,flush=True)
        with torch.no_grad():summary=i.old.evaluate(ROOT,folder,es,models,torch,'cpu',quality)
        report[label]=dict(selection_eligible=False,summaries=summary,chosen_steps=chosen,primary_selection_changed=False)
        i.p.save_json(folder/'diagnostic_only.json',report[label])
    assert hashlib.sha256((output/'frozen_checkpoint.pt').read_bytes()).hexdigest()==primary_sha
    i.p.save_json(output/'recovered_nonselectable_diagnostics.json',dict(valid=True,training_updates_performed=0,primary_checkpoint_unchanged=True,diagnostics=report))
if __name__=='__main__':main(Path(sys.argv[1]))
