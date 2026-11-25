import copy,io,json,zipfile,hashlib
from . import task52g_gradient_preservation as g
f=g.f
def plan(root):
    p=json.loads((root/'experiments/task52i_preregistration_v1.json').read_text());assert p['task']=='52i' and not p['thresholds_changed'] and not p['abstraction_changed'];return p
def run(torch,root,source,cost,recovery,package,warm,out,device='cuda',override=None,coupled_override=None):
    p=override or plan(root)
    def train(torch,p,net,data,aut,cal,out,ordinary_count):
        baseline=copy.deepcopy(net)
        with zipfile.ZipFile(warm) as z:
            manifest=json.loads(z.read('source_manifest.json'));assert manifest['source_sha256']==p['warmstart']['source_sha256']
            raw=z.read('selected.pt');assert hashlib.sha256(raw).hexdigest()==manifest['selected_sha256']
            state=torch.load(io.BytesIO(raw),map_location=device,weights_only=True)
            freeze=json.loads(z.read('selection_freeze.json'));ix=p['warmstart']['indices'];assert [freeze['updates'][i] for i in ix]==p['warmstart']['selected_updates']
        indices=ix*len(p['arms']);net.load_state_dict({k:v[indices].clone() for k,v in state.items()})
        f.a.n.p.save_json(out/'warmstart_contract.json',dict(valid=True,indices=ix,updates=p['warmstart']['selected_updates'],same_for_all_arms=True,optimizer_reset=True,original_development_guard_retained=True))
        g.train(torch,p,net,data,aut,cal,out,ordinary_count,baseline_model=baseline)
    return f.run(torch,root,source,cost,recovery,package,out,device,p,coupled_override,train_fn=train,task_name='52i')
