"""Frozen exploratory transfer to new paired stimulus variants."""
import copy,hashlib,io,json,sys,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.giada_teacher.task52i_solution_retention import f

def panel(p):
    import numpy as np
    cases,inj,mult,rev=f.parent.design(p)
    rng=np.random.default_rng(95320261126)
    variants=[(0.85,13),(1.15,27)]
    stimuli=[]
    for gain,delay in variants:
        x=np.zeros_like(inj);shift=round(delay/p['dt_ms'])
        for k in range(len(p['protocols'])):
            ix=slice(k*len(p['interventions']),(k+1)*len(p['interventions']))
            x[shift:,ix]=inj[:-shift,ix]*gain*rng.uniform(.96,1.04)
        stimuli.append(x)
    return [dict(c,protocol=f"{c['protocol']}_variant{i}",variant=i,gain=g,delay_ms=d) for i,(g,d) in enumerate(variants) for c in cases],np.concatenate(stimuli,axis=1),np.tile(mult,(2,1)),np.tile(rev,(2,1))

def main():
    import torch,numpy as np
    source,cost,recovery,package,warm,out=map(Path,sys.argv[1:7]);out.mkdir(exist_ok=False)
    pin=f.a.n.contract(ROOT)[0]
    assert torch.cuda.is_available() and torch.__version__==pin['torch'] and torch.cuda.get_device_name()==pin['gpu']
    torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False
    p=f.parent.plan(ROOT); h=json.loads((ROOT/'experiments/task52j_preregistration_v1.json').read_text())
    with zipfile.ZipFile(warm) as z:
        raw=z.read('selected.pt');m=json.loads(z.read('source_manifest.json'));freeze=json.loads(z.read('selection_freeze.json'))
        assert m['source_sha256']=='695dfdd61f9343fd61aec45c85eae0ac23b684c54e2ba1da190d08cb306758ab'
        assert hashlib.sha256(raw).hexdigest()==m['selected_sha256']
        assert [(x['arm'],x['seed'],u) for x,u in zip(freeze['members'][3:6],freeze['updates'][3:6])]==[('long_current',17,400),('long_current',29,0),('long_current',43,400)]
        state=torch.load(io.BytesIO(raw),map_location='cuda',weights_only=True)
        cal=torch.load(io.BytesIO(z.read('calibration.pt')),map_location='cuda',weights_only=True)
    models=f.parent.load(torch,ROOT,package,source,cost,recovery,out,'cuda')
    h['arms']=[h['arms'][1]]
    model=f.c.model_class(torch)(h,cal,state,'cuda')
    model.load_state_dict({k:v[3:6] for k,v in state.items()});model.eval().requires_grad_(False)
    assert all(torch.isfinite(v).all() for v in model.parameters())
    models['compact']=model
    before={k:f.a.n.weights_hash(v) for k,v in models.items()}
    f.a.n.p.save_json(out/'task53_freeze.json',dict(source=m,indices=[3,4,5],updates=[400,0,400],weights=before,training_updates=0,logical_states=10))
    cases,inj,mult,rev=panel(p)
    f.a.n.p.save_json(out/'cases.json',cases)
    p['arms']=['canonical64','canonical32','P','compact','compact_imposed']
    inj,mult,rev=[torch.tensor(x,device='cuda',dtype=torch.float64) for x in (inj,mult,rev)]
    physics=f.a.n.contract(ROOT)[1:]
    with torch.no_grad():
        ref=f.parent.reference(torch,p,inj,mult,rev,physics,'cuda')
        trace=f.parent.simulate(torch,p,models,inj,mult,rev,physics,ref,'cuda',out)
        np.savez_compressed(out/'traces.npz',**trace)
        summary,rows=f.parent.evaluate(p,trace,ref,cases,out)
        assert all(r['passed'] for r in rows if r['arm']=='canonical64')
        f.parent.benchmark(torch,p,models,'cuda',out)
    assert before=={k:f.a.n.weights_hash(v) for k,v in models.items()}
    f.a.n.p.save_json(out/'final_report.json',dict(task='53',complete=True,exploratory=True,thresholds_changed=False,abstraction_changed=False,training_updates=0,selection_recomputed=False,promotion_authorized=False,summary=summary,limits='Frozen 52j long_current failed all composite cases on the reused panel; new timing/amplitude variants test transfer within existing protocol families, not definitive independent validation. Cost is not whole-neuron speed.'))

if __name__=='__main__':main()
