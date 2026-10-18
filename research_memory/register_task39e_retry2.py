"""Register checkpoint-level diagnosis, causal CUDA regression and retry2."""
import hashlib,json
from .mirror import ROOT,write_json
from .validated_batch import ValidatedBatchMirror
from src.giada_teacher.task39e_tail_support_heads import arms,CONFIG
def main():
    repo=ROOT.parent;folder=repo/'experiments/results/task39e_retry2_v1';launch=json.loads((folder/'launch.json').read_text())
    audit=json.loads((repo/'experiments/results/task39e_instability_v1/checkpoint_audit.json').read_text())
    probe=json.loads((repo/'experiments/results/task39e_instability_v1/cuda_lifetime_probe.json').read_text())
    m=ValidatedBatchMirror();ids={}
    def ref(code):
        r=m.query("SELECT record_id FROM v_records WHERE stable_code='"+code+"'")['rows'];assert len(r)==1;return r[0]['record_id']
    def put(table,suffix,**fields):
        code=f'{table}-giada-task39e-lifetime-{suffix}-v1';rid=m.local_upsert(table,{'Codice stabile':code,**fields})['record_id'];ids[code]=rid;return rid
    arts=[]
    for rel in ('experiments/task39e_instability_diagnosis.md','experiments/results/task39e_instability_v1/checkpoint_audit.json','experiments/results/task39e_instability_v1/cuda_lifetime_probe.json','experiments/results/task39e_instability_v1/failure_report.json','experiments/results/task39e_instability_v1/fit_gradient_probes.json','experiments/results/task39e_retry2_v1/launch.json'):
        p=repo/rel;arts.append(put('artifacts',p.name,Nome='Task39e lifetime '+p.name,Tipo='Report',Versione='retry2',Percorso=rel,**{'SHA-256':hashlib.sha256(p.read_bytes()).hexdigest(),'Dimensione byte':p.stat().st_size}))
    exp=ref('experiments-giada-task39e-tail-suite-v1');runs=[]
    for arm in arms():
        br=ref(f'arms-giada-task39e-tail-{arm["name"]}-v1')
        for seed in (17,29,43):
            code=f'runs-giada-task39e-tail-{arm["name"]}-seed{seed}-v1'
            m.local_upsert('runs',{'Codice stabile':code+'-retry1','Stato':'Fallita','Braccio':[br],'Validità tecnica':'Preserved failure at60000; evaluator gate=None TypeError;17/30 nonfinite. CUDA optimizer-state lifetime defect reproduced causally. Not a completed quality comparison.'})
            runs.append(put('runs',arm['name']+'-seed'+str(seed)+'-retry2',Nome=f'Task39e retry2 {arm["name"]} seed{seed}',Stato='In corso',Seed=str(seed),Braccio=[br],**{'Artefatti prodotti':arts,'Configurazione effettiva':json.dumps(dict(arm=arm,seed=seed,launch=launch)),'Hardware e ambiente':'Colab T4, pinned commit, official CLI, verified external checkpoint copies','Validità tecnica':'Optimizer owners retained for graph lifetime; failed checkpoints never selected. Scientific preregistration unchanged; full result pending.'}))
    finding=put('findings','implementation-failure',Nome='Task39e CUDA Adam ownership corruption reproduced',Esito='Misto',Esperimenti=[exp],**{'Risultato':(repo/'experiments/task39e_instability_diagnosis.md').read_text(),'Limitazioni':'Exact first old failing minibatch unknown; states/RNG missing from old run. No rejection of architecture/coverage hypotheses from incomplete trials. No full quality outcome yet.'})
    put('analyses','checkpoint-and-cuda',Nome='Task39e preserved checkpoint audit and causal allocator stress',Origine='Post hoc',Versione='v1',Procedura='scripts/diagnose_task39e_instability.py; scripts/probe_task39e_gradients.py; scripts/task39e_cuda_lifetime_probe.py',Configurazione=json.dumps(dict(checkpoint_audit=audit,cuda_probe=probe)),**{'Input artefatti':arts,'Run analizzati':runs,'12 · Risultati scientifici':[finding]})
    put('lessons','state-lifetime',Nome='CUDA graphs require ownership of warmup optimizer states',Lezione='Retain Adam and its eagerly allocated moments for entire CUDA Graph lifetime. Graph capture alone is not state ownership. Test allocator reuse plus parity and save moments/RNG externally.',Risultati=[finding],Limiti='Paired T4 regression proves implementation mechanism, not biological fidelity or full-run success.')
    print(json.dumps(dict(stage='staged',records=len(ids))),flush=True);m.commit_batch();write_json(ROOT/'data/airtable_snapshot.json',m.export_snapshot());v=m.verify();assert v['valid']
    write_json(folder/'memory_receipt.json',dict(valid=True,records=ids,verification=v,airtable_accessed=False));print(json.dumps(dict(valid=True,records=len(ids))),flush=True)
if __name__=='__main__':main()
