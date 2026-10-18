"""Preregister Task39e in the authoritative SQLite mirror; no Airtable calls."""
import hashlib
import json
from functools import lru_cache
from .mirror import ROOT,write_json
from .validated_batch import ValidatedBatchMirror
from src.giada_teacher.task39e_tail_support_heads import arms,contrast_pairs,CONFIG
def main():
    repo=ROOT.parent;spec=json.loads((repo/CONFIG).read_text());folder=repo/'experiments/results/task39e_protocol_v1';folder.mkdir(parents=True,exist_ok=True)
    m=ValidatedBatchMirror();ids={}
    def put(table,suffix,**fields):
        code=f'{table}-giada-task39e-tail-{suffix}-v1';rid=m.local_upsert(table,{'Codice stabile':code,**fields})['record_id'];ids[code]=rid;return rid
    @lru_cache(None)
    def ref(code):
        rr=m.query("SELECT record_id FROM v_records WHERE stable_code='"+code+"'")['rows'];assert len(rr)==1,code
        return rr[0]['record_id']
    paths=[CONFIG,'src/giada_teacher/task39e_tail_support_heads.py','scripts/run_task39e_tail_support_heads.py',
        'scripts/build_task39e_protocol.py','notebooks/giada_task39e_tail_support_heads.ipynb','tests/test_task39e_tail_support_heads.py',
        'scripts/colab.ps1','scripts/colab_checked_exec.py','scripts/colab_task39e_driver.py',
        'scripts/task39e_console_manager.py','docs/colab_console_operations.md','experiments/giada_modular_surrogate_roadmap.md',
        'experiments/results/task39e_protocol_v1/local_validation.json','research_memory/register_task39e.py']
    arts=[]
    for rel in paths:
        path=repo/rel
        arts.append(put('artifacts',rel,Nome='Task39e '+path.name,Tipo='Codice' if path.suffix in ('.py','.ps1') else 'Report',Versione='v1',Percorso=rel,
            **{'SHA-256':hashlib.sha256(path.read_bytes()).hexdigest(),'Dimensione byte':path.stat().st_size}))
    claims=[ref('claims-giada-task39d-current-residual-'+n+'-v1') for n in ('tail_precision_objective','stratified_coefficient_support','head_interference','long_variable_exposure','remaining_budget')]
    exp=put('experiments','suite',Nome='Task39e tail precision, coefficient support, heads and memory',Stato='Preregistrato',Ipotesi=claims,
        Descrizione=spec['matrix'],**{'Obiettivo informativo':spec['contrasts']+' '+spec['selection']})
    partitions=[ref(f'partitions-giada-task37-ionic-playground-{r}-v1') for r in ('fit','fit_extra','development','stress_diagnostic')]
    protocol=put('protocols','suite',Nome='Task39e30-member console protocol',Versione='v1',Modalità='Ricomposizione',Esperimento=[exp],
        Procedura=json.dumps(spec,ensure_ascii=False),**{'Partizioni autorizzate':partitions,'Artefatto preregistrazione':[arts[0]],
        'Hash preregistrazione':hashlib.sha256((repo/CONFIG).read_bytes()).hexdigest(),'Criteri di successo':spec['quality'],
        'Regole di arresto':spec['preflights']+' '+spec['limits'],'Registrato il':'2026-10-18'})
    impl=put('implementations','worker',Nome='Task39e console CUDA worker',**{'Repository URL':'https://github.com/Zagred47/giada',
        'Percorso entrypoint':'scripts/run_task39e_tail_support_heads.py','Artefatti codice':arts,
        'Ambiente':'Official Colab CLI0.7.4, WSL2, namedT4 session; no browser editing. Actual full commit recorded on dispatch.'})
    aa={}
    for a in arms():
        model=put('models',a['name'],Nome='Task39e '+a['name'],Ruolo='Surrogate',Descrizione=spec['architecture']+' '+json.dumps(a))
        aa[a['name']]=put('arms',a['name'],Nome='Task39e '+a['name'],Ruolo='Trattamento',Protocollo=[protocol],Modello=[model],
            Implementazione=[impl],Partizioni=partitions,**{'Configurazione residua':json.dumps(dict(arm=a,seeds=spec['seeds']))})
        for seed in spec['seeds']:
            put('runs',f'{a["name"]}-seed{seed}',Nome=f'Task39e {a["name"]} seed{seed}',Stato='Pianificata',Seed=str(seed),Braccio=[aa[a['name']]],
                **{'Configurazione effettiva':json.dumps(dict(arm=a,seed=seed,checkpoints=spec['checkpoints'])),
                   'Hardware e ambiente':'Prospective named Colab T4; CUDA preflights and result pending.'})
    for axis,a,z in contrast_pairs(arms()):
        put('contrasts',a+'-vs-'+z,Nome='Task39e '+axis+': '+a+' vs '+z,Protocollo=[protocol],Bracci=[aa[a],aa[z]],
            **{'Contrasto e coefficienti':spec['contrasts'],'Soglia interpretativa':'Median<=.8, all3<1, worst<=1.05; strict quality unchanged',
               'Confondenti controllati':spec['training']})
    for name,description in [('quality',spec['quality']),('selection',spec['selection']),('oracles',spec['oracles']),('cost',spec['cost'])]:
        metric=put('metrics',name,Nome='Task39e '+name,Formula=description)
        put('evaluations',name,Nome='Task39e '+name,Versione='v1',Metrica=[metric],Protocollo=[protocol],Partizioni=partitions[2:],Ruolo='Diagnostica',Target=description,
            **{'Aggregazione e pesi':'Per-channel, regime, timing and worst-case; paired seeds, no independent confirmation'})
    decision=put('decisions','authorize',Nome='Authorize Task39e console experiment',Esito='Modificare',Risultati=[ref('findings-giada-task39d-current-outcome-v1')],
        Motivazione=spec['matrix']+' '+spec['limits'],**{'Data':'2026-10-18'})
    put('actions','execute',Nome='Execute Task39e on Colab console',Stato='Proposta',Descrizione=spec['matrix'],**{'Decisioni di origine':[decision]})
    m.local_upsert('actions',{'Codice stabile':'actions-giada-task39d-current-proposed39e-v1','Stato':'Conclusa',
        'Descrizione':'User authorized preparation, preregistration, push and Colab console dispatch. Execution tracked separately.'})
    print(json.dumps(dict(stage='staged',records=len(ids))),flush=True)
    m.commit_batch();write_json(ROOT/'data/airtable_snapshot.json',m.export_snapshot());v=m.verify();assert v['valid']
    write_json(folder/'memory_receipt.json',dict(valid=True,records=ids,verification=v,airtable_accessed=False,scientific_result_claimed=False))
    print(json.dumps(dict(valid=True,records=len(ids))),flush=True)
if __name__=='__main__':main()
