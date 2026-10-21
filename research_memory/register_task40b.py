"""Prospective Task40b protocol through the authoritative SQLite interface."""
import json,hashlib
from .mirror import ROOT,write_json
from .validated_batch import ValidatedBatchMirror
from src.giada_teacher import task40b_calcium_precision as b

def main():
    repo=ROOT.parent;spec=json.loads((repo/b.CONFIG).read_text());m=ValidatedBatchMirror();ids={}
    def ref(code):
        rows=m.query("SELECT record_id FROM v_records WHERE stable_code='"+code+"'")['rows'];assert len(rows)==1,code
        return rows[0]['record_id']
    def put(table,suffix,**fields):
        code=f'{table}-giada-task40b-calcium-{suffix}-v1'
        rid=m.local_upsert(table,{'Codice stabile':code,**fields})['record_id'];ids[code]=rid;return rid
    arts=[]
    for rel in [b.CONFIG,'src/giada_teacher/task40b_calcium_precision.py','scripts/build_task40b_protocol.py',
                'scripts/run_task40b_calcium_precision.py','scripts/probe_task40b_gpu.py','tests/test_task40b_calcium_precision.py',
                'notebooks/giada_task40b_calcium_precision.ipynb']:
        path=repo/rel;arts.append(put('artifacts',rel,Nome='Task40b '+path.name,Tipo='Codice' if path.suffix=='.py' else 'Report',Versione='v1',Percorso=rel,
            **{'SHA-256':hashlib.sha256(path.read_bytes()).hexdigest(),'Dimensione byte':path.stat().st_size}))
    hypothesis=json.loads((repo/'experiments/results/task40_audited_v1/synthesis.json').read_text())['hypotheses']
    claims=[ref('claims-giada-task40-result-'+h['id']+'-v1') for h in hypothesis]
    exp=put('experiments','suite',Nome='Task40b learned calcium kinetic precision matrix',Stato='Preregistrato',Ipotesi=claims,
        Descrizione=spec['architecture'],**{'Obiettivo informativo':'Distinguish relaxation precision, paired-response objective, FIT coverage, capacity and optimization budget without changing quality thresholds.'})
    generator=put('generators','atomic',Nome='Task40b independent paired-voltage anchors',Versione='v1',Procedura=spec['fit'],
        **{'Seed e RNG':json.dumps({k:spec[k] for k in ('fit_seed','selection_seed','stream_seed')}),'Artefatti codice':arts})
    dataset=put('datasets','atomic',Nome='Task40b nested FIT and atomic selection',Versione='v1',Generatore=[generator],Artefatti=[arts[0]],
        Schema='Paired V,Ca,initial4Ca gates,offlineu/z; no diagnostic trajectories inFIT',Copertura=spec['fit'],
        Fingerprint=hashlib.sha256((spec['fit']+str(spec['fit_seed'])).encode()).hexdigest(),**{'Numero esempi':4352})
    partitions=[]
    for name,role in [('fit','Fit'),('atomic_selection','Diagnostico')]:
        partitions.append(put('partitions',name,Nome='Task40b '+name,Dataset=[dataset],Ruolo=role,
            **{'Regola di assegnazione':spec['fit'] if name=='fit' else spec['selection'],
               'Restrizioni di accesso':'Independent atomic selection only. Historical development,causal andstress paths afterfreeze;notfreshconfirmation.'}))
    partitions += [ref('partitions-giada-task37-ionic-playground-'+role+'-v1') for role in ('development','stress_diagnostic')]
    partitions.append(ref('partitions-giada-task40-causal-opened-v1'))
    protocol=put('protocols','suite',Nome='Task40b paired calcium correction protocol',Versione='v1',Modalità='Ricomposizione',Esperimento=[exp],Procedura=json.dumps(spec),
        **{'Partizioni autorizzate':partitions,'Artefatto preregistrazione':[arts[0]],'Hash preregistrazione':hashlib.sha256((repo/b.CONFIG).read_bytes()).hexdigest(),
           'Criteri di successo':spec['qualification'],'Regole di arresto':'Nonfinite parameters or failed numerical/optimizer/graph floor stop technical interpretation. No automatic scientific restart.','Registrato il':'2026-10-21'})
    impl=put('implementations','worker',Nome='Task40b vectorized independent correction worker',
        **{'Repository URL':'https://github.com/Zagred47/giada','Percorso entrypoint':'scripts/run_task40b_calcium_precision.py',
           'Artefatti codice':arts,'Ambiente':'Colab console T4;21learnedmembers,oneCUDA Graph,externalPCrecovery'})
    armids={}
    for a in spec['arms']:
        name=a['name'];model=put('models',name,Nome='Task40b '+name,Ruolo='Surrogate',Descrizione=spec['architecture']+' '+json.dumps(a))
        armids[name]=put('arms',name,Nome='Task40b '+name,Ruolo='Baseline' if name=='B0' else 'Trattamento',Protocollo=[protocol],Modello=[model],Implementazione=[impl],Partizioni=partitions,
            **{'Configurazione residua':json.dumps(a)})
        for seed in spec['seeds']:
            put('runs',name+'-'+str(seed),Nome=f'Task40b {name} seed{seed}',Stato='Pianificata',Seed=str(seed),Braccio=[armids[name]],
                **{'Configurazione effettiva':json.dumps(dict(arm=a,seed=seed,checkpoints=spec['checkpoints'],training=name!='B0'))})
    for contrast in spec['contrasts']:
        put('contrasts',contrast['axis'],Nome='Task40b '+contrast['axis'],Protocollo=[protocol],Bracci=[armids[contrast[k]] for k in ('control','treatment')],
            **{'Contrasto e coefficienti':json.dumps(contrast),'Soglia interpretativa':spec['contrast_rule'],'Confondenti controllati':spec['execution']+' '+spec['capacity_caveat']})
    for kind,value in [('quality',spec['current_quality']),('causal',spec['qualification']),('selection',spec['selection']),('atomic',spec['loss'])]:
        metric=put('metrics',kind,Nome='Task40b '+kind,Formula=json.dumps(value))
        put('evaluations',kind,Nome='Task40b '+kind,Versione='v1',Metrica=[metric],Protocollo=[protocol],Partizioni=partitions,Ruolo='Diagnostica',Target=json.dumps(value))
    put('actions','execute',Nome='Execute Task40b after CUDA smoke',Stato='Proposta',Descrizione='Preregistered only; launch and external backup verification require separate actual receipt.',
        **{'Decisioni di origine':[ref('decisions-giada-task40-result-40b-not41-v1')]})
    print(json.dumps(dict(stage='staged',records=len(ids))),flush=True)
    m.commit_batch();write_json(ROOT/'data/airtable_snapshot.json',m.export_snapshot());v=m.verify();assert v['valid']
    folder=repo/'experiments/results/task40b_protocol_v1';folder.mkdir(exist_ok=True)
    write_json(folder/'memory_receipt.json',dict(valid=True,records=ids,verification=v,launched=False,airtable_accessed=False))
    print('Task40b protocol registered',flush=True)
if __name__=='__main__':main()
