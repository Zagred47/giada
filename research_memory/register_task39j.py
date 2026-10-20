"""Prospective39j registration, separate from actual dispatch."""
import hashlib,json
from .mirror import ROOT,write_json
from .validated_batch import ValidatedBatchMirror
from .register_task39i_result import HYPOTHESES
from src.giada_teacher import task39j_atomic_relaxation as j

def main():
    repo=ROOT.parent;spec=json.loads((repo/j.CONFIG).read_text());m=ValidatedBatchMirror();ids={}
    def ref(code):
        rows=m.query("SELECT record_id FROM v_records WHERE stable_code='"+code+"'")['rows'];assert len(rows)==1,code
        return rows[0]['record_id']
    def put(table,suffix,**fields):
        code=f'{table}-giada-task39j-atomic-{suffix}-v1'
        rid=m.local_upsert(table,{'Codice stabile':code,**fields})['record_id'];ids[code]=rid;return rid
    arts=[]
    for rel in [j.CONFIG,'src/giada_teacher/task39j_atomic_relaxation.py','scripts/build_task39j_protocol.py',
                'scripts/run_task39j_atomic_relaxation.py','scripts/probe_task39j_gpu.py',
                'tests/test_task39j_atomic_relaxation.py','notebooks/giada_task39j_atomic_relaxation.ipynb']:
        path=repo/rel;arts.append(put('artifacts',rel,Nome='Task39j '+path.name,Tipo='Codice' if path.suffix=='.py' else 'Report',Versione='v1',Percorso=rel,
                    **{'SHA-256':hashlib.sha256(path.read_bytes()).hexdigest(),'Dimensione byte':path.stat().st_size}))
    claims=[ref('claims-giada-task39i-result-'+h[0]+'-v1') for h in HYPOTHESES]
    experiment=put('experiments','suite',Nome='Task39j atomic relaxation precision',Stato='Preregistrato',Ipotesi=claims,
        Descrizione=json.dumps(spec['matrix']),**{'Obiettivo informativo':spec['optimization']})
    partitions=[ref(f'partitions-giada-task37-ionic-playground-{role}-v1') for role in ('fit','fit_extra','development','stress_diagnostic')]
    generator=put('generators','fit',Nome='Task39j independent held-input anchors',Versione='v1',Procedura=spec['support'],
        **{'Seed e RNG':json.dumps(dict(fit=spec['fit_seed'],holdout=spec['holdout_seed'],stream=spec['stream_seed'])),'Artefatti codice':arts})
    dataset=put('datasets','atomic',Nome='Task39j ordinary and sensitive anchors',Versione='v1',Generatore=[generator],Artefatti=[arts[0]],Schema='V,Ca,initial18gates,offlineu/z,ticks',
        Copertura=spec['support'],Fingerprint='recipe:'+hashlib.sha256(spec['support'].encode()).hexdigest(),**{'Numero esempi':2304})
    for role in ('uniform','sensitive','atomic_holdout'):
        partitions.append(put('partitions',role,Nome='Task39j '+role,Ruolo='Fit' if role!='atomic_holdout' else 'Diagnostico',Dataset=[dataset],
            **{'Regola di assegnazione':spec['support'],'Restrizioni di accesso':'No evaluation data copying. Atomic holdout descriptive, not selection or independent confirmation.'}))
    protocol=put('protocols','suite',Nome='Task39j paired precision protocol',Versione='v1',Modalità='Ricomposizione',Esperimento=[experiment],Procedura=json.dumps(spec),
        **{'Partizioni autorizzate':partitions,'Artefatto preregistrazione':[arts[0]],'Hash preregistrazione':hashlib.sha256((repo/j.CONFIG).read_bytes()).hexdigest(),
           'Criteri di successo':spec['quality'],'Regole di arresto':spec['preflights'],'Registrato il':'2026-10-20'})
    impl=put('implementations','worker',Nome='Task39j atomic worker',**{'Repository URL':'https://github.com/Zagred47/giada',
        'Percorso entrypoint':'scripts/run_task39j_atomic_relaxation.py','Artefatti codice':arts,'Ambiente':'Colab console CUDA;external supervisor'})
    refs={}
    for a in j.arms():
        model=put('models',a['name'],Nome='Task39j '+a['name'],Ruolo='Surrogate',Descrizione=spec['architecture']+' '+json.dumps(a))
        refs[a['name']]=put('arms',a['name'],Nome='Task39j '+a['name'],Ruolo='Trattamento',Protocollo=[protocol],Modello=[model],Implementazione=[impl],Partizioni=partitions,
            **{'Configurazione residua':json.dumps(a)})
        for seed in spec['seeds']:put('runs',f'{a["name"]}-{seed}',Nome=f'Task39j {a["name"]} seed{seed}',Stato='Pianificata',Seed=str(seed),Braccio=[refs[a['name']]],
            **{'Configurazione effettiva':json.dumps(dict(arm=a,seed=seed,checkpoints=spec['checkpoints']))})
    for axis,a,b in j.contrasts():put('contrasts',a+'-'+b,Nome='Task39j '+axis,Protocollo=[protocol],Bracci=[refs[a],refs[b]],
        **{'Contrasto e coefficienti':'Treatment versus control paired byseed;report currentseverity,timing andnewregressions separately.',
           'Soglia interpretativa':'Existing medianratio<=.8,all3<1,worst<=1.05;full72/72andauditforpromotion.',
           'Confondenti controllati':spec['optimization']})
    for name,key in [('quality','quality'),('selection','selection'),('atomic','objective'),('cost','optimization')]:
        metric=put('metrics',name,Nome='Task39j '+name,Formula=spec[key])
        put('evaluations',name,Nome='Task39j '+name,Versione='v1',Metrica=[metric],Protocollo=[protocol],Partizioni=partitions,Ruolo='Diagnostica',Target=spec[key])
    put('actions','execute',Nome='Run39j afterGPUpreflight',Stato='Proposta',Descrizione='No launch claimed by preregistration; external backup verification is separate.')
    print(json.dumps(dict(stage='staged',records=len(ids))),flush=True)
    m.commit_batch();write_json(ROOT/'data/airtable_snapshot.json',m.export_snapshot());v=m.verify();assert v['valid']
    folder=repo/'experiments/results/task39j_protocol_v1';folder.mkdir(exist_ok=True)
    write_json(folder/'memory_receipt.json',dict(valid=True,records=ids,verification=v,airtable_accessed=False,launched=False))
    print(json.dumps(dict(valid=True,records=len(ids))),flush=True)

if __name__=='__main__':main()
