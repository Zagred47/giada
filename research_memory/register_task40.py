"""Prospective frozen Task40 protocol; only local SQLite, no result invented."""
import hashlib,json
from .mirror import ROOT,write_json
from .validated_batch import ValidatedBatchMirror
from src.giada_teacher import task40_causal_reinsertion as t

def main():
    repo=ROOT.parent;spec=json.loads((repo/t.CONFIG).read_text());m=ValidatedBatchMirror();ids={}
    def put(table,suffix,**fields):
        code=f'{table}-giada-task40-causal-{suffix}-v1'
        rid=m.local_upsert(table,{'Codice stabile':code,**fields})['record_id'];ids[code]=rid;return rid
    def ref(code):
        rows=m.query("SELECT record_id FROM v_records WHERE stable_code='"+code+"'")['rows'];assert len(rows)==1,code
        return rows[0]['record_id']
    artifacts=[]
    for rel in [t.CONFIG,'src/giada_teacher/task40_causal_reinsertion.py','scripts/build_task40_protocol.py',
                'scripts/run_task40_causal_reinsertion.py','scripts/probe_task40_gpu.py','tests/test_task40_causal_reinsertion.py',
                'notebooks/giada_task40_causal_reinsertion.ipynb','experiments/results/task40_protocol_v1/gpu_preflight.json']:
        path=repo/rel
        artifacts.append(put('artifacts',rel,Nome='Task40 '+path.name,Tipo='Codice' if path.suffix=='.py' else 'Report',Versione='v1',Percorso=rel,
            **{'SHA-256':hashlib.sha256(path.read_bytes()).hexdigest(),'Dimensione byte':path.stat().st_size}))
    claims=[put('claims',str(k+1),Nome=h,Tipo='Ipotesi',Stato='Aperta',Enunciato=h,
                Limiti='Frozen diagnostic. Not independent confirmation. Known stress failures not waived.') for k,h in enumerate(spec['hypotheses'])]
    exp=put('experiments','suite',Nome='Task40 frozen causal reinsertion',Stato='Preregistrato',Ipotesi=claims,
            Descrizione=spec['design'],**{'Obiettivo informativo':'Distinguish isolated accuracy, causal feedback, channel attribution, calcium phase and execution cost.'})
    partitions=[ref('partitions-giada-task37-ionic-playground-stress_diagnostic-v1')]
    dataset=put('datasets','native',Nome='Task40 opened native fixtures and paired controls',Versione='v1',Artefatti=artifacts[:1],
        Schema='Task35 native shadow trajectories plus explicit paired interventions',Copertura=spec['design'],**{'Numero esempi':88},
        Fingerprint=spec['sources']['experiments/results/task35_native_shadow_traces.npz'])
    partitions.append(put('partitions','opened',Nome='Task40 diagnostic opened historical replay',Dataset=[dataset],Ruolo='Diagnostico',
        **{'Regola di assegnazione':spec['design'],'Restrizioni di accesso':'No training or selection. Historical paths already opened, not independent.'}))
    protocol=put('protocols','suite',Nome='Task40 frozen paired causal protocol',Versione='v1',Modalità='Ricomposizione',Esperimento=[exp],Procedura=json.dumps(spec),
        **{'Partizioni autorizzate':partitions,'Artefatto preregistrazione':[artifacts[0]],'Hash preregistrazione':t.p.digest(repo/t.CONFIG),
           'Criteri di successo':spec['decisions'],'Regole di arresto':'Technical floor mismatch stops model judgment; scientific failure is reported separately.','Registrato il':'2026-10-21'})
    impl=put('implementations','worker',Nome='Task40 frozen vectorized CUDA graphs',**{'Repository URL':'https://github.com/Zagred47/giada',
        'Percorso entrypoint':'scripts/run_task40_causal_reinsertion.py','Artefatti codice':artifacts,'Ambiente':'Colab console Tesla T4; external verified ZIP supervisor'})
    arms={}
    for name in ('I3','J1'):
        model=put('models',name,Nome='Frozen '+name,Ruolo='Surrogate',Descrizione='Unchanged frozen checkpoint; all 18 states; analytical cheap current readout; J1 audited FP64 correction retained.')
        for family in t.FAMILIES:
            for mode in ('imposed','autonomous'):
                key=f'{name}-{family}-{mode}'
                arm=put('arms',key,Nome='Task40 '+key,Ruolo='Trattamento' if name=='J1' else 'Baseline',Protocollo=[protocol],Modello=[model],Implementazione=[impl],Partizioni=partitions,
                        **{'Configurazione residua':json.dumps(dict(model=name,family=family,mode=mode,training=False))})
                arms[key]=arm
                for seed in spec['seeds']:
                    put('runs',key+'-'+str(seed),Nome=f'Task40 {key} seed{seed}',Stato='Pianificata',Seed=str(seed),Braccio=[arm],
                        **{'Configurazione effettiva':json.dumps(dict(seed=seed,model=name,family=family,mode=mode,cases=88))})
    for mode in ('imposed','autonomous'):
        put('contrasts',mode,Nome='J1 versus I3 '+mode,Protocollo=[protocol],Bracci=[arms[f'{n}-all-{mode}'] for n in ('I3','J1')],
            **{'Contrasto e coefficienti':'Paired J1-I3, per case/channel/seed, no pooled waiver.','Soglia interpretativa':spec['decisions'],'Confondenti controllati':spec['phase']})
    for key,target in [('quality',spec['current_quality']),('causal',spec['compartment_gates']),('stress',spec['known_stress']),('cost',spec['cost'])]:
        metric=put('metrics',key,Nome='Task40 '+key,Formula=json.dumps(target))
        put('evaluations',key,Nome='Task40 '+key,Versione='v1',Metrica=[metric],Protocollo=[protocol],Partizioni=partitions,Ruolo='Diagnostica',Target=json.dumps(target))
    decision=ref('decisions-giada-task39j-result-prepare40-v1')
    put('actions','execute',Nome='Task40 execute after GPU preflight',Stato='Proposta',Descrizione='No launch claimed. External backups verified independently.',**{'Decisioni di origine':[decision]})
    print(json.dumps(dict(stage='staged',records=len(ids))),flush=True)
    m.commit_batch();write_json(ROOT/'data/airtable_snapshot.json',m.export_snapshot());verification=m.verify();assert verification['valid']
    write_json(repo/'experiments/results/task40_protocol_v1/memory_receipt.json',dict(valid=True,records=ids,verification=verification,launched=False,airtable_accessed=False))
    print('Task40 preregistration verified',flush=True)

if __name__=='__main__':main()
