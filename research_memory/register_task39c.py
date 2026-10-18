"""Preregister Task39c and record verified dispatch in authoritative SQLite."""
import argparse
import hashlib
import json
from functools import lru_cache
from .mirror import ROOT,write_json
from .validated_batch import ValidatedBatchMirror


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--launch',action='store_true')
    parser.add_argument('--code-only',action='store_true');args=parser.parse_args()
    repo=ROOT.parent;specpath=repo/'experiments/task39c_preregistration_v1.json';spec=json.loads(specpath.read_text())
    folder=repo/'experiments/results/task39c_protocol_v1';folder.mkdir(parents=True,exist_ok=True)
    m=ValidatedBatchMirror();ids={}
    def put(table,suffix,**fields):
        code=f'{table}-giada-task39c-dynamic-{suffix}-v1'
        rid=m.local_upsert(table,{'Codice stabile':code,**fields})['record_id'];ids[code]=rid;return rid
    @lru_cache(maxsize=None)
    def ref(code):
        rows=m.query("SELECT record_id FROM v_records WHERE stable_code='"+code+"'")['rows']
        assert len(rows)==1,code
        return rows[0]['record_id']
    from src.giada_teacher.task39c_dynamic_identification import arms,contrast_pairs
    defs=arms()
    paths=['experiments/task39c_preregistration_v1.json','src/giada_teacher/task39c_dynamic_identification.py',
        'scripts/run_task39c_dynamic_identification.py','scripts/build_task39c_notebook.py',
        'notebooks/giada_task39c_dynamic_identification.ipynb','tests/test_task39c_dynamic_identification.py',
        'research_memory/register_task39c.py','experiments/results/task39c_protocol_v1/local_validation.json']
    arts=[]
    for rel in paths:
        path=repo/rel
        arts.append(put('artifacts',rel,Nome='Task39c '+path.name,Tipo='Codice' if path.suffix=='.py' else 'Report',
            Versione='v1',Percorso=rel,**{'SHA-256':hashlib.sha256(path.read_bytes()).hexdigest(),'Dimensione byte':path.stat().st_size}))
    if args.launch:
        path=folder/'kaggle_launch.json';launch=json.loads(path.read_text())
        assert launch['status_at_verification'] in ('RUNNING','QUEUED') and launch['published_source_pin_verified']
        artifact=put('artifacts','launch',Nome='Task39c verified Kaggle dispatch',Tipo='Log',Versione='v1',
            Percorso=path.relative_to(repo).as_posix(),**{'SHA-256':hashlib.sha256(path.read_bytes()).hexdigest(),'Dimensione byte':path.stat().st_size})
        for d in defs:
            for seed in spec['seeds']:
                put('runs',f'{d["name"]}-seed{seed}',Nome=f'Task39c {d["name"]} seed{seed}',Stato='In corso',
                    **{'Artefatti prodotti':[artifact],'Configurazione effettiva':json.dumps(dict(arm=d,seed=seed,launch=launch)),
                       'Validità tecnica':'Kaggle dispatch/source verified; CUDA preflights and scientific outcome pending.'})
        put('implementations','worker',Nome='Task39c36-member CUDA worker',Commit=launch['revision'])
        put('actions','execute',Nome='Execute Task39c on Kaggle',Stato='In corso')
        m.local_upsert('actions',{'Codice stabile':'actions-giada-task39b-architecture-proposed39c-v1','Stato':'Conclusa',
            'Descrizione':'Task39c prepared, preregistered, pushed and dispatched; execution tracked separately.'})
    elif not args.code_only:
        claims=[ref('claims-giada-task39b-architecture-residual-'+name+'-v1') for name in
                ('dynamic_information','optimization_budget','local_support','shared_capacity','loss_conditioning','numerical_timing')]
        exp=put('experiments','suite',Nome='Task39c coefficient information, support and temporal identification',
            Stato='Preregistrato',Ipotesi=claims,Descrizione=spec['matrix'],**{'Obiettivo informativo':spec['contrasts']+' '+spec['oracles']})
        partitions=[ref(f'partitions-giada-task37-ionic-playground-{r}-v1') for r in ('fit','fit_extra','development','stress_diagnostic')]
        protocol=put('protocols','suite',Nome='Task39c12-arm prospective protocol',Versione='v1',Modalità='Ricomposizione',
            Esperimento=[exp],Procedura=json.dumps(spec,ensure_ascii=False),**{'Partizioni autorizzate':partitions,
            'Artefatto preregistrazione':[arts[0]],'Hash preregistrazione':hashlib.sha256(specpath.read_bytes()).hexdigest(),
            'Criteri di successo':spec['quality'],'Regole di arresto':spec['preflights']+' '+spec['oracles'],'Registrato il':'2026-10-18'})
        impl=put('implementations','worker',Nome='Task39c36-member CUDA worker',
            **{'Repository URL':'https://github.com/Zagred47/giada','Percorso entrypoint':'scripts/run_task39c_dynamic_identification.py',
               'Artefatti codice':arts,'Ambiente':'Kaggle T4; three architecture cohorts, member-vectorized Adam, actual source pin in dispatch record.'})
        armids={}
        for d in defs:
            model=put('models',d['name'],Nome='Task39c '+d['name'],Ruolo='Surrogate',Descrizione=spec['architecture']+' '+json.dumps(d))
            armids[d['name']]=put('arms',d['name'],Nome='Task39c '+d['name'],Ruolo='Trattamento',Protocollo=[protocol],
                Modello=[model],Implementazione=[impl],Partizioni=partitions,**{'Configurazione residua':json.dumps(dict(arm=d,seeds=spec['seeds']))})
            for seed in spec['seeds']:
                put('runs',f'{d["name"]}-seed{seed}',Nome=f'Task39c {d["name"]} seed{seed}',Stato='Pianificata',Braccio=[armids[d['name']]],
                    **{'Configurazione effettiva':json.dumps(dict(arm=d,seed=seed,checkpoints=spec['checkpoints'])),
                       'Hardware e ambiente':'Prospective Kaggle T4; no result claimed.'})
        for axis,a,c in contrast_pairs(defs):
            put('contrasts',a+'-vs-'+c,Nome='Task39c '+axis+': '+a+' vs '+c,Protocollo=[protocol],
                Bracci=[armids[a],armids[c]],**{'Contrasto e coefficienti':spec['contrasts'],
                'Soglia interpretativa':'Median paired ratio<=.8, all3<1, worst<=1.05; all strict quality unchanged.',
                'Confondenti controllati':spec['training']})
        for name,description in [('quality',spec['quality']),('selection',spec['selection']),('oracles',spec['oracles']),('cost',spec['cost'])]:
            metric=put('metrics',name,Nome='Task39c '+name,Formula=description)
            put('evaluations',name,Nome='Task39c '+name,Versione='v1',Metrica=[metric],Protocollo=[protocol],Partizioni=partitions[2:],
                Ruolo='Diagnostica',Target=description,**{'Aggregazione e pesi':'Individual gates/channels/families, paired seeds; no independent confirmation.'})
        decision=put('decisions','authorize',Nome='Authorize Task39c causal diagnosis',Esito='Modificare',
            Risultati=[ref('findings-giada-task39b-architecture-outcome-v1')],Motivazione=spec['matrix']+' '+spec['limits'],**{'Data':'2026-10-18'})
        put('actions','execute',Nome='Execute Task39c on Kaggle',Stato='Proposta',Descrizione=spec['matrix'],**{'Decisioni di origine':[decision]})
        # Update the earlier proposal without inventing a scientific outcome.
        m.local_upsert('actions',{'Codice stabile':'actions-giada-task39b-architecture-proposed39c-v1','Stato':'Proposta',
            'Descrizione':'User authorized implementation and Kaggle run; concrete protocol experiments/task39c_preregistration_v1.json.'})
    print(json.dumps(dict(stage='staged',records=len(ids),launch=args.launch)),flush=True)
    m.commit_batch();write_json(ROOT/'data/airtable_snapshot.json',m.export_snapshot());v=m.verify();assert v['valid']
    write_json(folder/('launch_memory_receipt.json' if args.launch else 'code_memory_receipt.json' if args.code_only else 'memory_receipt.json'),
        dict(valid=True,records=ids,verification=v,airtable_accessed=False,scientific_result_claimed=False))
    print(json.dumps(dict(valid=True,records=len(ids),launch=args.launch)),flush=True)


if __name__=='__main__':main()
