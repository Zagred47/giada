"""Task39b preregistration or verified dispatch, exclusive local mirror."""
import argparse
import hashlib
import json
from .mirror import ROOT,write_json
from .validated_batch import ValidatedBatchMirror


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--launch',action='store_true');args=parser.parse_args()
    repo=ROOT.parent;specpath=repo/'experiments/task39b_preregistration_v1.json';spec=json.loads(specpath.read_text())
    folder=repo/'experiments/results/task39b_protocol_v1';folder.mkdir(parents=True,exist_ok=True)
    m=ValidatedBatchMirror();ids={}
    def put(table,suffix,**fields):
        code=f'{table}-giada-task39b-architecture-{suffix}-v1'
        rid=m.local_upsert(table,{'Codice stabile':code,**fields})['record_id'];ids[code]=rid;return rid
    def ref(code):
        rows=m.query("SELECT record_id FROM v_records WHERE stable_code='"+code+"'")['rows'];assert len(rows)==1,code
        return rows[0]['record_id']
    from src.giada_teacher.task39b_architecture_diagnosis import arms
    defs=arms()
    if args.launch:
        path=folder/'kaggle_launch.json';launch=json.loads(path.read_text())
        assert launch['status_at_verification'] in ('RUNNING','QUEUED') and launch['published_source_pin_verified']
        art=put('artifacts','launch',Nome='Task39b verified Kaggle dispatch',Tipo='Log',Versione='v1',
            Percorso=path.relative_to(repo).as_posix(),**{'SHA-256':hashlib.sha256(path.read_bytes()).hexdigest(),'Dimensione byte':path.stat().st_size})
        for d in defs:
            put('runs',d['name'],Nome='Task39b running '+d['name'],Stato='In corso',
                **{'Artefatti prodotti':[art],'Configurazione effettiva':json.dumps(dict(arm=d,seeds=spec['seeds'],launch=launch)),
                   'Validità tecnica':'Dispatch verified, GPU preflights and scientific results pending.'})
        put('implementations','worker',Nome='Task39b36-member CUDA graph worker',Commit=launch['revision'])
        # Refresh final source hashes after technical preflight refinements.
        for rel in ['src/giada_teacher/task39b_architecture_diagnosis.py','research_memory/register_task39b.py',
                    'tests/test_task39b_architecture_diagnosis.py']:
            path=repo/rel
            put('artifacts',rel,Nome='Task39b '+path.name,Percorso=rel,Versione='v1',Tipo='Codice',
                **{'SHA-256':hashlib.sha256(path.read_bytes()).hexdigest(),'Dimensione byte':path.stat().st_size})
    else:
        paths=['experiments/task39b_preregistration_v1.json','experiments/task39b_design_v1.md',
            'src/giada_teacher/task39b_architecture_diagnosis.py','scripts/run_task39b_architecture_diagnosis.py',
            'scripts/build_task39b_notebook.py','notebooks/giada_task39b_architecture_diagnosis.ipynb',
            'tests/test_task39b_architecture_diagnosis.py','research_memory/register_task39b.py']
        arts=[]
        for rel in paths:
            path=repo/rel
            arts.append(put('artifacts',rel,Nome='Task39b '+path.name,Tipo='Codice' if path.suffix=='.py' else 'Report',
                Versione='v1',Percorso=rel,**{'SHA-256':hashlib.sha256(path.read_bytes()).hexdigest(),'Dimensione byte':path.stat().st_size}))
        hypotheses={
            'horizon':'64-tick recurrent training improves long behavior relative to8-tick resets at equal target exposure.',
            'collocation':'Replacing25percent of training windows with physical-state oracle collocation improves dynamic field identifiability.',
            'relative':'Relative individual current auxiliary improves weak-current accuracy without worst-case regression.',
            'grouped':'Shared+rapid/slow heads reduce gate interference without additional parameters or lower abstraction.',
            'multiscale':'Fixed multiscale fraction activation improves optimization of small relaxation fractions at same parameter count.',
            'interaction':'Architecture benefits may differ between minimal and complete training interventions; not assumed additive.'}
        claims=[put('claims',name,Nome='Task39b '+name,Tipo='Ipotesi',Stato='Aperta',Enunciato=statement,**{'Limiti':spec['limits']}) for name,statement in hypotheses.items()]
        exp=put('experiments','suite',Nome='Task39b training and architecture diagnosis',Stato='Preregistrato',Ipotesi=claims,
            Descrizione=spec['matrix'],**{'Obiettivo informativo':spec['probes']})
        partitions=[ref(f'partitions-giada-task37-ionic-playground-{role}-v1') for role in ('fit','fit_extra','development','stress_diagnostic')]
        protocol=put('protocols','suite',Nome='Task39b fixed12-arm protocol',Versione='v1',Modalità='Ricomposizione',
            Esperimento=[exp],Procedura=json.dumps(spec,ensure_ascii=False),**{'Partizioni autorizzate':partitions,
            'Artefatto preregistrazione':[arts[0]],'Hash preregistrazione':hashlib.sha256(specpath.read_bytes()).hexdigest(),
            'Criteri di successo':spec['quality'],'Regole di arresto':spec['preflights'],'Registrato il':'2026-10-18'})
        impl=put('implementations','worker',Nome='Task39b36-member CUDA graph worker',**{'Repository URL':'https://github.com/Zagred47/giada',
            'Percorso entrypoint':'scripts/run_task39b_architecture_diagnosis.py','Artefatti codice':arts,
            'Ambiente':'Kaggle pinned Task39 image, T4; actual dispatch revision stored separately.'})
        armids={}
        for d in defs:
            model=put('models',d['name'],Nome='Task39b '+d['name'],Ruolo='Surrogate',Descrizione=spec['architecture']+' '+json.dumps(d))
            armids[d['name']]=put('arms',d['name'],Nome='Task39b '+d['name'],Ruolo='Trattamento',Protocollo=[protocol],
                Modello=[model],Implementazione=[impl],Partizioni=partitions,**{'Configurazione residua':json.dumps(dict(arm=d,seeds=spec['seeds'],checkpoints=spec['checkpoints']))})
            put('runs',d['name'],Nome='Task39b planned '+d['name'],Stato='Pianificata',Braccio=[armids[d['name']]],
                **{'Configurazione effettiva':json.dumps(d),'Hardware e ambiente':'Prospective T4, no result claimed.'})
        pairs=[]
        for axis,low,high in [('horizon',8,64),('collocation',0,1),('relative',0,1)]:
            for a in defs[:8]:
                if a[axis]!=low:continue
                b=next(b for b in defs[:8] if b[axis]==high and all(a[k]==b[k] for k in ('horizon','collocation','relative') if k!=axis))
                pairs.append((axis,a,b))
        for b in defs[8:]:
            a=next(a for a in defs[:8] if all(a[k]==b[k] for k in ('horizon','collocation','relative')))
            pairs.append(('architecture',a,b))
        for axis,a,b in pairs:
            put('contrasts',a['name']+'-vs-'+b['name'],Nome='Task39b '+axis+': '+a['name']+' vs '+b['name'],
                Protocollo=[protocol],Bracci=[armids[a['name']],armids[b['name']]],**{'Contrasto e coefficienti':spec['contrasts'],
                'Soglia interpretativa':'Median paired ratio<=.8, all3<1,worst<=1.05; unchanged strict quality.',
                'Confondenti controllati':'Common paired streams, target counts, fit-only normalizers, independent Adam. Architecture mask/activation and effective capacity explicitly disclosed.'})
        for name,formula in [('strict_quality',spec['quality']),('selection',spec['selection']),('probes',spec['probes'])]:
            metric=put('metrics',name,Nome='Task39b '+name,Formula=formula)
            put('evaluations',name,Nome='Task39b '+name,Versione='v1',Metrica=[metric],Protocollo=[protocol],Partizioni=partitions[2:],
                Ruolo='Diagnostica',Target=spec['quality'],**{'Aggregazione e pesi':'Family/channel/seed strata; no event removed, no independent confirmation.'})
        put('decisions','authorize-diagnostic',Nome='Authorize39b, not Task40',Esito='Modificare',
            Risultati=[ref('findings-giada-task39-learnability-outcome-v1')],Motivazione=spec['matrix']+' '+spec['limits'],**{'Data':'2026-10-18'})
    print(json.dumps(dict(stage='ready',records=len(ids),launch=args.launch)),flush=True)
    m.commit_batch();write_json(ROOT/'data/airtable_snapshot.json',m.export_snapshot());verification=m.verify();assert verification['valid']
    write_json(folder/('launch_memory_receipt.json' if args.launch else 'memory_receipt.json'),dict(valid=True,records=ids,
        verification=verification,airtable_accessed=False,scientific_result_claimed=False))
    print(json.dumps(dict(valid=True,records=len(ids),launch=args.launch)),flush=True)

if __name__=='__main__':main()
