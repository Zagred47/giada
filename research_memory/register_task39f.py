"""Preregister39f or record its actual console dispatch in SQLite only."""
import argparse,hashlib,json
from functools import lru_cache
from .mirror import ROOT,write_json
from .validated_batch import ValidatedBatchMirror
from src.giada_teacher.task39f_temporal_recovery import arms,contrast_pairs,CONFIG

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--launch',action='store_true');args=parser.parse_args()
    repo=ROOT.parent;folder=repo/'experiments/results/task39f_protocol_v1';spec=json.loads((repo/CONFIG).read_text())
    m=ValidatedBatchMirror();ids={}
    def put(table,suffix,**fields):
        code=f'{table}-giada-task39f-temporal-{suffix}-v1';rid=m.local_upsert(table,{'Codice stabile':code,**fields})['record_id'];ids[code]=rid;return rid
    @lru_cache(None)
    def ref(code):
        r=m.query("SELECT record_id FROM v_records WHERE stable_code='"+code+"'")['rows'];assert len(r)==1,code;return r[0]['record_id']
    if args.launch:
        launch=json.loads((folder/'launch.json').read_text());backup=json.loads((folder/'external_recovery_verified.json').read_text());assert backup['valid']
        art=[]
        paths=[folder/n for n in ('launch.json','external_recovery_verified.json','dispatch_request.json','local_validation.json')]
        paths += [repo/rel for rel in ('src/giada_teacher/task39f_temporal_recovery.py','scripts/colab_profile_cli.py','scripts/colab.ps1','scripts/colab_checked_exec.py','scripts/colab_checked_download.py','scripts/launch_task39f_console.ps1','scripts/record_task39f_dispatch.py','docs/colab_console_operations.md','research_memory/register_task39f.py')]
        for path in paths:
            rel=path.relative_to(repo).as_posix()
            art.append(put('artifacts','dispatch-'+rel,Nome='Task39f dispatch '+path.name,Tipo='Codice' if path.suffix in ('.py','.ps1') else 'Report',Percorso=rel,Versione='reserve dispatch v1',**{'SHA-256':hashlib.sha256(path.read_bytes()).hexdigest(),'Dimensione byte':path.stat().st_size}))
        for a in arms():
            for seed in spec['seeds']:
                put('runs',f'{a["name"]}-{seed}',Nome=f'Task39f {a["name"]} seed{seed}',Stato='In corso',
                    **{'Configurazione effettiva':json.dumps(dict(arm=a,seed=seed,launch=launch)),'Validità tecnica':'Console launched; externally verified recovery. Scientific result pending.','Artefatti prodotti':art})
        put('actions','execute',Nome='Execute Task39f on Colab console',Stato='In corso',Descrizione=json.dumps(launch))
        put('experiments','suite',Nome='Task39f temporal recovery/support/capacity',Stato='In corso')
        put('implementations','worker',Nome='Task39f CUDA console worker',Commit=launch['revision'],**{'Ambiente':json.dumps(dict(profile=launch['account_profile'],account=launch['account'],gpu=launch['gpu'],session=launch['session']))})
    else:
        assert json.loads((folder/'local_validation.json').read_text())['valid']
        paths=[CONFIG,'src/giada_teacher/task39f_temporal_recovery.py','src/giada_teacher/task39e_tail_support_heads.py','src/giada_teacher/task39d_current_timing.py','scripts/run_task39f_temporal_recovery.py','scripts/build_task39f_protocol.py','scripts/validate_task39f.py','tests/test_task39f_temporal_recovery.py','notebooks/giada_task39f_temporal_recovery.ipynb','scripts/colab_task39f_driver.py','scripts/colab_task39e_driver.py','scripts/task39e_console_manager.py','scripts/task39e_local_supervisor.py','experiments/giada_modular_surrogate_roadmap.md','experiments/results/task39f_protocol_v1/local_validation.json','research_memory/register_task39f.py']
        arts=[]
        for rel in paths:
            p=repo/rel;arts.append(put('artifacts',rel,Nome='Task39f '+p.name,Tipo='Codice' if p.suffix=='.py' else 'Report',Versione='v1',Percorso=rel,**{'SHA-256':hashlib.sha256(p.read_bytes()).hexdigest(),'Dimensione byte':p.stat().st_size}))
        claims=[ref('claims-giada-task39e-result-residual-'+n+'-v1') for n in ('slow_tail_crossing','equilibrium_vs_rate','transition_support','capacity_or_budget','metric_or_precision')]
        exp=put('experiments','suite',Nome='Task39f temporal recovery/support/capacity',Stato='Preregistrato',Ipotesi=claims,Descrizione=spec['matrix'],**{'Obiettivo informativo':spec['oracles']+' '+spec['contrasts']})
        partitions=[ref(f'partitions-giada-task37-ionic-playground-{r}-v1') for r in ('fit','fit_extra','development','stress_diagnostic')]
        protocol=put('protocols','suite',Nome='Task39f paired24-model protocol',Versione='v1',Modalità='Ricomposizione',Esperimento=[exp],Procedura=json.dumps(spec),
            **{'Partizioni autorizzate':partitions,'Artefatto preregistrazione':[arts[0]],'Hash preregistrazione':hashlib.sha256((repo/CONFIG).read_bytes()).hexdigest(),'Criteri di successo':spec['quality'],'Regole di arresto':spec['preflights'],'Registrato il':'2026-10-19'})
        impl=put('implementations','worker',Nome='Task39f CUDA console worker',**{'Repository URL':'https://github.com/Zagred47/giada','Percorso entrypoint':'scripts/run_task39f_temporal_recovery.py','Artefatti codice':arts,'Ambiente':'Official Colab CLI0.7.4/WSL2;24 members,2 CUDA cohorts; external PC supervisor'})
        aa={}
        for a in arms():
            model=put('models',a['name'],Nome='Task39f '+a['name'],Ruolo='Surrogate',Descrizione=spec['architecture']+' '+json.dumps(a))
            aa[a['name']]=put('arms',a['name'],Nome='Task39f '+a['name'],Ruolo='Trattamento',Protocollo=[protocol],Modello=[model],Implementazione=[impl],Partizioni=partitions,**{'Configurazione residua':json.dumps(a)})
            for seed in spec['seeds']:
                put('runs',f'{a["name"]}-{seed}',Nome=f'Task39f {a["name"]} seed{seed}',Stato='Pianificata',Seed=str(seed),Braccio=[aa[a['name']]],**{'Configurazione effettiva':json.dumps(dict(arm=a,seed=seed,checkpoints=spec['checkpoints']))})
        for axis,a,z in contrast_pairs(arms()):
            put('contrasts',a+'-vs-'+z,Nome='Task39f '+axis+': '+a+' vs '+z,Protocollo=[protocol],Bracci=[aa[a],aa[z]],**{'Contrasto e coefficienti':spec['contrasts'],'Soglia interpretativa':'Median<=.8,all3<1,worst<=1.05; unchanged quality','Confondenti controllati':spec['training']})
        for n in ('quality','selection','oracles','cost'):
            metric=put('metrics',n,Nome='Task39f '+n,Formula=spec[n]);put('evaluations',n,Nome='Task39f '+n,Versione='v1',Metrica=[metric],Protocollo=[protocol],Partizioni=partitions[2:],Ruolo='Diagnostica',Target=spec[n])
        decision=put('decisions','authorize',Nome='Authorize39f, keep Task40 blocked',Esito='Modificare',Risultati=[ref('findings-giada-task39e-result-outcome-v1')],Motivazione=spec['matrix']+' '+spec['limits'],**{'Data':'2026-10-19'})
        put('actions','execute',Nome='Execute Task39f on Colab console',Stato='Proposta',Descrizione=spec['matrix'],**{'Decisioni di origine':[decision]})
        m.local_upsert('actions',{'Codice stabile':'actions-giada-task39e-result-proposed39f-v1','Stato':'Conclusa','Descrizione':'User authorized39f preparation, preregistration, push and console dispatch. Actual dispatch tracked separately.'})
    print(json.dumps(dict(stage='staged',records=len(ids),launch=args.launch)),flush=True)
    m.commit_batch();write_json(ROOT/'data/airtable_snapshot.json',m.export_snapshot());v=m.verify();assert v['valid']
    write_json(folder/('launch_memory_receipt.json' if args.launch else 'memory_receipt.json'),dict(valid=True,records=ids,verification=v,airtable_accessed=False,drive_backup_updated=False,scientific_result_claimed=False))
    print(json.dumps(dict(valid=True,records=len(ids))),flush=True)
if __name__=='__main__':main()
