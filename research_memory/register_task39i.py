"""Task39i protocol and actual launch in the exclusive local SQLite mirror."""
import argparse,hashlib,json
from functools import lru_cache
from .mirror import ROOT,write_json
from .validated_batch import ValidatedBatchMirror
from src.giada_teacher.task39i_coupled_relaxation import arms,contrast_pairs,CONFIG

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--launch',action='store_true');args=parser.parse_args()
    repo=ROOT.parent;folder=repo/'experiments/results/task39i_protocol_v1';spec=json.loads((repo/CONFIG).read_text())
    m=ValidatedBatchMirror();ids={}
    def put(table,suffix,**values):
        code=f'{table}-giada-task39i-temporal-{suffix}-v1'
        rid=m.local_upsert(table,{'Codice stabile':code,**values})['record_id'];ids[code]=rid;return rid
    @lru_cache(None)
    def ref(code):
        rows=m.query("SELECT record_id FROM v_records WHERE stable_code='"+code+"'")['rows'];assert len(rows)==1,code;return rows[0]['record_id']
    if not args.launch:
        assert json.loads((folder/'local_validation.json').read_text())['valid']
        paths=[CONFIG,'src/giada_teacher/task39i_coupled_relaxation.py','scripts/build_task39i_protocol.py','scripts/validate_task39i.py',
            'scripts/run_task39i_coupled_relaxation.py','scripts/launch_task39i_console.ps1','scripts/prepare_task39i_dispatch.py',
            'scripts/record_task39i_dispatch.py','scripts/colab_task39i_driver.py','scripts/probe_task39i_gpu.py','scripts/probe_task39i_driver.py',
            'scripts/task39e_console_manager.py','scripts/task39e_local_supervisor.py','tests/test_task39i_coupled_relaxation.py',
            'notebooks/giada_task39i_coupled_relaxation.ipynb','experiments/results/task39i_protocol_v1/local_validation.json',
            'experiments/results/task39i_protocol_v1/sk_transfer_cpu.json','scripts/probe_task39i_support.py',
            'experiments/results/task39i_protocol_v1/support_preflight.json','experiments/giada_modular_surrogate_roadmap.md','research_memory/register_task39i.py']
        arts=[]
        for rel in paths:
            path=repo/rel;arts.append(put('artifacts',rel,Nome='Task39i '+path.name,Tipo='Codice' if path.suffix=='.py' else 'Report',Versione='v1',Percorso=rel,
                **{'SHA-256':hashlib.sha256(path.read_bytes()).hexdigest(),'Dimensione byte':path.stat().st_size}))
        proposal=json.loads((repo/'experiments/task39h_residual_hypotheses.json').read_text())
        claims=[ref('claims-giada-task39h-result-residual-'+x['id']+'-v1') for x in proposal['hypotheses']]
        experiment=put('experiments','suite',Nome='Task39i coupled relaxation and learned SK shape',Stato='Preregistrato',Ipotesi=claims,
            Descrizione=spec['matrix'],**{'Obiettivo informativo':spec['contrasts']+' '+spec['diagnostics']})
        partitions=[ref(f'partitions-giada-task37-ionic-playground-{r}-v1') for r in ('fit','fit_extra','development','stress_diagnostic')]
        generator=put('generators','fit',Nome='Task39i independent held and history FIT',Versione='v1',Procedura=spec['expanded_support'],
            **{'Seed e RNG':json.dumps(dict(constant=spec['expanded_seeds'],history=spec['history_seeds'])),'Artefatti codice':arts,
            'Variabili registrate':'V,Ca,18gates,11currents,multipliers,reversals,offline logdecay labels,signatures and assignments'})
        dataset=put('datasets','fit',Nome='Task39i planned independent history support',Versione='v1',Generatore=[generator],Artefatti=[arts[0]],
            Fingerprint='recipe-sha256:'+hashlib.sha256(spec['expanded_support'].encode()).hexdigest(),Schema='Task37 NPZ;89field FIT windows with18offline decay labels;107field anchors',
            Copertura=spec['expanded_support'],**{'Numero esempi':144,'Descrizione':'72regenerated39h constant histories plus72new variable histories; actual hashes recorded onruntime before scientific fit.'})
        extra=[]
        for kind,seeds in [('constant',spec['expanded_seeds']),('history',spec['history_seeds'])]:
            for role,seed in seeds.items():
                extra.append(put('partitions',kind+'-'+role,Nome='Task39i '+kind+' '+role,Ruolo='Fit',Dataset=[dataset],
                    **{'Regola di assegnazione':json.dumps(dict(kind=kind,role=role,seed=seed)),
                    'Restrizioni di accesso':'Independent FIT generation only; no evaluation path copying or candidate teacher coefficients at inference.'}))
        protocol=put('protocols','suite',Nome='Task39i36model paired protocol',Versione='v1',Modalità='Ricomposizione',Esperimento=[experiment],Procedura=json.dumps(spec),
            **{'Partizioni autorizzate':partitions+extra,'Artefatto preregistrazione':[arts[0]],'Hash preregistrazione':hashlib.sha256((repo/CONFIG).read_bytes()).hexdigest(),
            'Criteri di successo':spec['quality'],'Regole di arresto':spec['preflights'],'Registrato il':'2026-10-19'})
        impl=put('implementations','worker',Nome='Task39i CUDA console worker',**{'Repository URL':'https://github.com/Zagred47/giada',
            'Percorso entrypoint':'scripts/run_task39i_coupled_relaxation.py','Artefatti codice':arts,'Ambiente':'Official Colab CLI/WSL2;member-vectorized CUDA cohorts; external PC supervisor'})
        armrefs={}
        for a in arms():
            model=put('models',a['name'],Nome='Task39i '+a['name'],Ruolo='Surrogate',Descrizione=spec['architecture']+' '+json.dumps(a))
            armrefs[a['name']]=put('arms',a['name'],Nome='Task39i '+a['name'],Ruolo='Baseline' if a['name'] in ('I0','I10','I11') else 'Trattamento',
                Protocollo=[protocol],Modello=[model],Implementazione=[impl],Partizioni=partitions+extra,**{'Configurazione residua':json.dumps(a)})
            for seed in spec['seeds']:put('runs',f'{a["name"]}-{seed}',Nome=f'Task39i {a["name"]} seed{seed}',Stato='Pianificata',Seed=str(seed),Braccio=[armrefs[a['name']]],
                **{'Configurazione effettiva':json.dumps(dict(arm=a,seed=seed,checkpoints=spec['checkpoints']))})
        for axis,a,z in contrast_pairs():put('contrasts',a+'-vs-'+z,Nome='Task39i '+axis+': '+a+' vs '+z,Protocollo=[protocol],Bracci=[armrefs[a],armrefs[z]],
            **{'Contrasto e coefficienti':spec['contrasts'],'Soglia interpretativa':'Median<=.8,all3<1,worst<=1.05;Task37quality unchanged','Confondenti controllati':spec['training']})
        for name in ('quality','selection','oracles','cost'):
            metric=put('metrics',name,Nome='Task39i '+name,Formula=spec[name]);put('evaluations',name,Nome='Task39i '+name,Versione='v1',Metrica=[metric],Protocollo=[protocol],Partizioni=partitions[2:],Ruolo='Diagnostica',Target=spec[name])
        decision=put('decisions','authorize',Nome='Authorize39i;Task40 remains blocked',Esito='Modificare',Risultati=[ref('findings-giada-task39h-result-outcome-v1')],Motivazione=spec['matrix'],**{'Data':'2026-10-19'})
        put('actions','execute',Nome='ExecuteTask39i onColab console',Stato='Proposta',Descrizione=spec['matrix'],**{'Decisioni di origine':[decision]})
        m.local_upsert('actions',{'Codice stabile':'actions-giada-task39h-result-proposed39i-v1','Stato':'Conclusa','Descrizione':'User authorized39i notebook,DB,push,console launch and completeDrive backup. Actualdispatch tracked separately.'})
    else:
        launch=json.loads((folder/'launch.json').read_text());backup=json.loads((folder/'external_recovery_verified.json').read_text());assert backup['valid']
        arts=[]
        for name in ('launch.json','external_recovery_verified.json','dispatch_request.json','gpu_preflight.json'):
            path=folder/name;rel=path.relative_to(repo).as_posix();arts.append(put('artifacts','dispatch-'+name,Nome='Task39i dispatch '+name,Tipo='Report',Versione='v1',Percorso=rel,
                **{'SHA-256':hashlib.sha256(path.read_bytes()).hexdigest(),'Dimensione byte':path.stat().st_size}))
        for a in arms():
            for seed in spec['seeds']:put('runs',f'{a["name"]}-{seed}',Nome=f'Task39i {a["name"]} seed{seed}',Stato='In corso',Seed=str(seed),
                Braccio=[ref(f'arms-giada-task39i-temporal-{a["name"]}-v1')],**{'Configurazione effettiva':json.dumps(dict(arm=a,seed=seed,launch=launch)),
                'Artefatti prodotti':arts,'Validità tecnica':'Verified GPU36member run and finite OFF-VM weight/Adam checkpoint; scientific result pending.'})
        put('experiments','suite',Nome='Task39i coupled relaxation and learned SK shape',Stato='In corso')
        put('actions','execute',Nome='ExecuteTask39i onColab console',Stato='In corso',Descrizione=json.dumps(launch))
        put('implementations','worker',Nome='Task39i CUDA console worker',Commit=launch['revision'],**{'Ambiente':json.dumps(launch)})
        put('observations','gpu-preflight',Nome='Task39i isolated optimizer andCUDAgraph parity',Numerosità=36,Descrizione=json.dumps(json.loads((folder/'gpu_preflight.json').read_text())),**{'Artefatti dettagliati':arts})
    print(json.dumps(dict(stage='staged',records=len(ids),launch=args.launch)),flush=True)
    m.commit_batch();write_json(ROOT/'data/airtable_snapshot.json',m.export_snapshot());verification=m.verify();assert verification['valid']
    write_json(folder/('launch_memory_receipt.json' if args.launch else 'memory_receipt.json'),dict(valid=True,records=ids,verification=verification,airtable_accessed=False,drive_backup_updated=False,scientific_result_claimed=False))
    print(json.dumps(dict(valid=True,records=len(ids))),flush=True)
if __name__=='__main__':main()
