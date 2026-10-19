"""Preregister39g or record its actual console dispatch in SQLite only."""
import argparse,hashlib,json,zipfile
from functools import lru_cache
from .mirror import ROOT,write_json
from .validated_batch import ValidatedBatchMirror
from src.giada_teacher.task39g_objective_horizon import arms,contrast_pairs,CONFIG

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--launch',action='store_true');args=parser.parse_args()
    repo=ROOT.parent;folder=repo/'experiments/results/task39g_protocol_v1';spec=json.loads((repo/CONFIG).read_text())
    m=ValidatedBatchMirror();ids={}
    def put(table,suffix,**fields):
        code=f'{table}-giada-task39g-temporal-{suffix}-v1';rid=m.local_upsert(table,{'Codice stabile':code,**fields})['record_id'];ids[code]=rid;return rid
    @lru_cache(None)
    def ref(code):
        r=m.query("SELECT record_id FROM v_records WHERE stable_code='"+code+"'")['rows'];assert len(r)==1,code;return r[0]['record_id']
    if args.launch:
        launch=json.loads((folder/'launch.json').read_text());backup=json.loads((folder/'external_recovery_verified.json').read_text());assert backup['valid']
        failed_archive=repo/'artifacts/task39g_reserve_external/recovery_final.zip'
        assert hashlib.sha256(failed_archive.read_bytes()).hexdigest()=='1696fca1a510777252c388ea7be106a213c0ee584dbcf1e5b6e0068cf193bf4d'
        with zipfile.ZipFile(failed_archive) as z:
            assert z.testzip() is None
            failed=dict(failure=json.loads(z.read('failure_report.json')),original_preregistration=json.loads(z.read('preregistration.json')),source_revision='8f8fa556a1d9b916c2f2b2d63c712f9f15d3bf13',training_updates=0,archive_sha256=hashlib.sha256(failed_archive.read_bytes()).hexdigest())
        assert failed['failure']['technical_failure'] and not failed['failure']['scientific_no_go']
        write_json(folder/'attempt1_failure.json',failed)
        probe32=json.loads((folder/'attempt1_parity_fp32.json').read_text());probe64=json.loads((folder/'attempt1_parity_fp64_affines.json').read_text())
        assert max(r['adam_first_step_error']['1e-08'] for r in probe32['rows'])>.001
        assert max(r['adam_first_step_error']['1e-08'] for r in probe64['rows'])<1e-5
        art=[]
        paths=[folder/n for n in ('launch.json','external_recovery_verified.json','dispatch_request.json','local_validation.json','attempt1_failure.json','attempt1_parity_fp32.json','attempt1_parity_fp64_affines.json')]
        paths += [repo/CONFIG,failed_archive,repo/'scripts/probe_task39g_gpu_parity.py',repo/'scripts/probe_task39g_fp64_affines.py']
        paths += [repo/rel for rel in ('src/giada_teacher/task39g_objective_horizon.py','scripts/colab_profile_cli.py','scripts/colab.ps1','scripts/colab_checked_exec.py','scripts/colab_checked_download.py','scripts/launch_task39g_console.ps1','scripts/record_task39g_dispatch.py','docs/colab_console_operations.md','research_memory/register_task39g.py')]
        art_by_path={}
        for path in paths:
            rel=path.relative_to(repo).as_posix()
            art.append(put('artifacts','dispatch-'+rel,Nome='Task39g dispatch '+path.name,Tipo='Codice' if path.suffix in ('.py','.ps1') else 'Report',Percorso=rel,Versione='reserve dispatch v1',**{'SHA-256':hashlib.sha256(path.read_bytes()).hexdigest(),'Dimensione byte':path.stat().st_size}))
            art_by_path[path]=art[-1]
        failed_runs=[];live_runs=[]
        for a in arms():
            for seed in spec['seeds']:
                failed_runs.append(put('runs',f'{a["name"]}-{seed}',Nome=f'Task39g attempt1 {a["name"]} seed{seed}',Stato='Fallita',
                    **{'Configurazione effettiva':json.dumps(dict(arm=a,seed=seed,revision=failed['source_revision'],training_updates=0)),'Validità tecnica':'Technical preflight stop before training; no scientific NO-GO. Full cohort not fitted.','Artefatti prodotti':[art_by_path[failed_archive],art_by_path[folder/'attempt1_failure.json']]}))
                live_runs.append(put('runs',f'{a["name"]}-{seed}-retry2',Nome=f'Task39g retry2 {a["name"]} seed{seed}',Stato='In corso',Seed=str(seed),Braccio=[ref(f'arms-giada-task39g-temporal-{a["name"]}-v1')],
                    **{'Configurazione effettiva':json.dumps(dict(arm=a,seed=seed,launch=launch,numerical_amendment=spec['numerical_amendment'])),'Validità tecnica':'Console launched; full30-member preflight and externally verified recovery. Scientific result pending.','Artefatti prodotti':art}))
        observation=put('observations','numerical-preflight',Nome='Task39g compact/vectorized FP32 gradient sensitivity on T4',Descrizione=json.dumps(dict(failed=failed,probe32=probe32,probe64=probe64)),Numerosità=18,
            **{'Artefatti dettagliati':[art_by_path[folder/n] for n in ('attempt1_failure.json','attempt1_parity_fp32.json','attempt1_parity_fp64_affines.json')]})
        finding=put('findings','numerical-preflight',Nome='Task39g precision amendment before training',Esito='Misto',Esperimenti=[ref('experiments-giada-task39g-temporal-suite-v1')],Osservazioni=[observation],
            **{'Risultato':spec['numerical_amendment'],'Limitazioni':'Paired18-member T4 probe concerns initial FIT preflight, not proof of cause of prior39f scientific failure. Full30-member parity and capture are required in retry2. Scientific thresholds/optimizer/loss factors unchanged.'})
        put('analyses','numerical-preflight',Nome='Task39g retained failed attempt and paired precision intervention',Origine='Post hoc',Versione='v1',Procedura='scripts/probe_task39g_gpu_parity.py; scripts/probe_task39g_fp64_affines.py',Configurazione=json.dumps(dict(source=failed['source_revision'],training_updates=0)),**{'Run analizzati':failed_runs,'Input artefatti':art,'12 · Risultati scientifici':[finding]})
        put('lessons','numerical-preflight',Nome='Warm near-canceling gradients need GPU update parity',Lezione=spec['numerical_amendment'],Risultati=[finding],Limiti='Inference remains FP32 and quality is evaluated there; training precision is an execution amendment common to all arms.')
        m.local_upsert('protocols',{'Codice stabile':'protocols-giada-task39g-temporal-suite-v1','Versione':spec['version'],'Procedura':json.dumps(spec),'Artefatto preregistrazione':[art_by_path[repo/CONFIG]],'Hash preregistrazione':hashlib.sha256((repo/CONFIG).read_bytes()).hexdigest()})
        put('actions','execute',Nome='Execute Task39g on Colab console',Stato='In corso',Descrizione=json.dumps(launch))
        put('experiments','suite',Nome='Task39g objective/horizon/independent support',Stato='In corso')
        put('implementations','worker',Nome='Task39g CUDA console worker',Commit=launch['revision'],**{'Ambiente':json.dumps(dict(profile=launch['account_profile'],account=launch['account'],gpu=launch['gpu'],session=launch['session']))})
    else:
        assert json.loads((folder/'local_validation.json').read_text())['valid']
        paths=[CONFIG,'src/giada_teacher/task39f_temporal_recovery.py','src/giada_teacher/task37_ionic_playground.py','scripts/launch_task39g_console.ps1','scripts/record_task39g_dispatch.py','src/giada_teacher/task39g_objective_horizon.py','src/giada_teacher/task39e_tail_support_heads.py','src/giada_teacher/task39d_current_timing.py','scripts/run_task39g_objective_horizon.py','scripts/build_task39g_protocol.py','scripts/validate_task39g.py','tests/test_task39g_objective_horizon.py','notebooks/giada_task39g_objective_horizon.ipynb','scripts/colab_task39g_driver.py','scripts/colab_task39e_driver.py','scripts/task39e_console_manager.py','scripts/task39e_local_supervisor.py','experiments/giada_modular_surrogate_roadmap.md','experiments/results/task39g_protocol_v1/local_validation.json','research_memory/register_task39g.py']
        arts=[]
        for rel in paths:
            p=repo/rel;arts.append(put('artifacts',rel,Nome='Task39g '+p.name,Tipo='Codice' if p.suffix=='.py' else 'Report',Versione='v1',Percorso=rel,**{'SHA-256':hashlib.sha256(p.read_bytes()).hexdigest(),'Dimensione byte':p.stat().st_size}))
        claims=[ref('claims-giada-task39f-result-residual-'+n+'-v1') for n in ('temporal_gradient_dominance','temporal_objective_anatomy','temporal_credit_horizon','rate_equilibrium_accumulation','independent_rare_support','event_conditioning_or_precision')]
        exp=put('experiments','suite',Nome='Task39g objective/horizon/independent support',Stato='Preregistrato',Ipotesi=claims,Descrizione=spec['matrix'],**{'Obiettivo informativo':spec['oracles']+' '+spec['contrasts']})
        partitions=[ref(f'partitions-giada-task37-ionic-playground-{r}-v1') for r in ('fit','fit_extra','development','stress_diagnostic')]
        generator=put('generators','expanded-fit',Nome='Task39g independent FIT teacher recipe',Versione='v1',Procedura=spec['expanded_support'],
            **{'Seed e RNG':json.dumps(spec['expanded_seeds']),'Distribuzioni input':'Same Task37 domains/families;12 independent histories per role/family; all11 reversal channels','Variabili registrate':'V,Ca,initial18gates,11multipliers,11reversals,full gate/current targets,sourceCa,input signatures','Artefatti codice':arts,'Descrizione':'Planned generation on runtime; not yet materialized at preregistration.'})
        dataset=put('datasets','expanded-fit',Nome='Task39g planned independent FIT extension',Versione='v1',Generatore=[generator],Artefatti=[arts[0]],
            Fingerprint='recipe-sha256:'+hashlib.sha256(spec['expanded_support'].encode()).hexdigest(),Schema='Same Task37 NPZ fields; offline episode event metadata is derived by39g.',Copertura=spec['expanded_support'],
            **{'Numero esempi':288,'Descrizione':'Planned288 trajectories=12families x12histories x2FIT roles. Actual source hashes and disjointness audited before training; not a new evaluation partition.'})
        additional=[]
        for role,seed in spec['expanded_seeds'].items():
            additional.append(put('partitions','expanded-'+role,Nome='Task39g independent '+role,Ruolo='Fit',Dataset=[dataset],
                **{'Regola di assegnazione':json.dumps(dict(role=role,seed=seed,trajectories_per_family=12)),'Restrizioni di accesso':'FIT only; no development/stress/confirmation generation or copied failing traces.','Descrizione':'Whole histories; runtime verifies disjoint input signatures against all original roles.'}))
        protocol=put('protocols','suite',Nome='Task39g paired30-model protocol',Versione='v1',Modalità='Ricomposizione',Esperimento=[exp],Procedura=json.dumps(spec),
            **{'Partizioni autorizzate':partitions+additional,'Artefatto preregistrazione':[arts[0]],'Hash preregistrazione':hashlib.sha256((repo/CONFIG).read_bytes()).hexdigest(),'Criteri di successo':spec['quality'],'Regole di arresto':spec['preflights'],'Registrato il':'2026-10-19'})
        impl=put('implementations','worker',Nome='Task39g CUDA console worker',**{'Repository URL':'https://github.com/Zagred47/giada','Percorso entrypoint':'scripts/run_task39g_objective_horizon.py','Artefatti codice':arts,'Ambiente':'Official Colab CLI0.7.4/WSL2;30 members,2 CUDA cohorts; external PC supervisor'})
        aa={}
        for a in arms():
            model=put('models',a['name'],Nome='Task39g '+a['name'],Ruolo='Surrogate',Descrizione=spec['architecture']+' '+json.dumps(a))
            aa[a['name']]=put('arms',a['name'],Nome='Task39g '+a['name'],Ruolo='Trattamento',Protocollo=[protocol],Modello=[model],Implementazione=[impl],Partizioni=partitions+(additional if a['expanded'] else []),**{'Configurazione residua':json.dumps(a)})
            for seed in spec['seeds']:
                put('runs',f'{a["name"]}-{seed}',Nome=f'Task39g {a["name"]} seed{seed}',Stato='Pianificata',Seed=str(seed),Braccio=[aa[a['name']]],**{'Configurazione effettiva':json.dumps(dict(arm=a,seed=seed,checkpoints=spec['checkpoints']))})
        for axis,a,z in contrast_pairs(arms()):
            put('contrasts',a+'-vs-'+z,Nome='Task39g '+axis+': '+a+' vs '+z,Protocollo=[protocol],Bracci=[aa[a],aa[z]],**{'Contrasto e coefficienti':spec['contrasts'],'Soglia interpretativa':'Median<=.8,all3<1,worst<=1.05; unchanged quality','Confondenti controllati':spec['training']})
        for n in ('quality','selection','oracles','cost'):
            metric=put('metrics',n,Nome='Task39g '+n,Formula=spec[n]);put('evaluations',n,Nome='Task39g '+n,Versione='v1',Metrica=[metric],Protocollo=[protocol],Partizioni=partitions[2:],Ruolo='Diagnostica',Target=spec[n])
        decision=put('decisions','authorize',Nome='Authorize39g, keep Task40 blocked',Esito='Modificare',Risultati=[ref('findings-giada-task39f-result-outcome-v1')],Motivazione=spec['matrix']+' '+spec['limits'],**{'Data':'2026-10-19'})
        put('actions','execute',Nome='Execute Task39g on Colab console',Stato='Proposta',Descrizione=spec['matrix'],**{'Decisioni di origine':[decision]})
        m.local_upsert('actions',{'Codice stabile':'actions-giada-task39f-result-proposed39g-v1','Stato':'Conclusa','Descrizione':'User authorized39g preparation, preregistration, push and console dispatch. Actual dispatch tracked separately.'})
    print(json.dumps(dict(stage='staged',records=len(ids),launch=args.launch)),flush=True)
    m.commit_batch();write_json(ROOT/'data/airtable_snapshot.json',m.export_snapshot());v=m.verify();assert v['valid']
    write_json(folder/('launch_memory_receipt.json' if args.launch else 'memory_receipt.json'),dict(valid=True,records=ids,verification=v,airtable_accessed=False,drive_backup_updated=False,scientific_result_claimed=False))
    print(json.dumps(dict(valid=True,records=len(ids))),flush=True)
if __name__=='__main__':main()
