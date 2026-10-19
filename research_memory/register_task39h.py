"""Preregister39h or record its actual console dispatch in SQLite only."""
import argparse,hashlib,json,zipfile
from functools import lru_cache
from .mirror import ROOT,write_json
from .validated_batch import ValidatedBatchMirror
from src.giada_teacher.task39h_relaxation_identification import arms,contrast_pairs,CONFIG

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--launch',action='store_true');args=parser.parse_args()
    repo=ROOT.parent;folder=repo/'experiments/results/task39h_protocol_v1';spec=json.loads((repo/CONFIG).read_text())
    m=ValidatedBatchMirror();ids={}
    def put(table,suffix,**fields):
        code=f'{table}-giada-task39h-temporal-{suffix}-v1';rid=m.local_upsert(table,{'Codice stabile':code,**fields})['record_id'];ids[code]=rid;return rid
    @lru_cache(None)
    def ref(code):
        r=m.query("SELECT record_id FROM v_records WHERE stable_code='"+code+"'")['rows'];assert len(r)==1,code;return r[0]['record_id']
    if args.launch:
        launch=json.loads((folder/'launch.json').read_text());backup=json.loads((folder/'external_recovery_verified.json').read_text());assert backup['valid']
        paths=[folder/n for n in ('launch.json','external_recovery_verified.json','dispatch_request.json')]
        paths += [folder/n for n in ('gpu_preflight.json','gpu_preflight_attempt1.json','residual_preflight.json')]
        paths += [repo/'research_memory/register_task39h.py']
        arts=[]
        for path in paths:
            rel=path.relative_to(repo).as_posix()
            arts.append(put('artifacts','dispatch-'+rel,Nome='Task39h dispatch '+path.name,Tipo='Codice' if path.suffix=='.py' else 'Report',Versione='v1',Percorso=rel,**{'SHA-256':hashlib.sha256(path.read_bytes()).hexdigest(),'Dimensione byte':path.stat().st_size}))
        for a in arms():
            for seed in spec['seeds']:
                put('runs',f'{a["name"]}-{seed}',Nome=f'Task39h {a["name"]} seed{seed}',Stato='In corso',Seed=str(seed),
                    Braccio=[ref(f'arms-giada-task39h-temporal-{a["name"]}-v1')],
                    **{'Configurazione effettiva':json.dumps(dict(arm=a,seed=seed,launch=launch)),'Artefatti prodotti':arts,
                    'Validità tecnica':'Verified GPU console launch and OFF-VM finite checkpoint/Adam; scientific result pending.'})
        put('experiments','suite',Nome='Task39h relaxation and conservative optimization',Stato='In corso')
        put('actions','execute',Nome='Execute Task39h on Colab console',Stato='In corso',Descrizione=json.dumps(launch))
        put('implementations','worker',Nome='Task39h CUDA console worker',Commit=launch['revision'],**{'Ambiente':json.dumps(launch)})
        diagnostic={n:json.loads((folder/n).read_text()) for n in ('gpu_preflight.json','gpu_preflight_attempt1.json','residual_preflight.json')}
        observation=put('observations','gpu-preflight',Nome='Task39h all30 masked Adam and CUDA graph checks',Numerosità=30,
            Descrizione=json.dumps(dict(**diagnostic,scientific_training_updates=0,note='30members are numerical parity probes,not independent scientific repetitions.')),
            **{'Artefatti dettagliati':arts})
        finding=put('findings','gpu-preflight',Nome='Task39h immutable CUDA constants must outlive capture',Esito='Misto',
            Esperimenti=[ref('experiments-giada-task39h-temporal-suite-v1')],Osservazioni=[observation],
            **{'Risultato':'Initial GPU update parity passed; capture rejected a CPU-to-CUDA constant copy. Per-device immutable edge caching passed the repeated isolated GPU test with exact two-step replay and zero frozen-u drift. No threshold/objective/architecture change.',
            'Limitazioni':'Technical preflight,zero scientific training updates. Actual run quality pending.'})
        put('lessons','gpu-capture',Nome='Allocate immutable loss constants before CUDA graph capture',Lezione=diagnostic['gpu_preflight_attempt1.json']['correction'],Risultati=[finding],
            Limiti='Validated on30members,TeslaT4,PyTorch2.11.0+cu130; repeat parity on a changed backend.')
    if not args.launch:
        assert json.loads((folder/'local_validation.json').read_text())['valid']
        paths=[CONFIG,'src/giada_teacher/task39g_objective_horizon.py','src/giada_teacher/task39f_temporal_recovery.py','src/giada_teacher/task37_ionic_playground.py','scripts/launch_task39h_console.ps1','scripts/record_task39h_dispatch.py','scripts/probe_task39h_gpu.py','src/giada_teacher/task39h_relaxation_identification.py','src/giada_teacher/task39e_tail_support_heads.py','src/giada_teacher/task39d_current_timing.py','scripts/run_task39h_relaxation_identification.py','scripts/build_task39h_protocol.py','scripts/validate_task39h.py','tests/test_task39h_relaxation_identification.py','notebooks/giada_task39h_relaxation_identification.ipynb','scripts/colab_task39h_driver.py','scripts/colab_task39e_driver.py','scripts/task39e_console_manager.py','scripts/task39e_local_supervisor.py','experiments/giada_modular_surrogate_roadmap.md','experiments/results/task39h_protocol_v1/local_validation.json','research_memory/register_task39h.py']
        arts=[]
        paths += ['experiments/results/task39h_protocol_v1/gpu_preflight.json','experiments/results/task39h_protocol_v1/gpu_preflight_attempt1.json','experiments/results/task39h_protocol_v1/source_attempt1.py','experiments/results/task39h_protocol_v1/residual_preflight.json','scripts/probe_task39h_driver.py','scripts/probe_task39h_residual.py','scripts/prepare_task39h_dispatch.py']
        for rel in paths:
            p=repo/rel;arts.append(put('artifacts',rel,Nome='Task39h '+p.name,Tipo='Codice' if p.suffix=='.py' else 'Report',Versione='v1',Percorso=rel,**{'SHA-256':hashlib.sha256(p.read_bytes()).hexdigest(),'Dimensione byte':p.stat().st_size}))
        claims=[ref('claims-giada-task39g-result-residual-'+n+'-v1') for n in ('relaxation_bias_accumulation','warm_finetuning_step_interference','mean_loss_hides_rare_rate_error','rare_recovery_support','event_conditioning_and_other_currents')]
        exp=put('experiments','suite',Nome='Task39h relaxation and conservative optimization',Stato='Preregistrato',Ipotesi=claims,Descrizione=spec['matrix'],**{'Obiettivo informativo':spec['oracles']+' '+spec['contrasts']})
        partitions=[ref(f'partitions-giada-task37-ionic-playground-{r}-v1') for r in ('fit','fit_extra','development','stress_diagnostic')]
        generator=put('generators','expanded-fit',Nome='Task39h recovery FIT teacher recipe',Versione='v1',Procedura=spec['expanded_support'],
            **{'Seed e RNG':json.dumps(spec['expanded_seeds']),'Distribuzioni input':'2048nodes;18gates x2 quantiles x2FIT roles;36held histories per role','Variabili registrate':'V,Ca,initial18gates,11multipliers,11reversals,full gate/current targets,sourceCa,input signatures','Artefatti codice':arts,'Descrizione':'Planned generation on runtime; not yet materialized at preregistration.'})
        dataset=put('datasets','expanded-fit',Nome='Task39h planned independent recovery FIT',Versione='v1',Generatore=[generator],Artefatti=[arts[0]],
            Fingerprint='recipe-sha256:'+hashlib.sha256(spec['expanded_support'].encode()).hexdigest(),Schema='Same Task37 NPZ fields;71-field variable windows and107-field coefficient anchors. No event metadata in candidate inputs.',Copertura=spec['expanded_support'],
            **{'Numero esempi':72,'Descrizione':'Planned72 trajectories=18gates x2quantiles x2FIT roles. Actual source hashes and disjointness audited before training; not a new evaluation partition.'})
        additional=[]
        for role,seed in spec['expanded_seeds'].items():
            additional.append(put('partitions','expanded-'+role,Nome='Task39h independent '+role,Ruolo='Fit',Dataset=[dataset],
                **{'Regola di assegnazione':json.dumps(dict(role=role,seed=seed,recovery_histories_per_role=36)),'Restrizioni di accesso':'FIT only; no development/stress/confirmation generation or copied failing traces.','Descrizione':'Whole histories; runtime verifies disjoint input signatures against all original roles.'}))
        protocol=put('protocols','suite',Nome='Task39h paired30-model protocol',Versione='v1',Modalità='Ricomposizione',Esperimento=[exp],Procedura=json.dumps(spec),
            **{'Partizioni autorizzate':partitions+additional,'Artefatto preregistrazione':[arts[0]],'Hash preregistrazione':hashlib.sha256((repo/CONFIG).read_bytes()).hexdigest(),'Criteri di successo':spec['quality'],'Regole di arresto':spec['preflights'],'Registrato il':'2026-10-19'})
        impl=put('implementations','worker',Nome='Task39h CUDA console worker',**{'Repository URL':'https://github.com/Zagred47/giada','Percorso entrypoint':'scripts/run_task39h_relaxation_identification.py','Artefatti codice':arts,'Ambiente':'Official Colab CLI0.7.4/WSL2;30 members,2 CUDA cohorts; external PC supervisor'})
        aa={}
        for a in arms():
            model=put('models',a['name'],Nome='Task39h '+a['name'],Ruolo='Surrogate',Descrizione=spec['architecture']+' '+json.dumps(a))
            aa[a['name']]=put('arms',a['name'],Nome='Task39h '+a['name'],Ruolo='Trattamento',Protocollo=[protocol],Modello=[model],Implementazione=[impl],Partizioni=partitions+(additional if a['expanded'] else []),**{'Configurazione residua':json.dumps(a)})
            for seed in spec['seeds']:
                put('runs',f'{a["name"]}-{seed}',Nome=f'Task39h {a["name"]} seed{seed}',Stato='Pianificata',Seed=str(seed),Braccio=[aa[a['name']]],**{'Configurazione effettiva':json.dumps(dict(arm=a,seed=seed,checkpoints=spec['checkpoints']))})
        for axis,a,z in contrast_pairs(arms()):
            put('contrasts',a+'-vs-'+z,Nome='Task39h '+axis+': '+a+' vs '+z,Protocollo=[protocol],Bracci=[aa[a],aa[z]],**{'Contrasto e coefficienti':spec['contrasts'],'Soglia interpretativa':'Median<=.8,all3<1,worst<=1.05; unchanged quality','Confondenti controllati':spec['training']})
        for n in ('quality','selection','oracles','cost'):
            metric=put('metrics',n,Nome='Task39h '+n,Formula=spec[n]);put('evaluations',n,Nome='Task39h '+n,Versione='v1',Metrica=[metric],Protocollo=[protocol],Partizioni=partitions[2:],Ruolo='Diagnostica',Target=spec[n])
        decision=put('decisions','authorize',Nome='Authorize39h, keep Task40 blocked',Esito='Modificare',Risultati=[ref('findings-giada-task39g-result-outcome-v1')],Motivazione=spec['matrix']+' '+spec['limits'],**{'Data':'2026-10-19'})
        put('actions','execute',Nome='Execute Task39h on Colab console',Stato='Proposta',Descrizione=spec['matrix'],**{'Decisioni di origine':[decision]})
        m.local_upsert('actions',{'Codice stabile':'actions-giada-task39g-result-proposed39h-v1','Stato':'Conclusa','Descrizione':'User authorized39h preparation, preregistration, push and console dispatch. Actual dispatch tracked separately.'})
    print(json.dumps(dict(stage='staged',records=len(ids),launch=args.launch)),flush=True)
    m.commit_batch();write_json(ROOT/'data/airtable_snapshot.json',m.export_snapshot());v=m.verify();assert v['valid']
    write_json(folder/('launch_memory_receipt.json' if args.launch else 'memory_receipt.json'),dict(valid=True,records=ids,verification=v,airtable_accessed=False,drive_backup_updated=False,scientific_result_claimed=False))
    print(json.dumps(dict(valid=True,records=len(ids))),flush=True)
if __name__=='__main__':main()
