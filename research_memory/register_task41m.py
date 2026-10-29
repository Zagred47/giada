"""Preregister frozen41m before run, and later record verified launch. Local only."""
import argparse,hashlib,json
from datetime import datetime,timezone
from functools import lru_cache
from .mirror import ROOT,write_json
from .incremental_batch import IncrementalBatchMirror
from .register_task41d_result import ResultPreview,PreviewComplete

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--validate-only',action='store_true');ap.add_argument('--running',action='store_true');args=ap.parse_args()
    repo=ROOT.parent;folder=repo/'experiments/results/task41m_protocol_v1';folder.mkdir(exist_ok=True)
    path=repo/'experiments/task41m_preregistration_v1.json';spec=json.loads(path.read_text());sha=hashlib.sha256(path.read_bytes()).hexdigest()
    gpu=json.loads((folder/'gpu_preflight.json').read_text());assert gpu['valid'] and gpu['all_parameters_finite'] and gpu['protocol_sha256']==sha
    baseline_path=folder/'memory_receipt.json' if args.running else repo/'experiments/results/task41l_audited_v1/memory_receipt.json'
    baseline=json.loads(baseline_path.read_text())['verification'];m=ResultPreview() if args.validate_only else IncrementalBatchMirror();ids={}
    if not args.validate_only:assert m.baseline_hash==baseline['snapshot_hash']
    def put(table,suffix,**fields):
        code=f'{table}-giada-task41m-{suffix}-v1';rid=m.local_upsert(table,{'Codice stabile':code,**fields})['record_id'];ids[code]=rid;return rid
    @lru_cache(None)
    def ref(code):
        if code in ids:return ids[code]
        rows=m.query("SELECT record_id FROM v_records WHERE stable_code='"+code+"'")['rows'];assert len(rows)==1,code;return rows[0]['record_id']
    date=datetime.now(timezone.utc).date().isoformat();conditions=[model+'/'+policy for model in spec['models'] for policy in spec['policies']]
    if args.running:
        launch=json.loads((folder/'launch.json').read_text());assert launch['valid'] and launch['external_backup']['valid']
        obs=put('observations','launch',Nome='Task41m actual console launch and off-runtime backup',Descrizione=json.dumps(launch))
        exp=put('experiments','frozen-contract',Stato='In corso')
        for condition in conditions:put('runs',condition,Stato='In corso',**{'Validità tecnica':'Console worker observed; first immutable external bundle SHA/CRC verified. Evaluation only, no result claimed.'})
        put('findings','launch',Nome='Task41m actual frozen verification running',Esito='Positivo',Esperimenti=[exp],Osservazioni=[obs],Risultato='Pinned Colab T4 console launch with external backup verified.0training updates.',Limitazioni='Technical progress only; no scientific claim.')
    else:
        arts=[]
        for filename in [str(path.relative_to(repo)),*spec['sources'],'scripts/task41m_colab_manager.py','scripts/task41m_colab_supervisor.py','scripts/task41m_colab_prepare.py','research_memory/register_task41m.py','experiments/results/task41m_protocol_v1/gpu_preflight.json']:
            item=repo/filename;raw=item.read_bytes()
            arts.append(put('artifacts',filename,Nome='Task41m '+item.name,Tipo='Codice' if item.suffix=='.py' else 'Report',Versione='v1',Percorso=filename,**{'SHA-256':hashlib.sha256(raw).hexdigest(),'Dimensione byte':len(raw)}))
        claims=[]
        for h in spec['hypotheses']:
            claims.append(put('claims',h['id'],Nome='Task41m '+h['id'],Tipo='Ipotesi',Stato='Aperta',Enunciato=h['statement'],**{'Condizioni di falsificazione':h['falsifier'],'Limiti':'Prospective frozen diagnostic, not confirmation. '+h['contrast']}))
        exp=put('experiments','frozen-contract',Nome='Task41m frozen selective repair and numerical contract',Stato='Preregistrato',Ipotesi=claims,Descrizione=spec['purpose'],**{'Obiettivo informativo':'Full-panel transfer and matched/original-reference discrimination; bounded60minutes, no training/compression.'})
        protocol=put('protocols','design',Nome='Task41m sealed bounded verification',Versione='v1',Modalità='Ricomposizione',Esperimento=[exp],Procedura=json.dumps(spec),**{'Artefatto preregistrazione':[arts[0]],'Hash preregistrazione':sha,'Criteri di successo':json.dumps(spec['outcomes']),'Regole di arresto':json.dumps(spec['stop_rules']),'Registrato il':date})
        aa={}
        for condition in conditions:
            model,policy=condition.split('/')
            aa[condition]=put('arms',condition,Nome='Task41m '+condition,Ruolo='Controllo' if model=='swap000' else 'Trattamento',Protocollo=[protocol],**{'Configurazione residua':json.dumps(dict(model=model,numerics=policy,weights='immutable',training_updates=0,seeds=spec['seeds'],references=['original','matched']))})
            put('runs',condition,Nome='Task41m frozen evaluation '+condition,Stato='Pianificata',Seed='17,29,43 paired/vectorized',Braccio=[aa[condition]],**{'Configurazione effettiva':json.dumps(spec),'Validità tecnica':'Actual GPU preflight passed; scientific evaluation not started.'})
        for model in spec['models']:
            for policy in spec['policies'][1:]:
                put('contrasts',model+'-'+policy,Nome='Task41m '+model+' '+policy+' vs primary',Protocollo=[protocol],Bracci=[aa[model+'/primary'],aa[model+'/'+policy]],**{'Contrasto e coefficienti':'Paired case/seed failures and continuous errors, original and matched reference; also compare elapsed integrated cost.','Soglia interpretativa':'Original exact predicates retained; no automatic qualification or Task42.','Confondenti controllati':'Same frozen weights,full batch,seed,stimulus,dt; numerical policy only.'})
        for key in ('quality','compartment_gates','cost'):
            metric=put('metrics',key,Nome='Task41m unchanged '+key,Formula=json.dumps(spec[key]))
            put('evaluations',key,Nome='Task41m '+key,Versione='v1',Metrica=[metric],Protocollo=[protocol],Ruolo='Primaria' if key!='cost' else 'Secondaria',Target=json.dumps(spec[key]))
        history=[json.loads(p.read_text()) for p in sorted(folder.glob('preflight_failure_*.json'))]
        put('observations','preflight',Nome='Task41m CPU/GPU execution readiness',Descrizione=json.dumps(dict(gpu=gpu,technical_preflight_history=history)),**{'Artefatti dettagliati':arts})
        prior=[ref(code) for code in ('findings-giada-task41f-result-numerical-v1','findings-giada-task41i-result-numerics-v1','findings-giada-task41k-result-numerical-interaction-v1','findings-giada-task41c-audit-ca_mechanism-v1')]
        put('lessons','historical-pattern-and-gap',Nome='41m query-derived pattern: numeric feedback is known, full selective transfer is not',Lezione='Readonly queries retrieve41f/41i/41k numerical-feedback findings and41c Ca_HVA equilibrium-offset attribution. Repeating selected-case precision success would add little.41m tests complete panels and explicitly crosses original versus matched reference, with frozen Ca-only transplant and actual H timing. Associations guide contrasts, not causal conclusions.',Risultati=prior+[ref('findings-giada-task41l-result-local_separability-v1')],Limiti='No new independent evidence from merely querying memory; oldFP64 successes never prove all stress gates or useful deployment speed.')
        put('decisions','bounded-go',Nome='User-authorized bounded41m, no indefinite training',Esito='Continuare',Risultati=[ref('findings-giada-task41l-result-stop_policy-v1'),ref('findings-giada-task41l-result-local_separability-v1'),ref('findings-giada-task41l-result-feedback_numerics-v1')],Motivazione='User asks to preserve all evidence and proceed, with material progress or actionable information required. Frozen evaluation only; no automatic additional training. DBbackup stays local.',**{'Data':date})
    change=m.commit_batch()
    if args.validate_only:return
    verified=m.verify_incremental(baseline);assert verified['valid']
    write_json(folder/('launch_memory_receipt.json' if args.running else 'memory_receipt.json'),dict(valid=True,verification=verified,change=change,records=ids,protocol_sha256=sha,registration_before_run=not args.running,training_updates=0,airtable_accessed=False,DB_backup='local refresh pending'))
    if args.running:m.export_snapshot_to(ROOT/'data/airtable_snapshot.json')
    print('Task41m mirror verified',flush=True)

if __name__=='__main__':
    try:main()
    except PreviewComplete:print('Schema preview valid; no writes')
