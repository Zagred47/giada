"""Sealed41n and verified console launch; local mirror only."""
import argparse,hashlib,json,subprocess
from datetime import datetime,timezone
from functools import lru_cache
from .mirror import ROOT,write_json
from .incremental_batch import IncrementalBatchMirror
from .register_task41d_result import ResultPreview,PreviewComplete

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--validate-only',action='store_true');ap.add_argument('--running',action='store_true');args=ap.parse_args()
    repo=ROOT.parent;folder=repo/'experiments/results/task41n_protocol_v1';spec_path=repo/'experiments/task41n_preregistration_v1.json';spec=json.loads(spec_path.read_text());digest=hashlib.sha256(spec_path.read_bytes()).hexdigest()
    local=json.loads((folder/'local_verification.json').read_text());assert local['valid'] and local['protocol_sha256']==digest
    baseline_path=folder/'memory_receipt.json' if args.running else repo/'experiments/results/task41m_audited_v1/memory_receipt.json'
    baseline=json.loads(baseline_path.read_text())['verification'];m=ResultPreview() if args.validate_only else IncrementalBatchMirror();ids={}
    if not args.validate_only:assert m.baseline_hash==baseline['snapshot_hash']
    def put(table,suffix,**fields):
        code=f'{table}-giada-task41n-{suffix}-v1';rid=m.local_upsert(table,{'Codice stabile':code,**fields})['record_id'];ids[code]=rid;return rid
    @lru_cache(None)
    def ref(code):
        if code in ids:return ids[code]
        rows=m.query("SELECT record_id FROM v_records WHERE stable_code='"+code+"'")['rows'];assert len(rows)==1,code;return rows[0]['record_id']
    names=spec['execution_variants']+['H_control','dual_fp32'];date=datetime.now(timezone.utc).date().isoformat()
    if args.running:
        launch=json.loads((folder/'launch.json').read_text());assert launch['valid'] and launch['external_backup']['valid']
        obs=put('observations','launch',Nome='Task41n pinned console process and off-runtime checkpoint',Descrizione=json.dumps(launch))
        exp=put('experiments','confirmation',Stato='In corso')
        for name in names:put('runs',name,Stato='In corso',**{'Validità tecnica':'Pinned console manager observed; first off-runtime bundle verified. Native/GPU preflight gates still precede fresh interpretation. No training.'})
        put('findings','launch',Nome='Task41n console workflow active',Esito='Positivo',Esperimenti=[exp],Osservazioni=[obs],Risultato='Actual pinned Colab process and first external archive verified.',Limitazioni='Technical launch, not a scientific result or proof all preflight phases passed.')
    else:
        arts=[]
        files=['experiments/task41n_preregistration_v1.json',*spec['sources'],'scripts/task41n_colab_manager.py','scripts/task41n_colab_supervisor.py','scripts/colab_task41n_driver.py','scripts/dispatch_task41n_colab.py','scripts/verify_task41n_local.py','scripts/task41n_native_import_probe.py','research_memory/register_task41n.py','experiments/results/task41n_protocol_v1/local_verification.json']
        for filename in files:
            raw=(repo/filename).read_bytes();arts.append(put('artifacts',filename,Nome='Task41n '+filename,Tipo='Codice' if filename.endswith('.py') else 'Report',Versione='v1',Percorso=filename,**{'SHA-256':hashlib.sha256(raw).hexdigest(),'Dimensione byte':len(raw)}))
        source=put('artifacts','frozen-input',Nome='Task41n immutable41m selected function weights',Tipo='Report',Versione='v1',Percorso=spec['frozen_input']['local'],**{'SHA-256':spec['frozen_input']['sha256'],'Dimensione byte':spec['frozen_input']['bytes']})
        claims=[put('claims',h['id'],Nome='Task41n '+h['id'],Tipo='Ipotesi',Stato='Aperta',Enunciato=h['statement'],**{'Condizioni di falsificazione':h['falsifier'],'Limiti':'Prospective; no fresh results or optimization guaranteed.'}) for h in spec['hypotheses']]
        exp=put('experiments','confirmation',Nome='Task41n independent frozen confirmation and exact execution',Tipo='Confermativo',Stato='Preregistrato',Ipotesi=claims,Descrizione='New native protocols and release seeds; frozen Ca-only mixed64 function. Exact execution variants, not new learned architecture.',**{'Obiettivo informativo':'Separate independent quality transfer, exact function preservation and practical acceleration; all thresholds unchanged.'})
        gen=put('generators','fresh',Nome='Task41n fresh native/stress generator',Versione='v1',Procedura=json.dumps(spec),**{'Seed e RNG':json.dumps(dict(native=spec['native_seeds'],stress=spec['stress_seeds'],calibration=spec['calibration_seed'])),'Artefatti codice':arts})
        ds=put('datasets','fresh',Nome='Task41n prospectively sealed independent inputs',Versione='v1',Generatore=[gen],Schema='176native traces and24imposed1200ms histories; no samples generated at registration.',Copertura=json.dumps(dict(protocols=spec['protocols'],panels=spec['panels'],interventions=spec['interventions'],stress_families=spec['stress_families'])))
        test=put('partitions','confirmation',Nome='Task41n independent confirmation',Dataset=[ds],Ruolo='Test sigillato',**{'Regola di assegnazione':'All new native/stress seeds; modelseeds17/29/43 remain frozen.','Restrizioni di accesso':'Open only after GPU bitwise eligibility freeze; no fitting,implementation-selection or source repair using fresh outcomes.'})
        protocol=put('protocols','sealed',Nome='Task41n independent native and exact execution protocol',Versione='v1',Modalità='Ricomposizione',Esperimento=[exp],Procedura=json.dumps(spec),**{'Partizioni autorizzate':[test],'Artefatto preregistrazione':[arts[0]],'Hash preregistrazione':digest,'Criteri di successo':spec['qualification'],'Regole di arresto':'Native/canonical/phase floor failure prevents judging model. Any bit difference invalidates that implementation. No training,threshold change or automaticTask42.90minute scientific budget; external backups.','Registrato il':date})
        revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()
        impl=put('implementations','worker',Nome='Task41n isolated CPU/CUDA workers',Commit=revision,**{'Repository URL':'https://github.com/Zagred47/giada','Percorso entrypoint':'scripts/run_task41n_independent.py','Artefatti codice':arts,'Ambiente':'PinnedTorch2.11.0+cu130/T4;NEURON8.2.7/NumPy1.26.4 isolatedPython3.12. ActualGPUpreflight required before fresh access.'})
        aa={}
        for name in names:
            model=put('models',name,Nome='Task41n '+name,Ruolo='Surrogate',Descrizione='Frozen41m coefficients; no trainable change. '+name,**{'Contratto di stato':'Same18occupancies,FP32NN/gates/calcium;V/membranesolveFP64 except dual_fp32 control.','Contratto di input e output':'Same causalV/Ca,state,multipliers,injection/synapse inputs;readout and membrane equations unchanged.'})
            aa[name]=put('arms',name,Nome='Task41n '+name,Ruolo='Baseline' if name in ('dual','H_control','dual_fp32') else 'Trattamento',Protocollo=[protocol],Modello=[model],Implementazione=[impl],Partizioni=[test],**{'Configurazione residua':json.dumps(dict(execution=name,training_updates=0,weights=spec['frozen_input'],bitwise_required=name in spec['execution_variants']))})
            put('runs',name,Nome='Task41n frozen evaluation '+name,Stato='Pianificata',Seed='17,29,43 frozen models; fresh native/release seeds in protocol',Braccio=[aa[name]],**{'Configurazione effettiva':json.dumps(spec),'Validità tecnica':'Local CPU tests passed; actual GPU parity/freeze and native floors required before fresh evaluation.'})
        for control,treatment in [('dual','donor_pruned'),('donor_pruned','shared_features'),('H_control','dual'),('dual_fp32','dual')]:
            put('contrasts',control+'-'+treatment,Nome='Task41n '+control+' vs '+treatment,Protocollo=[protocol],Bracci=[aa[control],aa[treatment]],**{'Contrasto e coefficienti':'Paired case/seed quality; exact trajectory/state/current parity for execution variants; paired wall/CUDA timings.','Soglia interpretativa':'Original quality/cost predicates. Bitwise equivalence stricter than numerical tolerance; never waive mismatches for speed.','Confondenti controllati':'Same fixed weights/inputs/dt for execution; original and matched canonical references. No fresh selection.'})
        for name in ('quality','compartment_gates','cost','parity'):
            metric=put('metrics',name,Nome='Task41n '+name,Formula=json.dumps(spec[name]))
            put('evaluations',name,Nome='Task41n '+name,Versione='v1',Metrica=[metric],Protocollo=[protocol],Partizioni=[test],Ruolo='Primaria' if name in ('quality','compartment_gates') else 'Secondaria',Target=json.dumps(spec[name]))
        put('observations','local-preflight',Nome='Task41n actual CPU tests and exact parity',Descrizione=json.dumps(local),**{'Artefatti dettagliati':arts})
        put('decisions','authorized',Nome='User-approved independent confirmation with exact execution',Esito='Continuare',Risultati=[ref('findings-giada-task41m-result-quality-v1'),ref('findings-giada-task41m-result-separation-v1'),ref('findings-giada-task41m-result-cost-v1')],Motivazione='User explicitly requests DB,one notebook,unchanged thresholds,push and Kaggle/Colab run with external backups. No new training or learned compression.',**{'Data':date})
    print('Staged41n; schema validation before any write',flush=True);change=m.commit_batch()
    if args.validate_only:return
    verification=m.verify_incremental(baseline);assert verification['valid']
    write_json(folder/('launch_memory_receipt.json' if args.running else 'memory_receipt.json'),dict(valid=True,verification=verification,change=change,records=ids,protocol_sha256=digest,registration_before_run=not args.running,airtable_accessed=False,DB_backup='local refresh after launch'))
    m.export_snapshot_to(ROOT/'data/airtable_snapshot.json')
    print('Task41n mirror verified and exported',flush=True)

if __name__=='__main__':
    try:main()
    except PreviewComplete:print('41n schema preview valid;no writes')
