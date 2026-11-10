"""Typed45 audit and45b proposed hypotheses; no launch or training authority."""
import argparse,json,hashlib
from .mirror import ROOT,write_json
from .incremental_batch import IncrementalBatchMirror
from .register_task41d_result import ResultPreview,PreviewComplete

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--validate-only',action='store_true');args=ap.parse_args()
    repo=ROOT.parent;folder=repo/'experiments/results/task45_audited_v1'
    audit=json.loads((folder/'audit.json').read_text());proposal=json.loads((repo/'experiments/task45b_proposed_contrasts_v1.json').read_text())
    previous=json.loads((repo/'experiments/results/task45_protocol_v1/result_memory_receipt.json').read_text())
    assert audit['valid'] and not proposal['execution_authorized'] and not proposal['training_authorized']
    m=ResultPreview() if args.validate_only else IncrementalBatchMirror();ids={}
    if not args.validate_only:assert m.baseline_hash==previous['verification']['snapshot_hash']
    def put(table,suffix,**fields):
        code=f'{table}-giada-task45-audit-{suffix}-v1';rid=m.local_upsert(table,{'Codice stabile':code,**fields})['record_id'];ids[code]=rid;return rid
    exp=previous['records']['experiments-giada-task45-coverage-v1'];artifact45=previous['records']['artifacts-giada-task45-final-v1']
    data=(folder/'audit.json').read_bytes()
    art=put('artifacts','derived',Nome='45 detailed audit and immutable-weight coefficient probes',Tipo='Report',Versione='v1',Percorso=str(folder/'audit.json'),**{'SHA-256':hashlib.sha256(data).hexdigest(),'Dimensione byte':len(data)})
    obs=put('observations','audit',Nome='45 native-domain audit, horizon failures and coefficient counterfactuals',Descrizione=data.decode('utf-8'),**{'Artefatti dettagliati':[artifact45,art]})
    proposed=put('observations','45b-proposal',Nome='45b proposed contrasts, not executed',Descrizione=json.dumps(proposal))
    domain=put('findings','domain',Nome='45 reference numerical floors pass but nine physical-regime controls fail',Esito='Misto',Esperimenti=[exp],Osservazioni=[obs],Risultato='Native teacher reaches -148.647958mV in disconnected hyperpolarization, below unchanged -140mV physical bound. Do not attribute these nine model-seed checks to the network. Preserve them; calibrate future protocol with teacher only.',Limitazioni='Not a software or native reconstruction failure. Global reference decision blocked;72 other checks per mode remain interpretable, not retroactively all-pass.')
    outcome=put('findings','interpretable-failures',Nome='45 genuine Ih/Nap_h gaps persist with valid references',Esito='Misto',Esperimenti=[exp],Osservazioni=[obs],Risultato=json.dumps(dict(horizons=audit['horizons'],geometries=audit['geometries'],failed_cases=audit['full_horizon_failed_cases'],gate_failures=audit['full_horizon_gate_failures'])),Limitazioni='Same failed identities imposed/autonomous; gate and current failures matter despite small V errors.1000ms cannot waive500ms failures. Opened diagnostics, no independent qualification.')
    rate=put('findings','rate-counterfactual',Nome='Nap_h rate-map regression isolated in frozen coefficient probe; Ih rate bias pre-exists',Esito='Misto',Esperimenti=[exp],Osservazioni=[obs],Risultato=json.dumps(proposal['evidence']),Limitazioni='Constant-voltage coefficient diagnostics, not proof that a rollback passes coupled regressions. Local hat features vanish outside their support but the full MLP still receives voltage. FP32 shadow only tests recurrence error with fixed coefficients.')
    put('lessons','rate-head-regression',Nome='A repair can preserve equilibrium but destroy a slow rate outside opened support',Lezione='Probe both x_inf and effective tau before/after learned-head repairs, with tail-voltage and long-horizon guards. At -106mV seed29 Nap_h prior tau1378.9ms versus teacher1321.2ms, current41r tau.5917ms while x_inf remains accurate. Do not infer correct memory from small output or one-step error.',Risultati=[rate,outcome],Limiti='Scoped frozen probes; old quality gains may regress under rollback and must be retested.')
    put('lessons','domain-vs-learning',Nome='Separate physical protocol validity from real learned coverage failures',Lezione='Do not train to repair an out-of-domain reference. Conversely, invalid cases do not erase real failures in valid subsets. Calibrate new stimuli without candidate outcomes and retain prior valid cases unchanged.',Risultati=[domain,outcome],Limiti='No acceptance thresholds changed; original45 retained.')
    for hypothesis in proposal['hypotheses']:
        claim=put('claims',hypothesis['id'],Nome='45b '+hypothesis['id'],Tipo='Ipotesi',Stato='Aperta',Enunciato=hypothesis['claim'],Limiti=hypothesis['test']+' '+hypothesis['discriminator'])
        selected=domain if hypothesis['id']=='protocol_domain' else rate
        support='Sostiene' if hypothesis['id'] in ('protocol_domain','nap_rate_regression','ih_rate_bias') else 'Indeterminata'
        put('evidence',hypothesis['id'],Nome='45 evidence for proposed '+hypothesis['id'],Esito=support,**{'Affermazione valutata':[claim],'Risultati a sostegno':[selected],'Argomentazione':hypothesis['discriminator'],'Limiti e spiegazioni alternative':'Proposal not executed. Training cause, rollback regression safety and full varying-path behavior remain to test. Constant-clamp evidence has its stated narrow scope.','Data valutazione':'2026-11-10'})
    put('decisions','proposed-only',Nome='45b plan proposed after45 audit; no execution authorized',Esito='Continuare',Motivazione='User requested DB recording and hypothesis analysis only. Proposed native calibration, coefficient probes, learned readout controls and constrained support/budget tests require later explicit implementation/launch approval. Original thresholds and abstraction preserved.',Risultati=[domain,outcome,rate],**{'Data':'2026-11-10'})
    m.commit_batch()
    if args.validate_only:return
    verification=m.verify_incremental(previous['verification']);assert verification['valid']
    write_json(folder/'memory_receipt.json',dict(valid=True,verification=verification,records=ids,airtable_accessed=False,task45b_launched=False))
    m.export_snapshot_to(ROOT/'data/airtable_snapshot.json');print('45 audit and45b proposed hypotheses verified in SQLite.',flush=True)
if __name__=='__main__':
    try:main()
    except PreviewComplete:print('45 audit/proposal schema valid; no writes')
