"""Register detailed45b evidence, gate-name correction and45c design, SQLite only."""
import argparse,hashlib,json
from .mirror import ROOT,write_json
from .incremental_batch import IncrementalBatchMirror
from .register_task41d_result import ResultPreview,PreviewComplete

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--validate-only',action='store_true');args=ap.parse_args()
    repo=ROOT.parent;folder=repo/'experiments/results/task45b_audited_v1'
    audit=json.loads((folder/'audit.json').read_text());plan=json.loads((repo/'experiments/task45c_design_v1.json').read_text())
    previous=json.loads((repo/'experiments/results/task45b_protocol_v1/result_memory_receipt.json').read_text())
    assert audit['valid'] and plan['status']=='DESIGN_PREPARED_NOT_EXECUTABLE' and not plan['remote_execution_authorized']
    m=ResultPreview() if args.validate_only else IncrementalBatchMirror();ids={}
    if not args.validate_only:assert m.baseline_hash==previous['verification']['snapshot_hash']
    def put(table,suffix,**fields):
        code=f'{table}-giada-task45b-audit-{suffix}-v1';rid=m.local_upsert(table,{'Codice stabile':code,**fields})['record_id'];ids[code]=rid;return rid
    exp=previous['records']['experiments-giada-task45b-rate-coverage-v1'];source=previous['records']['artifacts-giada-task45b-final-v1']
    data=(folder/'audit.json').read_bytes()
    art=put('artifacts','derived',Nome='45b detailed factorial audit and gate attribution correction',Tipo='Report',Versione='v1',Percorso=str(folder/'audit.json'),**{'SHA-256':hashlib.sha256(data).hexdigest(),'Dimensione byte':len(data)})
    obs=put('observations','derived',Nome='45b all arms, prefixes, channel timing and corrected hidden-state identity',Descrizione=data.decode(),**{'Artefatti dettagliati':[source,art]})
    plan_obs=put('observations','45c-design',Nome='45c design from f011; not yet executable or launched',Descrizione=json.dumps(plan))
    findings={}
    facts={
        'f011':('Ih joint repair improves45 without new41 stress or44 regressions',
            'f011: original45 imposed/autonomous69/72 at100ms,68/72 at500/1000ms versus66/61/64; historical stress66/72;44 all54/54. All unchanged original gates.',
            'Exploratory starting point only. Original45 still four failed cases at1000ms and inherited41 stress six. Not a qualified release.'),
        'nap':('Nap_h rollback fixes tail memory but introduces two nominal stress regressions',
            'Every Nap rollback arm: historical stress64/72 versus66/72; new failing identities seed17 calcium_swing and seed29 near_threshold_staircase. f111 recovers original45 to72/71/71 of72 but not regression-safe.',
            'No rollback arm is automatically promoted. Compensation is a hypothesis supported by the controlled contrast, not a fully proven mechanistic decomposition.'),
        'ih':('Ih amplitude and timing objectives are not interchangeable',
            'Original45 full-horizon autonomous Ih worst nmax3.613% f000,0.350% rate-only,0.051% joint; worst onset/offset9.3ms baseline,13.4ms rate-only,0.9ms joint. Same0.3ms threshold.',
            'One residual joint-repair failure at node1 chain6 weak hyperpolarization seed17. Root/distal-only timing probes omitted that intermediate node;45c must include it.'),
        'gate-correction':('Correct prior assistant misattribution: index3 is Ca_LVAst.h, not NaTs2_t.h',
            json.dumps(audit['correction'])+' Per-gate trace verification: '+json.dumps(audit['corrected_gate_failures']),
            'Interpretation corrected explicitly, no raw results, counts or thresholds rewritten. NaTs2_t.h is index7 and remains a diagnostic negative control in45c.'),
        'coverage':('Tail coverage beat nominal4x refinement budget in45b',
            json.dumps([dict(candidate=r['candidate'],worst=r['worst']) for r in audit['repair_freeze']['all_candidates']]),
            'DEV logit RMSE on declared grid, not automatic behavioral qualification or a universal scaling law. More updates were not monotonically better.'),
        'reference':('45b completed finitely with preserved45 baseline and verified final archive',
            json.dumps(dict(sha=audit['source_archive_sha256'],bytes=audit['source_archive_bytes'],checkpoints=audit['checkpoints'],original45=audit['original45_reproduction'])),
            'All41r original accepted limitations remain; native physical-invalid original45 cases not promoted; no speedup or independent47 claim.')}
    for key,(name,result,limits) in facts.items():
        findings[key]=put('findings',key,Nome=name,Esito='Misto',Esperimenti=[exp],Osservazioni=[obs],Risultato=result,Limitazioni=limits)
    put('lessons','gate-names',Nome='Persist channel-state identity alongside flattened gate index',Lezione='Never infer biological identity by counting rows from memory. Validate STATE_NAMES/OFFSETS against full18-state traces before naming a failure or choosing a training head. This caught a Ca_LVAst.h/NaTs2_t.h interpretation error.',Risultati=[findings['gate-correction']],Limiti='Data and numeric metrics were not lost; the previous narrative attribution was wrong and is now explicitly corrected.')
    put('lessons','functional-preservation',Nome='A global kinetic rollback may recover tail states yet break previously correct event timing',Lezione='Keep nonregressing f011 as starting point. Test rate-only versus joint inf/rate with functional preservation on separate FIT/DEV histories; retain per-identity stress guards. Do not average recovered tail cases against newly lost nominal cases.',Risultati=[findings['nap'],findings['f011']],Limiti='Hypothesis for45c, not proof that the proposed constrained fit will succeed.')
    for h in plan['hypotheses']:
        claim=put('claims','45c-'+h['id'],Nome='45c '+h['id'],Tipo='Ipotesi',Stato='Aperta',Enunciato=h['claim'],Limiti=h['test']+' '+h['discriminator'])
        source_finding=findings['gate-correction'] if h['id']=='gate_identity_negative_control' else findings['nap'] if h['id'].startswith('nap_') else findings['ih'] if h['id'].startswith('ih_') else findings['gate-correction']
        support='Contraddice' if h['id']=='gate_identity_negative_control' else ('Sostiene' if h['id'] in ('nap_compensation','ih_crossing_residual','calva_tail') else 'Indeterminata')
        put('evidence','45c-'+h['id'],Nome='45b evidence informing45c '+h['id'],Esito=support,**{'Affermazione valutata':[claim],'Risultati a sostegno':[source_finding],'Argomentazione':h['discriminator'],'Limiti e spiegazioni alternative':'45c is a design only. Current45b contrasts and named gate identity evidence do not guarantee repair success; mechanisms remain scoped hypotheses.','Data valutazione':'2026-11-11'})
    put('decisions','45c-design',Nome='Prepare45c from f011 with original thresholds; correct actual calcium-gate target',Esito='Continuare',Motivazione='User requested audit recording and45c preparation. Design covers Nap preservation factorial, intermediate-node Ih timing, actual Ca_LVAst.h tail repair and NaTs2_t.h attribution control, plus frozen combination. Numeric loss/transport/executable details pending; no notebook or remote run claimed.',Risultati=list(findings.values()),**{'Data':'2026-11-11'})
    m.commit_batch()
    if args.validate_only:return
    verification=m.verify_incremental(previous['verification']);assert verification['valid']
    write_json(folder/'memory_receipt.json',dict(valid=True,verification=verification,records=ids,airtable_accessed=False,task45c_launched=False,gate_identity_correction_recorded=True))
    m.export_snapshot_to(ROOT/'data/airtable_snapshot.json');print('45b detailed audit, correction and45c design verified/exported in SQLite.',flush=True)

if __name__=='__main__':
    try:main()
    except PreviewComplete:print('45b audit/45c design schema valid; no writes')
