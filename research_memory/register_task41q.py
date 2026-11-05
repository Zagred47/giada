"""Record post-hoc41q CPU forensic audit; no invented prospective claims."""
import argparse,json,hashlib
from .mirror import ROOT,write_json
from .incremental_batch import IncrementalBatchMirror
from .register_task41d_result import ResultPreview,PreviewComplete

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--validate-only',action='store_true');args=ap.parse_args()
    repo=ROOT.parent;folder=repo/'experiments/results/task41q_audit_v1';plan=json.loads((repo/'experiments/task41q_preregistration_v1.json').read_text())
    report=json.loads((folder/'audit.json').read_text());summary=json.loads((folder/'decision_summary.json').read_text());backup=json.loads((folder/'external_backup_receipt.json').read_text())
    assert report['valid'] and report['complete'] and backup['valid']
    m=ResultPreview() if args.validate_only else IncrementalBatchMirror();ids={}
    baseline={} if args.validate_only else json.loads((repo/'experiments/results/task41p_repair_v1/result_memory_receipt.json').read_text())['verification']
    if not args.validate_only:assert m.baseline_hash==baseline['snapshot_hash']
    def put(table,suffix,**fields):
        code=f'{table}-giada-task41q-{suffix}-v1';rid=m.local_upsert(table,{'Codice stabile':code,**fields})['record_id'];ids[code]=rid;return rid
    exp=put('experiments','audit',Nome='Task41q existing-checkpoint objective and readout audit',Tipo='Forensic',Stato='Concluso',Descrizione=json.dumps(plan),**{'Obiettivo informativo':'Distinguish objective conflict, unused same-feature optimization room and numerical materialization; no GPU/new stochastic training or re-selection.'})
    hypotheses=[]
    for name,text in plan['hypotheses'].items():
        hypotheses.append(put('claims',name,Nome='Task41q post-hoc '+name,Tipo='Ipotesi',Stato='Indeterminata',Enunciato=text,**{'Limiti':'Post-hoc forensic alternatives, not prospectively preregistered predictions. Evidence and limits in linked result; no global capacity proof or deployed repair.'}))
    put('experiments','audit',**{'Ipotesi':hypotheses})
    art=put('artifacts','bundle',Nome='Task41q verified local forensic bundle',Tipo='Report',Versione='v1',Percorso=backup['local'],**{'SHA-256':backup['sha256'],'Dimensione byte':backup['bytes']})
    protocol=put('protocols','posthoc',Nome='Task41q post-hoc audit plan',Versione='v1',Modalità='Ricomposizione',Esperimento=[exp],Procedura=json.dumps(plan),**{'Criteri di successo':'Source/fit-byte verification, all checkpoint objectives, independent-anchor coefficient errors, archived timing comparisons and bounded same-head diagnostic fits; no model qualification.','Regole di arresto':'Source/checkpoint identity or nonfinite failure; do not retrain or edit original gates.'})
    obs=put('observations','raw',Nome='Task41q complete numerical forensic audit',Descrizione=(folder/'audit.json').read_text(),**{'Artefatti dettagliati':[art]})
    put('observations','summary',Nome='Task41q verified quantitative decision summary',Descrizione=json.dumps(summary),**{'Artefatti dettagliati':[art]})
    for key in ('objective_conflict','same_feature_room','recommended_next_step','limits'):
        put('findings',key,Nome='Task41q '+key,Esito='Misto',Esperimenti=[exp],Osservazioni=[obs],Risultato=json.dumps(summary[key]),Limitazioni='CPU coefficient audit, not GPU behavioral requalification. All direct fits diagnostic; original FP32 failures218/234 retained. No new selection or Task42 authorization.')
    m.commit_batch()
    if args.validate_only:return
    verification=m.verify_incremental(baseline);assert verification['valid']
    write_json(folder/'memory_receipt.json',dict(valid=True,verification=verification,records=ids,airtable_accessed=False,posthoc_analysis=True))
    m.export_snapshot_to(ROOT/'data/airtable_snapshot.json');print('Task41q mirror verified/exported',flush=True)
if __name__=='__main__':
    try:main()
    except PreviewComplete:print('Task41q schema preview valid; no writes')
