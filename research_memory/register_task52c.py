import argparse,json
from .mirror import ROOT,write_json
from .incremental_batch import IncrementalBatchMirror
from .register_task41d_result import ResultPreview,PreviewComplete
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--validate-only',action='store_true');args=ap.parse_args();repo=ROOT.parent
    old=json.loads((repo/'experiments/results/task52b_audit_v1/memory_receipt.json').read_text());p=json.loads((repo/'experiments/task52c_preregistration_v1.json').read_text())
    m=ResultPreview() if args.validate_only else IncrementalBatchMirror();ids={}
    if not args.validate_only:assert m.baseline_hash==old['verification']['snapshot_hash']
    def put(table,key,**fields):
        code=f'{table}-giada-task52c-{key}-v1';rid=m.local_upsert(table,{'Codice stabile':code,**fields})['record_id'];ids[code]=rid;return rid
    exp=put('experiments','component-repair',Nome='52c decoder-first encoder x recurrence repair',Tipo='Esplorativo',Stato='Preregistrato',Descrizione=json.dumps(p),**{'Obiettivo informativo':'D/DE/DR/DER conditional on common calibrated decoder, distinguish encoder and recurrent repair. Frozen52b and passive negative control; same latent10 and quality gates.'})
    proto=put('protocols','factorial',Nome='52c paired component-repair matrix',Versione='v1',Modalità='Ricomposizione',Esperimento=[exp],Procedura=json.dumps(p),**{'Criteri di successo':'Original52/46 thresholds, current/timing and autonomous outcomes. No success from passive-like voltage or training loss alone.','Regole di arresto':'Nonfinite/hash/frozen-component error stops. Preserve200/800/2400checkpoints and final external backup.'})
    claims=[put('claims',h['id'],Nome='52c '+h['id'],Tipo='Ipotesi',Stato='Aperta',Enunciato=h['claim'],Limiti=h['contrast']) for h in p['hypotheses']];put('experiments','component-repair',Ipotesi=claims)
    for arm in p['arms']:
        br=put('arms',arm['id'],Nome='52c '+arm['id'],Ruolo='Trattamento',Protocollo=[proto]);put('runs',arm['id'],Nome='52c '+arm['id'],Stato='Pianificata',Braccio=[br],Seed='17,29,43 independent paired members')
    put('observations','causal-boundaries',Nome='52c inference boundaries',Descrizione='Same calibrated decoder and fitting objective in all4 arms. DE-D,DER-DR isolate encoder; DR-D,DER-DE isolate recurrence; full interaction retained. Comparisons to52b confound decoder capacity/calibration/objective/sampling and must not be called a pure decoder effect. Future-assisted h0, teacher-state oracle and tiny-fit probes nonselectable. No extra online inputs, no analytic gates restored, no change of evaluation thresholds. Memory backup local only.')
    put('decisions','authorization',Nome='Authorize52c four-arm test',Esito='Continuare',Motivazione='User says proceed with the proposed matrix. Prepare, register, push and run privateKaggle with external checkpoint/result recovery; DB backup local. No automatic promotion to53.')
    print('52c schema staged',flush=True);m.commit_batch()
    if args.validate_only:return
    v=m.verify_incremental(old['verification']);assert v['valid'];folder=repo/'experiments/results/task52c_protocol_v1';folder.mkdir(exist_ok=True);write_json(folder/'preregistration_receipt.json',dict(valid=True,verification=v,records=ids,airtable_accessed=False));m.export_snapshot_to(ROOT/'data/airtable_snapshot.json');print('52c DB verified',flush=True)
if __name__=='__main__':
    try:main()
    except PreviewComplete:print('52c schema valid')
