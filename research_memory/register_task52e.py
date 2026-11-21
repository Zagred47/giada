import argparse,json
from .mirror import ROOT,write_json
from .incremental_batch import IncrementalBatchMirror
from .register_task41d_result import ResultPreview,PreviewComplete
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--validate-only',action='store_true');args=ap.parse_args();repo=ROOT.parent
    old=json.loads((repo/'experiments/results/task52d_audit_v1/memory_receipt.json').read_text());p=json.loads((repo/'experiments/task52e_preregistration_v1.json').read_text());probe=json.loads((repo/'experiments/results/task52e_protocol_v1/decoder_preflight.json').read_text());m=ResultPreview() if args.validate_only else IncrementalBatchMirror();ids={}
    if not args.validate_only:assert m.baseline_hash==old['verification']['snapshot_hash']
    def put(table,key,**fields):
        code=f'{table}-giada-task52e-{key}-v1';rid=m.local_upsert(table,{'Codice stabile':code,**fields})['record_id'];ids[code]=rid;return rid
    exp=put('experiments','support-family',Nome='52e support x protected family objectives',Tipo='Esplorativo',Stato='Preregistrato',Descrizione=json.dumps(p),**{'Obiettivo informativo':'Separate targeted training support and family-resolved objective after fit-only decoder diagnosis. Same logical10state abstraction and scientific thresholds.'})
    proto=put('protocols','factorial',Nome='52e paired four-arm protocol',Versione='v1',Modalità='Ricomposizione',Esperimento=[exp],Procedura=json.dumps(p),**{'Criteri di successo':'Original52quality and intervention gates, plus common development non-regression guard. Initial52d checkpoint is fallback, not new success.','Regole di arresto':'Stop on nonfinite/hash/encoder drift. Preserve checkpoints; no automatic promotion.'})
    claims=[]
    for key,claim in [('support','Targeted canonical histories improve slow and transient current learning.'),('family','Family-resolved objective improves channel fidelity without sacrificing autonomous behavior.'),('interaction','Targeted support and family objective interact beneficially.')]:claims.append(put('claims',key,Nome='52e '+key,Tipo='Ipotesi',Stato='Aperta',Enunciato=claim))
    put('experiments','support-family',Ipotesi=claims)
    for arm in p['arms']:
        br=put('arms',arm['id'],Nome='52e '+arm['id'],Ruolo='Trattamento',Protocollo=[proto]);put('runs',arm['id'],Nome='52e '+arm['id'],Stato='Pianificata',Braccio=[br],Seed='17,29,43 paired')
    obs=put('observations','decoder-probe',Nome='52e fit-only frozen decoder probe',Descrizione=json.dumps(probe))
    put('findings','decoder',Nome='Ih pointwise readout available but encoded state inaccurate',Esito='Misto',Esperimenti=[exp],Osservazioni=[obs],Risultato='Ih scalar grid max absolute log10error<=0.000666 across36fit points, versus median encoded log10error around1.0. Free-latent fit improves many channels. Weights unchanged, no fresh access.',Limitazioni='Only sampled fit states. No proof of global representability or recurrent learnability. Do not unfreeze encoder or add physical states; decoder remains trainable as registered.')
    put('decisions','proceed',Nome='Proceed52e after diagnostic decoder check',Esito='Continuare',Motivazione='User authorized four-arm notebook, DB, push andprivateKaggle execution. Local CPUpreflight andremoteGPU assertion required. External checkpoint/result backups, DBbackup local only.')
    print('52e DB staged',flush=True);m.commit_batch()
    if args.validate_only:return
    v=m.verify_incremental(old['verification']);assert v['valid'];folder=repo/'experiments/results/task52e_protocol_v1';write_json(folder/'preregistration_receipt.json',dict(valid=True,verification=v,records=ids,airtable_accessed=False));m.export_snapshot_to(ROOT/'data/airtable_snapshot.json');print('52e DB verified',flush=True)
if __name__=='__main__':
    try:main()
    except PreviewComplete:print('52e schema valid')
