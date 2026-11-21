import argparse,hashlib,json
from .mirror import ROOT,write_json
from .incremental_batch import IncrementalBatchMirror
from .register_task41d_result import ResultPreview,PreviewComplete
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--validate-only',action='store_true');args=ap.parse_args();repo=ROOT.parent
    old=json.loads((repo/'experiments/results/task52c_audit_v1/memory_receipt.json').read_text());p=json.loads((repo/'experiments/task52d_preflight_v1.json').read_text());folder=repo/'experiments/results/task52d_protocol_v1';result=json.loads((folder/'decoder_feasibility.json').read_text());assert result['weights_unchanged'] and result['point_witness_total']==300
    m=ResultPreview() if args.validate_only else IncrementalBatchMirror();ids={}
    if not args.validate_only:assert m.baseline_hash==old['verification']['snapshot_hash']
    def put(table,key,**fields):
        code=f'{table}-giada-task52d-preflight-{key}-v1';rid=m.local_upsert(table,{'Codice stabile':code,**fields})['record_id'];ids[code]=rid;return rid
    exp=put('experiments','decoder',Nome='52d frozen decoder feasibility preflight',Tipo='Esplorativo',Stato='Concluso',Descrizione=json.dumps(p),**{'Obiettivo informativo':'Check whether recurrence-only repair is justified before GPU training.'})
    protocol=put('protocols','decoder',Nome='52d free-latent diagnostic witnesses',Versione='v1',Modalità='Ricomposizione',Esperimento=[exp],Procedura=json.dumps(p),**{'Criteri di successo':'Find current_nmax<=0.05witness for every sampled point; this remains weaker than temporal feasibility.','Regole di arresto':'Do not label failed local optimization as impossibility; no model update or threshold change.'})
    obs=put('observations','result',Nome='52d full decoder feasibility output',Descrizione=json.dumps(result))
    witness=repo/'artifacts/task52d_decoder_witnesses.pt'
    art=put('artifacts','witness',Nome='52d nondeployable free-latent witnesses',Tipo='Report',Versione='v1',Percorso=witness.relative_to(repo).as_posix(),**{'SHA-256':hashlib.sha256(witness.read_bytes()).hexdigest(),'Dimensione byte':witness.stat().st_size})
    put('observations','witness-scope',Nome='52d witness scope',Descrizione='Free latent state per point; no temporal consistency, model training, selection or deployment eligibility.',**{'Artefatti dettagliati':[art]})
    finding=put('findings','not-established',Nome='52d decoder feasibility remains unverified',Esito='Misto',Esperimenti=[exp],Osservazioni=[obs],Risultato='210/300representative seed-point checks have a numerical current witness.90remain unresolved; largest normalized current error0.997 versus0.05limit. Broad intervals contain targets but do not prove simultaneous reachability. Weights unchanged; no GPU training.',Limitazioni='Not proof that the decoder is mathematically impossible. Search limits and joint output constraints remain. Not a recurrent model acceptance result.')
    put('decisions','await-direction',Nome='Do not launch recurrence-only matrix before resolving decoder contract',Esito='Continuare',Motivazione='Continue diagnosis only. Four arms remain planned, not launched. User asked whether to allow recalibration or deepen frozen decoder checks; no silent unfreezing or threshold/abstraction change.',Risultati=[finding])
    print('52d preflight staged',flush=True);m.commit_batch()
    if args.validate_only:return
    v=m.verify_incremental(old['verification']);assert v['valid'];write_json(folder/'memory_receipt.json',dict(valid=True,verification=v,records=ids,airtable_accessed=False));m.export_snapshot_to(ROOT/'data/airtable_snapshot.json');print('52d preflight DB verified',flush=True)
if __name__=='__main__':
    try:main()
    except PreviewComplete:print('52d preflight schema valid')
