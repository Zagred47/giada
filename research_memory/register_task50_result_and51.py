"""Local-only complete50 evidence and prospective51 factorial protocol."""
import argparse,gzip,hashlib,json,zipfile
from .mirror import ROOT,write_json
from .incremental_batch import IncrementalBatchMirror
from .register_task41d_result import ResultPreview,PreviewComplete
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--validate-only',action='store_true');args=ap.parse_args();repo=ROOT.parent;old=json.loads((repo/'experiments/results/task50_protocol_v1/preregistration_receipt.json').read_text());audit=json.loads((repo/'experiments/results/task50_audit_v1/audit.json').read_text());p=json.loads((repo/'experiments/task51_preregistration_v1.json').read_text());archive=repo/'artifacts/task50_kaggle_external/task50_final.zip'
    with archive.open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==audit['archive_sha256']
    m=ResultPreview() if args.validate_only else IncrementalBatchMirror();base={} if args.validate_only else old['verification'];ids={}
    if not args.validate_only:assert m.baseline_hash==base['snapshot_hash']
    def put(table,task,key,**fields):
        code=f'{table}-giada-task{task}-{key}-v1';rid=m.local_upsert(table,{'Codice stabile':code,**fields})['record_id'];ids[code]=rid;return rid
    exp=old['records']['experiments-giada-task50-compact-memory-v1'];obs=[];dest=repo/'experiments/results/task50_audit_v1'
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for name in z.namelist():
            if not(name.endswith('.json') or name.endswith('.json.gz')):continue
            raw=z.read(name);text=gzip.decompress(raw).decode() if name.endswith('.gz') else raw.decode();target=dest/(name if name.endswith('.gz') else name+'.gz')
            with gzip.open(target,'wt',encoding='utf-8') as f:f.write(text)
            art=put('artifacts','50-result',name,Nome='50 full '+name,Tipo='Report',Versione='v1',Percorso=target.relative_to(repo).as_posix(),**{'SHA-256':hashlib.sha256(target.read_bytes()).hexdigest(),'Dimensione byte':target.stat().st_size})
            obs.append(put('observations','50-result',name,Nome='50 raw '+name,Descrizione=text,**{'Artefatti dettagliati':[art]}))
    obs.append(put('observations','50-result','audit',Nome='50 full accuracy/budget/initialization/cost audit',Descrizione=json.dumps(audit)))
    finding=put('findings','50-result','failure',Nome='50 all compact arms fail quality despite development improvement',Esito='Negativo',Esperimenti=[exp],Osservazioni=obs,Risultato=json.dumps(audit['summary']),Limitazioni=' '.join(audit['limitations']))
    put('findings','50-result','learning-and-cost',Nome='50 loss improves but does not transfer; light packed execution is not qualified speedup',Esito='Misto',Esperimenti=[exp],Osservazioni=obs,Risultato=json.dumps(dict(scaling=audit['scaling'],reset=audit['reset_control'],cost=audit['cost_median_ms'],initial_probe=audit['posthoc_steady_initial_readout_probe'])),Limitazioni='Training6.4ms, development25.6ms versusfresh1000ms. Initial outputs already badly calibrated. No latent trajectories/encoder timing saved. Packed15versus3replicas; padding prevents actual state storage reduction.')
    put('experiments','50','compact-memory',Stato='Concluso')
    for arm in ['L5_T16','L10_T16','L10_T32','L18_T16','L18_T32']:put('runs','50',arm,Stato='Completata')
    put('lessons','50-result','loss-not-quality',Nome='50 development loss decrease is not sufficient accuracy or memory evidence',Lezione='Start with initial readout and calibration before attributing catastrophic errors to slow memory. Separate training normalization from evaluation thresholds, state dimension from trunk and active logical dimensions from padded storage. Use matched seeds and explicit interactions.',Risultati=[finding],Limiti='No impossibility proof for nonlinear compression; no budget-only explanation established.')
    e=put('experiments','51','diagnosis',Nome='Task51 initial/loss/memory factorial diagnosis',Tipo='Esplorativo',Stato='Preregistrato',Descrizione=json.dumps(p),**{'Obiettivo informativo':'Distinguish initial supervision, compound current calibration, temporal training coverage and interactions at fixed architecture.'})
    protocol=put('protocols','51','diagnosis',Nome='51 paired2x2x2 matrix fixedL10/T16',Versione='v1',Modalità='Ricomposizione',Esperimento=[e],Procedura=json.dumps(p),**{'Criteri di successo':'Same quality thresholds per case, unchangedP, development-only selection; probe initial errors and memory separately.','Regole di arresto':'Nonfinite parameters, changed frozen hash or budget expiry: preserve partial checkpoint; no relaxed gate.'})
    claims=[put('claims','51',h['id'],Nome='51 '+h['id'],Tipo='Ipotesi',Stato='Aperta',Enunciato=h['claim'],Limiti=h['contrast']) for h in p['hypotheses']];put('experiments','51','diagnosis',Ipotesi=claims)
    for arm in p['arms']:
        br=put('arms','51',arm['id'],Nome='51 '+arm['id'],Ruolo='Trattamento',Protocollo=[protocol]);put('runs','51',arm['id'],Nome='51 '+arm['id'],Stato='Pianificata',Braccio=[br],Seed='17,29,43; paired initial weights and stream, independent parameters')
    put('decisions','51','proceed',Nome='Proceed51 causal factorial diagnosis',Esito='Continuare',Motivazione='Explicit user authorization: all contrasts in one notebook, same thresholds, frozenP, private Kaggle and external backups.',Risultati=[finding],**{'Data':'2026-11-15'})
    print('50 full results and51 protocol staged',flush=True);m.commit_batch()
    if args.validate_only:return
    v=m.verify_incremental(base);assert v['valid'];folder=repo/'experiments/results/task51_protocol_v1';folder.mkdir(exist_ok=True);receipt=dict(valid=True,verification=v,records=ids,airtable_accessed=False);write_json(folder/'preregistration_receipt.json',receipt);write_json(dest/'memory_receipt.json',receipt);m.export_snapshot_to(ROOT/'data/airtable_snapshot.json');print('50/51 DB verified',flush=True)
if __name__=='__main__':
    try:main()
    except PreviewComplete:print('50/51 schema valid')
