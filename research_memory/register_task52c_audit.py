import argparse,gzip,hashlib,json,zipfile
from .mirror import ROOT,write_json
from .incremental_batch import IncrementalBatchMirror
from .register_task41d_result import ResultPreview,PreviewComplete
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--validate-only',action='store_true');args=ap.parse_args();repo=ROOT.parent
    old=json.loads((repo/'experiments/results/task52c_protocol_v1/preregistration_receipt.json').read_text());dest=repo/'experiments/results/task52c_audit_v1';receipt=json.loads((repo/'artifacts/task52c_kaggle_external/kaggle_receipt.json').read_text());archive=repo/'artifacts/task52c_kaggle_external'/receipt['archive']
    with archive.open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==receipt['sha256']
    m=ResultPreview() if args.validate_only else IncrementalBatchMirror();ids={};obs=[]
    if not args.validate_only:assert m.baseline_hash==old['verification']['snapshot_hash']
    def put(table,key,**fields):
        code=f'{table}-giada-task52c-audit-{key}-v1';rid=m.local_upsert(table,{'Codice stabile':code,**fields})['record_id'];ids[code]=rid;return rid
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for name in z.namelist():
            if not(name.endswith('.json') or name.endswith('.json.gz')):continue
            raw=z.read(name);text=gzip.decompress(raw).decode() if name.endswith('.gz') else raw.decode();target=dest/(name if name.endswith('.gz') else name+'.gz');target.parent.mkdir(parents=True,exist_ok=True)
            if not target.exists():
                with gzip.open(target,'wt',encoding='utf-8') as f:f.write(text)
            art=put('artifacts',name,Nome='52c original '+name,Tipo='Report',Versione='v1',Percorso=target.relative_to(repo).as_posix(),**{'SHA-256':hashlib.sha256(target.read_bytes()).hexdigest(),'Dimensione byte':target.stat().st_size})
            obs.append(put('observations',name,Nome='52c original '+name,Descrizione=text,**{'Artefatti dettagliati':[art]}))
    obs.append(put('observations','audit',Nome='52c full paired audit',Descrizione=(dest/'audit.json').read_text()))
    exp=old['records']['experiments-giada-task52c-component-repair-v1']
    finding=put('findings','result',Nome='52c marginal current improvement does not transfer to net current',Esito='Misto',Esperimenti=[exp],Osservazioni=obs,Risultato='2400updates complete, all finite,0/36strict cases perarm. AutonomousV D0.911/DE0.903/DR1.237/DER1.578mV. Net-current error already worsens on imposed inputs by~22percent D->DR and36percent DE->DER; relaxation dominates. Ca_HVA algebraic covariance contribution is largest despite lower individual RMSE. No general increase in sampled maximum local V gain. Latent coordinates shift with recurrence learning. All select final2400; convergence and validation alignment unresolved.',Limitazioni='Algebraic covariance is not causal replacement evidence. Local derivatives not full stability. Shift relative to calibration not proof of out-of-support. No promotion or new experiment authorized by this audit.')
    if not args.validate_only:
        m.local_upsert('experiments',{'Stato':'Concluso'},record_id=exp)
        for key,rid in old['records'].items():
            if key.startswith('runs-'):m.local_upsert('runs',{'Stato':'Completata'},record_id=rid)
    design=json.loads((repo/'experiments/task52c_failure_hypotheses_v1.json').read_text())
    for h in design['hypotheses']:put('claims',h['id'],Nome='52c followup '+h['id'],Tipo='Ipotesi',Stato='Aperta',Enunciato=h['claim'],Limiti=h['contrast']+' Falsifier: '+h['falsifier'])
    put('observations','proposal',Nome='52c proposed balance x trajectory objective matrix',Descrizione=json.dumps(design))
    put('lessons','net-versus-marginal',Nome='Distinguish channel marginals from net membrane forcing',Lezione='Lower marginal channel errors can produce worse net-current residual through timing/sign/covariance changes. Inspect imposed net current before blaming feedback; use nonselectable channel replacements to test algebraic attribution.',Risultati=[finding],Limiti='Retain individual-channel acceptance gates to reject artificial cancellation.')
    put('decisions','analysis-only',Nome='Propose next controlled recurrence diagnosis',Esito='Continuare',Motivazione='User requested registration and analysis only; unchanged thresholds/abstraction. Proposed four-arm balance x autonomous-trajectory loss matrix conditional on anchored encoder/readout; no run launched.',Risultati=[finding])
    print('52c audit staged',flush=True);m.commit_batch()
    if args.validate_only:return
    v=m.verify_incremental(old['verification']);assert v['valid'];write_json(dest/'memory_receipt.json',dict(valid=True,verification=v,records=ids,airtable_accessed=False));m.export_snapshot_to(ROOT/'data/airtable_snapshot.json');print('52c audit DB verified',flush=True)
if __name__=='__main__':
    try:main()
    except PreviewComplete:print('52c audit schema valid')
