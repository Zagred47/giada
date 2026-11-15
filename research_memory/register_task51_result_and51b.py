"""Full51 archive and static51b protocol, one verified local-only batch."""
import argparse,gzip,hashlib,json,zipfile
from .mirror import ROOT,write_json
from .incremental_batch import IncrementalBatchMirror
from .register_task41d_result import ResultPreview,PreviewComplete
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--validate-only',action='store_true');args=ap.parse_args();repo=ROOT.parent;old=json.loads((repo/'experiments/results/task51_protocol_v1/preregistration_receipt.json').read_text());p=json.loads((repo/'experiments/task51b_preregistration_v1.json').read_text());archive=repo/'artifacts/task51_kaggle_external/task51_final.zip';expected=json.loads((archive.parent/'kaggle_receipt.json').read_text())
    with archive.open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==expected['sha256']
    m=ResultPreview() if args.validate_only else IncrementalBatchMirror();base={} if args.validate_only else old['verification'];ids={}
    if not args.validate_only:assert m.baseline_hash==base['snapshot_hash']
    def put(table,task,key,**fields):
        code=f'{table}-giada-task{task}-{key}-v1';rid=m.local_upsert(table,{'Codice stabile':code,**fields})['record_id'];ids[code]=rid;return rid
    exp=old['records']['experiments-giada-task51-diagnosis-v1'];obs=[];dest=repo/'experiments/results/task51_audit_v1';dest.mkdir(exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for name in z.namelist():
            if not(name.endswith('.json') or name.endswith('.json.gz')):continue
            raw=z.read(name);text=gzip.decompress(raw).decode() if name.endswith('.gz') else raw.decode();target=dest/(name if name.endswith('.gz') else name+'.gz')
            with gzip.open(target,'wt',encoding='utf-8') as f:f.write(text)
            art=put('artifacts','51-result',name,Nome='51 full '+name,Tipo='Report',Versione='v1',Percorso=target.relative_to(repo).as_posix(),**{'SHA-256':hashlib.sha256(target.read_bytes()).hexdigest(),'Dimensione byte':target.stat().st_size})
            obs.append(put('observations','51-result',name,Nome='51 raw '+name,Descrizione=text,**{'Artefatti dettagliati':[art]}))
    result=put('findings','51-result','factorial',Nome='51 calibration helps but all compact arms fail',Esito='Negativo',Esperimenti=[exp],Osservazioni=obs,Risultato='All8arms0/2592; P1848/2592. Calibration reduces pooled medianNRMSE from119.56to60.46; startup anchor alone72.49. Longer102.4ms almost no benefit. All selected600. Parameters finite. Reset difference mostly erased by10ms; A1B0 retained some200ms, none1000ms.',Limitazioni='Pooled horizons not independent. Different loss/development vs50. No minimal-state or impossibility claim; longer training factor retainsTBPTT6.4ms. No qualified speedup:24packed vs3replicas, poor accuracy.')
    put('experiments','51','diagnosis',Stato='Concluso')
    for a in ['A0_B0_C0','A1_B0_C0','A0_B1_C0','A1_B1_C0','A0_B0_C1','A1_B0_C1','A0_B1_C1','A1_B1_C1']:put('runs','51',a,Stato='Completata')
    put('lessons','51-result','isolate-readout',Nome='51 isolate tiny opening readout before memory',Lezione='Strong startup error and poor small-current calibration persist after longer training segments. Isolate the static physical-state/readout map with analytic controls before blaming recurrent memory. Loss calibration helps but not enough; anchor and calibration not additive in this run.',Risultati=[result],Limiti='No evidence all representation options are impossible.')
    e=put('experiments','51b','small-openings',Nome='Task51b static small-opening representation',Tipo='Esplorativo',Stato='Preregistrato',Descrizione=json.dumps(p),**{'Obiettivo informativo':'Disentangle small-positive-output parameterization and objective/sampling from temporal recurrence.'})
    protocol=put('protocols','51b','small-openings',Nome='51b3x2staticreadout matrix',Versione='v1',Modalità='Ricomposizione',Esperimento=[e],Procedura=json.dumps(p),**{'Criteri di successo':'Unchanged current amplitude thresholds; decade diagnostics separately; controls exact where mathematically applicable. No temporal release.','Regole di arresto':'Nonfinite/checksum/control failure: preserve artifacts; no threshold changes.'})
    claims=[put('claims','51b',h['id'],Nome='51b '+h['id'],Tipo='Ipotesi',Stato='Aperta',Enunciato=h['claim'],Limiti=h['contrast']) for h in p['hypotheses']];put('experiments','51b','small-openings',Ipotesi=claims)
    for rep in p['representations']:
        for loss in p['losses']:
            key=rep+'_'+loss;br=put('arms','51b',key,Nome='51b '+key,Ruolo='Trattamento',Protocollo=[protocol]);put('runs','51b',key,Nome='51b '+key,Stato='Pianificata',Braccio=[br],Seed='17,29,43; same matrices perseed; fit-only calibration')
    put('decisions','51b','proceed',Nome='Proceed51b static diagnostics',Esito='Continuare',Motivazione='User approved isolating tiny openings from temporal dynamics, same thresholds, Kaggle and external backups.',Risultati=[result],**{'Data':'2026-11-15'})
    print('51 full results and51b protocol staged',flush=True);m.commit_batch()
    if args.validate_only:return
    v=m.verify_incremental(base);assert v['valid'];folder=repo/'experiments/results/task51b_protocol_v1';folder.mkdir(exist_ok=True);receipt=dict(valid=True,verification=v,records=ids,airtable_accessed=False);write_json(folder/'preregistration_receipt.json',receipt);write_json(dest/'memory_receipt.json',receipt);m.export_snapshot_to(ROOT/'data/airtable_snapshot.json');print('51/51b DB verified',flush=True)
if __name__=='__main__':
    try:main()
    except PreviewComplete:print('51/51b schema valid')
