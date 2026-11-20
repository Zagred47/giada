import gzip,hashlib,json,zipfile
from .mirror import ROOT,write_json
from .incremental_batch import IncrementalBatchMirror
def main():
    repo=ROOT.parent;old=json.loads((repo/'experiments/results/task52_protocol_v1/preregistration_receipt.json').read_text());folder=repo/'artifacts/task52_kaggle_external';dest=repo/'experiments/results/task52_audit_v1'
    receipt=json.loads((folder/'kaggle_receipt.json').read_text());archive=folder/receipt['archive']
    with archive.open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==receipt['sha256']
    m=IncrementalBatchMirror();assert m.baseline_hash==old['verification']['snapshot_hash'];ids={}
    def put(table,key,**fields):
        code=f'{table}-giada-task52-audit-{key}-v1';rid=m.local_upsert(table,{'Codice stabile':code,**fields})['record_id'];ids[code]=rid;return rid
    observations=[]
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for name in z.namelist():
            if not(name.endswith('.json') or name.endswith('.json.gz')):continue
            raw=z.read(name);text=gzip.decompress(raw).decode() if name.endswith('.gz') else raw.decode();target=dest/(name if name.endswith('.gz') else name+'.gz')
            with gzip.open(target,'wt',encoding='utf-8') as f:f.write(text)
            art=put('artifacts',name,Nome='52 original '+name,Tipo='Report',Versione='v1',Percorso=target.relative_to(repo).as_posix(),**{'SHA-256':hashlib.sha256(target.read_bytes()).hexdigest(),'Dimensione byte':target.stat().st_size})
            observations.append(put('observations',name,Nome='52 original '+name,Descrizione=text,**{'Artefatti dettagliati':[art]}))
    for filename in ('audit.json','fixed_model_probe.json','kaggle_receipt.original.json','kaggle_receipt.corrected.json','runtime_receipt.corrected.json'):
        observations.append(put('observations',filename,Nome='52 audit '+filename,Descrizione=(dest/filename).read_text()))
    exp=old['records']['experiments-giada-task52-closed-loop-v1']
    finding=put('findings','results',Nome='52 readout transfer passes; compact block fails',Esito='Misto',Esperimenti=[exp],Osservazioni=observations,Risultato='At1000ms full-state canonical/P/P_J controls36/36. Compact0/36, pooledV6.677mV; V/Ca imposed currents stillfail; Ca-only forcedV6.688mV. OracleCaV3.249,Na4.920,K6.906,Ih6.357,SK6.699; all-oracle36/36. Reset negligible. Frozen decoder bound alone prevents36/36 trajectories. Sampled transition Jacobian max singular0.0255655.',Limitazioni='Old L10 is not calibrated51j readout. Readout requires18physical gates. Local Jacobian not global theorem; no newNEURON validation; no whole-compartment speedup. Original wrapper receipts falsely say5000updates; corrected receipts explicitly0, archive preserved.')
    m.local_upsert('experiments',{'Stato':'Concluso'},record_id=exp)
    for key,rid in old['records'].items():
        if key.startswith('runs-'):m.local_upsert('runs',{'Stato':'Completata'},record_id=rid)
    # Correct the protocol label; no thresholds, methods or original evidence change.
    m.local_upsert('protocols',{'Nome':'52 frozen compressed-state closed-loop matrix'},record_id=old['records']['protocols-giada-task52-closed-loop-v1'])
    design=json.loads((repo/'experiments/task52_failure_hypotheses_v1.json').read_text())
    proposals=[]
    for h in design['hypotheses']:proposals.append(put('claims',h['id'],Nome='52 followup '+h['id'],Tipo='Ipotesi',Stato='Aperta',Enunciato=h['claim'],Limiti=h['contrast']))
    previous=[rid for key,rid in old['records'].items() if key.startswith('claims-')]
    m.local_upsert('experiments',{'Ipotesi':previous+proposals},record_id=exp)
    put('observations','proposed-matrix',Nome='52 proposed four-arm decoder-memory matrix',Descrizione=json.dumps(design))
    put('lessons','range-before-state',Nome='Check output feasibility before blaming latent initialization',Lezione='A fixed decoder lower bound can rule out success for every latent initialization. Separate representability of frozen parameters from architecture learnability. Diagnose recurrence with full state Jacobian, not direct update coefficient alone.',Risultati=[finding],Limiti='No general impossibility claim about10dimensions.')
    put('decisions','proposal-only',Nome='Propose controlled52followup; no new run authorized',Esito='Continuare',Motivazione='User requested failure analysis only. Four paired decoder x memory-initialization arms proposed, plus nonselectable initialization probes; keep thresholds and abstraction.',Risultati=[finding])
    print('52 audit and hypotheses staged',flush=True);m.commit_batch();verification=m.verify_incremental(old['verification']);assert verification['valid']
    write_json(dest/'memory_receipt.json',dict(valid=True,verification=verification,records=ids,airtable_accessed=False));m.export_snapshot_to(ROOT/'data/airtable_snapshot.json');print('52 audit DB verified',flush=True)
if __name__=='__main__':main()
