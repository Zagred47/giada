"""Register complete saved52d evidence and proposed52e hypotheses; local only."""
import argparse,gzip,hashlib,json,zipfile
from .mirror import ROOT,write_json
from .incremental_batch import IncrementalBatchMirror
from .register_task41d_result import ResultPreview,PreviewComplete

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--validate-only',action='store_true');ap.add_argument('--refine-proposal',action='store_true');args=ap.parse_args();repo=ROOT.parent
    if args.refine_proposal:
        dest=repo/'experiments/results/task52d_audit_v1';receipt=json.loads((dest/'memory_receipt.json').read_text());design=json.loads((repo/'experiments/task52e_hypotheses_v1.json').read_text());m=IncrementalBatchMirror()
        assert m.baseline_hash==receipt['verification']['snapshot_hash']
        m.local_upsert('observations',{'Descrizione':json.dumps(design)},record_id=receipt['records']['observations-giada-task52d-audit-52e-proposal-v1'])
        h=next(h for h in design['hypotheses'] if h['id']=='objective_interference')
        m.local_upsert('claims',{'Limiti':h['test']+' Discriminator: '+h['discriminator']},record_id=receipt['records']['claims-giada-task52d-audit-objective_interference-v1'])
        h=next(h for h in design['hypotheses'] if h['id']=='compensating_currents')
        code='claims-giada-task52d-audit-compensating_currents-v1'
        receipt['records'][code]=m.local_upsert('claims',{'Codice stabile':code,'Nome':'52e proposed compensating_currents','Tipo':'Ipotesi','Stato':'Aperta','Enunciato':h['claim'],'Limiti':h['test']+' Discriminator: '+h['discriminator']})['record_id']
        m.commit_batch();v=m.verify_incremental(receipt['verification']);assert v['valid'];receipt['verification']=v;receipt['proposal_refined_common_selection']=True;write_json(dest/'memory_receipt.json',receipt);m.export_snapshot_to(ROOT/'data/airtable_snapshot.json');print('Common selection refinement verified',flush=True);return
    old=json.loads((repo/'experiments/results/task52d_protocol_v2/preregistration_receipt.json').read_text())
    dest=repo/'experiments/results/task52d_audit_v1';audit=json.loads((dest/'audit.json').read_text());design=json.loads((repo/'experiments/task52e_hypotheses_v1.json').read_text())
    receipt=json.loads((repo/'artifacts/task52d_kaggle_external/kaggle_receipt.json').read_text());archive=repo/'artifacts/task52d_kaggle_external'/receipt['archive']
    with archive.open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==receipt['sha256']
    m=ResultPreview() if args.validate_only else IncrementalBatchMirror();ids={};obs=[]
    if not args.validate_only:assert m.baseline_hash==old['verification']['snapshot_hash']
    def put(table,key,**fields):
        code=f'{table}-giada-task52d-audit-{key}-v1';rid=m.local_upsert(table,{'Codice stabile':code,**fields})['record_id'];ids[code]=rid;return rid
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for name in z.namelist():
            if not(name.endswith('.json') or name.endswith('.json.gz')):continue
            raw=z.read(name);text=gzip.decompress(raw).decode() if name.endswith('.gz') else raw.decode()
            target=dest/(name if name.endswith('.gz') else name+'.gz');target.parent.mkdir(parents=True,exist_ok=True)
            if not target.exists():
                with gzip.open(target,'wt',encoding='utf-8') as f:f.write(text)
            art=put('artifacts',name,Nome='52d original '+name,Tipo='Report',Versione='v1',Percorso=target.relative_to(repo).as_posix(),**{'SHA-256':hashlib.sha256(target.read_bytes()).hexdigest(),'Dimensione byte':target.stat().st_size})
            obs.append(put('observations',name,Nome='52d original '+name,Descrizione=text,**{'Artefatti dettagliati':[art]}))
    for name in ['audit.json','residuals.json']:
        obs.append(put('observations',name,Nome='52d '+name,Descrizione=(dest/name).read_text()))
    exp=old['records']['experiments-giada-task52d-v2-balance-trajectory']
    finding=put('findings','result',Nome='52d autonomous objective transfers but current fidelity remains unresolved',Esito='Misto',Esperimenti=[exp],Osservazioni=obs,Risultato='Completed800updates,10checkpointfiles finite, SHA/CRC verified final archive. Allarms0/36strict. V1s52cDE0.902732;control0.923743;balance0.939695;trajectory0.730414;combined0.732370mV. Trajectory improves36/36Vcases, net-current imposedRMSE-13.76percent, but calcium+0.425percent and family regressions. Ih suppressed atinitialsample and imposedinputs, many missingcurrentevents. Alltrajectoryseedsselect300. Atomic speed4.88xFP64 not whole-neuron speed.',Limitazioni=design['limitations'])
    if not args.validate_only:
        m.local_upsert('experiments',{'Stato':'Concluso'},record_id=exp)
        for key,rid in old['records'].items():
            if key.startswith('runs-'):m.local_upsert('runs',{'Stato':'Completata'},record_id=rid)
    for h in design['hypotheses']:
        put('claims',h['id'],Nome='52e proposed '+h['id'],Tipo='Ipotesi',Stato='Aperta',Enunciato=h['claim'],Limiti=h['test']+' Discriminator: '+h['discriminator'])
    put('observations','52e-proposal',Nome='52e proposed support x protected-family repair matrix',Descrizione=json.dumps(design))
    put('lessons','trajectory-not-sufficiency',Nome='Autonomous voltage improvement does not ensure ionic fidelity',Lezione='Keep missing current events and initial amplitude errors visible alongside voltage. Compare imposed inputs before attributing all errors to feedback. Additional no-regression selection constraints do not relax original gates.',Risultati=[finding],Limiti='Exploratory reused panel; not independent confirmation or proof of decoder infeasibility.')
    put('decisions','analysis-only',Nome='Propose52e without launching',Esito='Continuare',Motivazione='User requested DBregistration and hypotheses. Preserve thresholds and10state abstraction; freeze52dtrajectory as exploratory baseline. Proposed four arms require subsequent implementation instruction.',Risultati=[finding])
    m.commit_batch()
    if args.validate_only:return
    v=m.verify_incremental(old['verification']);assert v['valid'];write_json(dest/'memory_receipt.json',dict(valid=True,verification=v,records=ids,airtable_accessed=False));m.export_snapshot_to(ROOT/'data/airtable_snapshot.json');print('52d audit and52e hypotheses DBverified',flush=True)
if __name__=='__main__':
    try:main()
    except PreviewComplete:print('52d audit schema valid')
