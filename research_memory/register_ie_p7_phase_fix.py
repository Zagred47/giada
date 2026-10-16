"""Archive IE-P7 calibration-only technical stop and fixed-step correction."""
import hashlib
import json
from .mirror import ROOT,write_json
from .validated_batch import ValidatedBatchMirror

def main():
    repo=ROOT.parent;folder=repo/'experiments/results/ie_p7_calibration_blocked_v1'
    report=json.loads((folder/'final_report.json').read_text())
    assert not report['model_judged'] and report['diagnosis']=='NATIVE_INTERFACE_OR_FORMULA_FLOOR_BLOCKED'
    mirror=ValidatedBatchMirror()
    def put(table,suffix,**fields):
        return mirror.local_upsert(table,{'Codice stabile':f'{table}-giada-ie-p7-phase-fix-{suffix}-v1',**fields})['record_id']
    artifacts=[]
    paths=list(folder.iterdir())+[repo/'experiments/ie_p7_event_phase_fix_v1.md']
    for p in paths:
        artifacts.append(put('artifacts',p.name,Nome='IE-P7 event phase '+p.name,Tipo='Report',Versione='v1',Percorso=p.relative_to(repo).as_posix(),**{'SHA-256':hashlib.sha256(p.read_bytes()).hexdigest(),'Dimensione byte':p.stat().st_size}))
    put('findings','diagnosis',Nome='IE-P7 continuous convolution versus native EMS fixed-step phase',Esito='Misto',**{'Risultato':(repo/'experiments/ie_p7_event_phase_fix_v1.md').read_text(),'Limitazioni':'Calibration-only technical diagnosis. No learned candidate judged; no accuracy/speed outcome for IE-P7. No threshold relaxation or training.'})
    put('runs','blocked-calibration',Nome='IE-P7 v1 calibration-only controlled stop',Stato='Completata',**{'Artefatti prodotti':artifacts,'Configurazione effettiva':json.dumps(report),'Validità tecnica':'Controlled preflight stop; native interface and analytic floor fail. Candidate phase not run. Not a scientific no-go of learned models.'})
    mirror.commit_batch();write_json(ROOT/'data/airtable_snapshot.json',mirror.export_snapshot());assert mirror.verify()['valid']
    print(json.dumps({'valid':True,'airtable_accessed':False,'candidate_judged':False}))

if __name__=='__main__':main()
