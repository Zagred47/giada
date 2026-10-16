"""Record partial IE-P7 v2 outputs and the implementation-only compile repair."""
import hashlib
import json
from .mirror import ROOT,write_json
from .validated_batch import ValidatedBatchMirror

def main():
    repo=ROOT.parent;folder=repo/'experiments/results/ie_p7_kaggle_v2'
    failure=json.loads((folder/'gpu_failure.json').read_text())
    timing=json.loads((folder/'timing_progress.json').read_text())
    assert not failure['scientific_no_go'] and len(timing['rows'])==56 and not timing['graph_failures']
    mirror=ValidatedBatchMirror()
    def put(table,suffix,**fields):
        return mirror.local_upsert(table,{'Codice stabile':f'{table}-giada-ie-p7-compile-fix-{suffix}-v1',**fields})['record_id']
    artifacts=[]
    for p in list(folder.iterdir())+[repo/'experiments/ie_p7_compile_fix_v1.md']:
        artifacts.append(put('artifacts',p.name,Nome='IE-P7 v2 '+p.name,Tipo='Report',Versione='v2',Percorso=p.relative_to(repo).as_posix(),**{'SHA-256':hashlib.sha256(p.read_bytes()).hexdigest(),'Dimensione byte':p.stat().st_size}))
    put('runs','partial',Nome='IE-P7 v2 partial results and operational compile failure',Stato='Completata',**{'Artefatti prodotti':artifacts,'Configurazione effettiva':json.dumps(failure),'Validità tecnica':'Native and fresh evaluations completed; 56 core timing cells, zero graph failures. Staged operational compilation failed, so no complete speed decision.'})
    put('findings','diagnosis',Nome='IE-P7 immutable NumPy area metadata inside compiled frame',Esito='Misto',**{'Risultato':(repo/'experiments/ie_p7_compile_fix_v1.md').read_text(),'Limitazioni':'No complete operational speed result. No model promotion, retraining or threshold change; real GPU compilation preflights mandatory in v3.'})
    mirror.commit_batch();write_json(ROOT/'data/airtable_snapshot.json',mirror.export_snapshot());assert mirror.verify()['valid']
    print(json.dumps({'valid':True,'airtable_accessed':False}))

if __name__=='__main__':main()
