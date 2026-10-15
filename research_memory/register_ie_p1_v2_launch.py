"""Close the technical v1 incident and register its prospective v2 retry."""
import hashlib
import json
from .mirror import ROOT, write_json
from .validated_batch import ValidatedBatchMirror


def main():
    repo=ROOT.parent
    mirror=ValidatedBatchMirror()
    read=lambda p: json.loads((repo/p).read_text(encoding='utf-8'))
    launch=read('experiments/ie_p1_kaggle_launch_v2.json')
    amendment=read(launch['amendment'])
    assert launch['run_requested'] and read(amendment['prior_result'])['valid'] is False
    def put(table,code,**fields):
        return mirror.local_upsert(table,{'Codice stabile':code,**fields})['record_id']
    put('runs','runs-giada-ie-p1-kaggle-v1',Stato='Fallita',**{
        'Validità tecnica':'CLOSED TECHNICAL FAILURE: compiled/eager preflight blocked batch642. Partial batch1 timings are not a completed benchmark. No candidate scientific NO-GO.'})
    ids=[]
    for path in ('experiments/ie_p1_kaggle_launch_v2.json',launch['amendment'],amendment['prior_result']):
        f=repo/path
        ids.append(put('artifacts','artifacts-giada-ie-p1-v2-'+f.stem.replace('_','-'),
            Nome='IE-P1 v2 '+f.name,Tipo='Report',Percorso=path,Versione='v2',**{
                'SHA-256':hashlib.sha256(f.read_bytes()).hexdigest(),'Dimensione byte':f.stat().st_size}))
    protocol=put('protocols','protocols-giada-ie-p1-compile-v2',Nome='IE-P1 prospective compile amendment v2',
        Versione='v2',Modalità='Ricomposizione',Procedura=json.dumps(amendment,ensure_ascii=False),**{
            'Criteri di successo':'Reuse unchanged CC-N1 compiler FP32 gates; eager decomposition and scientific quality criteria unchanged.',
            'Hash preregistrazione':hashlib.sha256((repo/launch['amendment']).read_bytes()).hexdigest()})
    arm=put('arms','arms-giada-ie-p1-v2-execution',Nome='IE-P1 v2 prospective numerical preflight',
        Ruolo='Oracle diagnostico',Protocollo=[protocol],**{'Configurazione residua':json.dumps(launch)})
    put('runs','runs-giada-ie-p1-kaggle-v2',Nome='IE-P1 Kaggle v2',Stato='In corso',Braccio=[arm],**{
        'Artefatti prodotti':ids,'Configurazione effettiva':json.dumps(launch),
        'Validità tecnica':'Retry requested after preserved v1 failure; metrics pending; no scientific promotion.'})
    latest=repo/'experiments/ie_p1_kaggle_launch_v3.json'
    if latest.is_file():
        third=json.loads(latest.read_text(encoding='utf-8'))
        put('runs','runs-giada-ie-p1-kaggle-v2',Stato='Fallita',**{
            'Validità tecnica':'CLOSED SUBMISSION FAILURE: revision injection removed setup imports, NameError subprocess before experiment. Scientific source unchanged; v3 restores original imports.'})
        artifact=put('artifacts','artifacts-giada-ie-p1-v3-launch',Nome='IE-P1 v3 submission correction',
            Tipo='Report',Percorso=latest.relative_to(repo).as_posix(),Versione='v3',**{
                'SHA-256':hashlib.sha256(latest.read_bytes()).hexdigest(),'Dimensione byte':latest.stat().st_size})
        put('runs','runs-giada-ie-p1-kaggle-v3',Nome='IE-P1 Kaggle v3',Stato='In corso',Braccio=[arm],**{
            'Artefatti prodotti':[artifact],'Configurazione effettiva':json.dumps(third),
            'Validità tecnica':'v3 requested from repository notebook with all setup imports preserved; scientific outcome pending.'})
    mirror.commit_batch()
    write_json(ROOT/'data/airtable_snapshot.json',mirror.export_snapshot())
    assert mirror.verify()['valid']
    print(json.dumps({'valid':True,'v1_technical_failure_preserved':True,'v2_launch_registered':True,'airtable_accessed':False}))


if __name__=='__main__':main()
