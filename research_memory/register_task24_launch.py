"""Persist the pinned autonomous Kaggle submission; SQLite only."""
import argparse
import hashlib
import json
from pathlib import Path
from .mirror import ROOT,write_json
from .validated_batch import ValidatedBatchMirror

def main():
    p=argparse.ArgumentParser();p.add_argument('manifest');a=p.parse_args()
    manifest=json.loads(Path(a.manifest).read_text(encoding='utf-8'))
    root=ROOT.parent/'experiments/results'/('task24_kaggle_'+manifest['code_revision'][:7]);root.mkdir(parents=True,exist_ok=True)
    if (root/'final_report.json').exists():raise RuntimeError('Do not reset completed result')
    write_json(root/'launch_manifest.json',manifest)
    nb=json.loads((ROOT.parent/'notebooks/24_roadmap_sodium_family_composition.ipynb').read_text(encoding='utf-8'))
    nb['cells'].insert(1,dict(id='task24-pin',cell_type='code',execution_count=None,metadata={},outputs=[],source=['import os\n',"os.environ['GIADA_CODE_REVISION'] = "+repr(manifest['code_revision'])+'\n',"os.environ['GIADA_MANUAL_DOWNLOAD'] = '0'\n"]))
    write_json(root/'submitted_notebook.ipynb',nb)
    m=ValidatedBatchMirror();artifacts=[]
    for f in root.iterdir():
        artifacts.append(m.local_upsert('artifacts',{'Codice stabile':'artifacts-giada-roadmap-task24-launch-'+f.stem+'-v1','Nome':'Task24 '+f.name,'Tipo':'Report','Percorso':f.relative_to(ROOT.parent).as_posix(),'SHA-256':hashlib.sha256(f.read_bytes()).hexdigest(),'Dimensione byte':f.stat().st_size,'Versione':'v1'})['record_id'])
    block=m.query("SELECT record_id FROM v_records WHERE stable_code='blocks-giada-roadmap-task24-fit-v1'")['rows'][0]['record_id']
    m.local_upsert('runs',{'Codice stabile':'runs-giada-roadmap-task24-kaggle-orchestration-v1','Nome':'Task24 Kaggle MCP — 18sistemi','Stato':'In corso','Blocco':[block],'Seed':'17,29,43; data240xxx','Hardware e ambiente':'Kaggle '+manifest['machine_shape'],'Configurazione effettiva':'Commit '+manifest['code_revision']+';18sistemi;30ksteps','Validità tecnica':'Pin verificato; esito scientifico ancora da valutare','Artefatti prodotti':artifacts,'Descrizione':manifest['url']})
    exp=m.query("SELECT record_id FROM v_records WHERE stable_code='experiments-giada-roadmap-task24-matrix-v1'")['rows'][0]['record_id']
    m.local_upsert('experiments',{'Codice stabile':'experiments-giada-roadmap-task24-matrix-v1','Stato':'In corso'})
    m.commit_batch();write_json(ROOT/'data/airtable_snapshot.json',m.export_snapshot());assert m.verify()['valid']
    print(json.dumps({'valid':True,'status':manifest['status'],'airtable_accessed':False}))

if __name__=='__main__':main()
