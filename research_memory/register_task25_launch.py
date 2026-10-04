"""Register the autonomous, pinned Task25 launch in SQLite only."""
import argparse
import hashlib
import json
from pathlib import Path
from .mirror import ROOT,write_json
from .validated_batch import ValidatedBatchMirror


def main():
    p=argparse.ArgumentParser();p.add_argument('manifest');a=p.parse_args();manifest=json.loads(Path(a.manifest).read_text())
    folder=ROOT.parent/'experiments/results'/('task25_kaggle_'+manifest['code_revision'][:7]);folder.mkdir(parents=True,exist_ok=True)
    if (folder/'final_report.json').exists():raise RuntimeError('Do not reset completed experiment')
    write_json(folder/'launch_manifest.json',manifest)
    nb=json.loads((ROOT.parent/'notebooks/25_roadmap_heterogeneous_mechanisms.ipynb').read_text())
    nb['cells'].insert(1,dict(id='task25-pin',cell_type='code',execution_count=None,metadata={},outputs=[],source=['import os\n',"os.environ['GIADA_CODE_REVISION'] = "+repr(manifest['code_revision'])+'\n',"os.environ['GIADA_MANUAL_DOWNLOAD'] = '0'\n"]))
    write_json(folder/'submitted_notebook.ipynb',nb)
    m=ValidatedBatchMirror();artifacts=[]
    for path in folder.iterdir():
        artifacts.append(m.local_upsert('artifacts',{'Codice stabile':'artifacts-giada-roadmap-task25-launch-'+path.stem+'-v1','Nome':'Task25 '+path.name,'Tipo':'Report','Percorso':path.relative_to(ROOT.parent).as_posix(),'SHA-256':hashlib.sha256(path.read_bytes()).hexdigest(),'Dimensione byte':path.stat().st_size,'Versione':'v1'})['record_id'])
    block=m.query("SELECT record_id FROM v_records WHERE stable_code='blocks-giada-roadmap-task25-fresh-v1'")['rows'][0]['record_id']
    m.local_upsert('runs',{'Codice stabile':'runs-giada-roadmap-task25-kaggle-orchestration-v1','Nome':'Task25 Kaggle MCP — frozen2x2','Stato':'In corso','Blocco':[block],'Seed':'17,29,43','Hardware e ambiente':'Kaggle '+manifest['machine_shape'],'Configurazione effettiva':'Commit '+manifest['code_revision']+';4frozenbundles;no training','Validità tecnica':'Pin verified; scientific outcome pending','Artefatti prodotti':artifacts,'Descrizione':manifest['url']})
    m.local_upsert('experiments',{'Codice stabile':'experiments-giada-roadmap-task25-matrix-v1','Stato':'In corso'})
    m.commit_batch();write_json(ROOT/'data/airtable_snapshot.json',m.export_snapshot());assert m.verify()['valid']
    print(json.dumps(dict(valid=True,status=manifest['status'],airtable_accessed=False)))


if __name__=='__main__':main()
