"""Record autonomous Task27 Kaggle launch in the local mirror."""
import hashlib
import json
from .mirror import ROOT,write_json
from .validated_batch import ValidatedBatchMirror


def main():
    repo=ROOT.parent
    folder=repo/'experiments/results/task27_kaggle_c072e69'
    manifest=json.loads((folder/'launch_manifest.json').read_text(encoding='utf-8'))
    if (folder/'final_report.json').exists():raise RuntimeError('Do not reset completed Task27')
    notebook=json.loads((repo/'notebooks/27_roadmap_current_supervision.ipynb').read_text(encoding='utf-8'))
    notebook['cells'].insert(1,dict(id='task27-pin',cell_type='code',execution_count=None,metadata={},outputs=[],
        source=['import os\n',"os.environ['GIADA_CODE_REVISION'] = "+repr(manifest['code_revision'])+'\n',
                "os.environ['GIADA_MANUAL_DOWNLOAD'] = '0'\n"]))
    write_json(folder/'submitted_notebook.ipynb',notebook)
    mirror=ValidatedBatchMirror();artifacts=[]
    for path in folder.iterdir():
        artifacts.append(mirror.local_upsert('artifacts',{'Codice stabile':'artifacts-giada-roadmap-task27-launch-'+path.stem+'-v1',
            'Nome':'Task27 '+path.name,'Tipo':'Report','Percorso':path.relative_to(repo).as_posix(),
            'SHA-256':hashlib.sha256(path.read_bytes()).hexdigest(),'Dimensione byte':path.stat().st_size,'Versione':'v1'})['record_id'])
    block=mirror.query("SELECT record_id FROM v_records WHERE stable_code='blocks-giada-roadmap-task27-fresh-v1'")['rows'][0]['record_id']
    mirror.local_upsert('runs',{'Codice stabile':'runs-giada-roadmap-task27-kaggle-orchestration-v1',
        'Nome':'Task27 Kaggle MCP — current supervision','Stato':'In corso','Blocco':[block],
        'Seed':'17,29,43','Hardware e ambiente':'Kaggle '+manifest['machine_shape'],
        'Configurazione effettiva':'Commit '+manifest['code_revision']+';2famiglie x4loss;3seed;ladder60k',
        'Validità tecnica':'Pin and source verified; scientific outcome pending','Artefatti prodotti':artifacts,
        'Descrizione':manifest['url']})
    mirror.local_upsert('experiments',{'Codice stabile':'experiments-giada-roadmap-task27-matrix-v1','Stato':'In corso'})
    mirror.commit_batch();write_json(ROOT/'data/airtable_snapshot.json',mirror.export_snapshot())
    assert mirror.verify()['valid']
    print(json.dumps({'valid':True,'status':manifest['status'],'airtable_accessed':False}))


if __name__=='__main__':main()
