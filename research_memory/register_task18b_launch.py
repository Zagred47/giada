"""Register the actual Task18b Kaggle submission, without touching Airtable."""
import argparse,hashlib,json
from pathlib import Path
from .mirror import Mirror,ROOT,write_json

def main():
    parser=argparse.ArgumentParser();parser.add_argument('manifest');args=parser.parse_args()
    manifest=json.loads(Path(args.manifest).read_text(encoding='utf-8'))
    root=ROOT.parent/'experiments/results'/('task18b_kaggle_'+manifest['code_revision'][:7]);root.mkdir(parents=True,exist_ok=True)
    if (root/'final_report.json').exists():raise RuntimeError('Do not reset completed result')
    write_json(root/'launch_manifest.json',manifest)
    notebook=json.loads((ROOT.parent/'notebooks/18b_roadmap_potassium_diagnosis.ipynb').read_text(encoding='utf-8'))
    notebook['cells'].insert(1,{'cell_type':'code','execution_count':None,'metadata':{},'outputs':[],'source':['import os\n',"os.environ['GIADA_CODE_REVISION'] = "+repr(manifest['code_revision'])+'\n',"os.environ['GIADA_MANUAL_DOWNLOAD'] = '0'\n"]})
    write_json(root/'submitted_notebook.ipynb',notebook)
    mirror=Mirror();artifacts=[]
    for p in sorted(root.iterdir()):
        artifacts.append(mirror.local_upsert('artifacts',{'Codice stabile':'artifacts-giada-task18b-launch-'+p.stem+'-v1','Nome':'Task18b Kaggle — '+p.name,'Tipo':'Report','Percorso':p.relative_to(ROOT.parent).as_posix(),'SHA-256':hashlib.sha256(p.read_bytes()).hexdigest(),'Dimensione byte':p.stat().st_size,'Versione':'v1'})['record_id'])
    block=mirror.query("SELECT record_id FROM v_records WHERE stable_code='blocks-giada-task18b-fit-v1'")['rows'][0]['record_id']
    mirror.local_upsert('runs',{'Codice stabile':'runs-giada-task18b-kaggle-orchestration-v1','Nome':'Task18b Kaggle MCP — 24 modelli','Stato':'In corso','Blocco':[block],'Seed':'17,29,43; dati181101/181102; fresh181401/181402/181403','Hardware e ambiente':'Kaggle GPU '+manifest['machine_shape'],'Configurazione effettiva':'Commit '+manifest['code_revision']+';24 modelli indipendenti;60k passi; nessun input Dataset.','Validità tecnica':'Submission verificata; esito scientifico non ancora disponibile.','Artefatti prodotti':artifacts,'Descrizione':manifest['url']})
    write_json(ROOT/'data/airtable_snapshot.json',mirror.export_snapshot());assert mirror.verify()['valid']
    print(json.dumps({'valid':True,'status':manifest['status'],'airtable_accessed':False}))

if __name__=='__main__':main()
