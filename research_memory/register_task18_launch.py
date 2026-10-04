"""Version the real Kaggle submission and its orchestration record; SQLite only."""
import hashlib
import json
from .mirror import Mirror,ROOT,write_json

REVISION='a094fa9b1b9094c48ee816105ee9e0009d9ff0f9'

def main():
    mirror=Mirror();root=ROOT.parent/'experiments/results/task18_kaggle_a094fa9'
    if (root/'final_report.json').exists():raise RuntimeError('Do not reset a completed run')
    root.mkdir(exist_ok=True,parents=True)
    manifest={'code_revision':REVISION,'kernel_id':137030433,'version_number':1,'url':'https://www.kaggle.com/code/alessandrobelli/giada-task18-hh-family-transfer','observed_at_utc':'2026-10-04 11:39:26 UTC','status':'RUNNING','enable_gpu':True,'machine_shape':'NvidiaTeslaT4','is_private':True,'enable_internet':True,'docker_image':'gcr.io/kaggle-private-byod/python@sha256:2757e0c7d1e0a9cb43da657b97e223c321a98f5014bdf64f44f2f6b083ad2b2f','source_revision_readback_verified':True,'dataset_inputs':[]}
    write_json(root/'launch_manifest.json',manifest)
    notebook=json.loads((ROOT.parent/'notebooks/18_roadmap_hh_family_transfer.ipynb').read_text(encoding='utf-8'))
    notebook['cells'].insert(1,{'cell_type':'code','execution_count':None,'metadata':{},'outputs':[],'source':['import os\n',"os.environ['GIADA_CODE_REVISION'] = "+repr(REVISION)+'\n',"os.environ['GIADA_MANUAL_DOWNLOAD'] = '0'\n"]})
    write_json(root/'submitted_notebook.ipynb',notebook)
    artifacts=[]
    for p in sorted(root.iterdir()):
        artifacts.append(mirror.local_upsert('artifacts',{'Codice stabile':'artifacts-giada-roadmap-task18-launch-'+p.stem+'-v1','Nome':'Task18 Kaggle — '+p.name,'Tipo':'Report','Percorso':p.relative_to(ROOT.parent).as_posix(),'SHA-256':hashlib.sha256(p.read_bytes()).hexdigest(),'Dimensione byte':p.stat().st_size,'Versione':'v1'})['record_id'])
    block=mirror.query("SELECT record_id FROM v_records WHERE stable_code='blocks-giada-roadmap-task18-fit-v1'")['rows'][0]['record_id']
    mirror.local_upsert('runs',{'Codice stabile':'runs-giada-roadmap-task18-kaggle-orchestration-v1','Nome':'Task18 Kaggle MCP — orchestrazione 36 modelli','Stato':'In corso','Blocco':[block],'Seed':'17,29,43; dati180101/180201/180301; fresh180401/180402/180403','Hardware e ambiente':'Kaggle NvidiaTeslaT4 GPU; '+manifest['docker_image'],'Configurazione effettiva':'Commit '+REVISION+'; due ensemble indipendenti; 15k passi; nessun input Dataset.','Validità tecnica':'MCP status RUNNING e source revision verificata; risultato scientifico non ancora disponibile.','Artefatti prodotti':artifacts,'Descrizione':manifest['url']+'; kernel137030433; v1. Run aggregato, non attribuito a un singolo braccio.'})
    write_json(ROOT/'data/airtable_snapshot.json',mirror.export_snapshot())
    assert mirror.verify()['valid']
    print(json.dumps({'valid':True,'status':'RUNNING','airtable_accessed':False}))

if __name__=='__main__':main()
