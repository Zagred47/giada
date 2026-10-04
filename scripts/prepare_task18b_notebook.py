"""Generate Task18b notebook mechanically from tested Task18 supervisor template."""
import json
from pathlib import Path
root=Path(__file__).resolve().parents[1]
n=json.loads((root/'notebooks/18_roadmap_hh_family_transfer.ipynb').read_text(encoding='utf-8'))
for c in n['cells']:
    if c['cell_type']=='code':
        c['source']=[line.replace('giada_task18_hh_transfer_','giada_task18b_potassium_').replace('giada_task18_','giada_task18b_').replace('run_roadmap_task18.py','run_roadmap_task18b.py').replace("'transfer_passed','per_channel_transfer'","'potassium_passed','selected'") for line in c['source']]
n['cells'][0]['source']=['# 🧬 GIADA Task18b — diagnosi K_Pst\n','Copertura coda negativa × supervisione tau × capacità; tre seed, checkpoint15k/30k/60k. 24 modelli vettorizzati. Nessun Dataset da montare. Audit NEURON e CUDA in processi separati. Preregistrazione: `experiments/task18b_potassium_diagnosis.json`.']
n['cells'][2]['source']=['## ⚗️ Matrice e freeze\n','Campionamento uniforme o25% di tuple nella coda[-135,-115]mV; peso logtau0.01/0.1; width32/64. Stessi stati, dt e indici appaiati. Una configurazione comune ai tre seed è congelata tramite development prima del nuovo test. Soglie della Task18 invariate; la coda interna ha un gate separato. OOD diagnostico.']
(root/'notebooks/18b_roadmap_potassium_diagnosis.ipynb').write_text(json.dumps(n,ensure_ascii=False,indent=1)+'\n',encoding='utf-8')
