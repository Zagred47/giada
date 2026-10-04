"""Mechanical generation from the verified Task22 orchestration template."""
import json
from pathlib import Path

root=Path(__file__).resolve().parents[1]
n=json.loads((root/'notebooks/22_roadmap_controlled_gate_sharing.ipynb').read_text(encoding='utf-8'))
for c in n['cells']:
    if 'id' in c:c['id']=c['id'].replace('task22','task23')
    c['source']=[s.replace('Task22','Task23').replace('task22','task23').replace('controlled_sharing','calcium_pair') for s in c['source']]
n['cells'][0]['source']=['# 🧩 GIADA Task23 originale — prima composizione calcio\n','Ca-HVA + Ca-LVA: riferimento indipendente, trunk con head separate, identità condizionata. 18 sistemi vettorizzati; nessun dataset da montare.\n']
n['cells'][2]['source']=['## 🔬 Contratto\n','Audit Ca-LVA/Ca-HVA contro NEURON prima del training. Gate, correnti individuali e somma; due width e budget crescenti; freeze prima di fresh. Voltaggio imposto, non dinamica autonoma. Baseline LUT sulle transizioni; percorsi imposti e rollout held-V sui candidati appresi.\n']
n['cells'][3]['source']=[s.replace("'task23_authorized'","'task24_preparation_authorized'") for s in n['cells'][3]['source']]
(root/'notebooks/23_roadmap_calcium_pair_composition.ipynb').write_text(json.dumps(n,ensure_ascii=False,indent=1)+'\n',encoding='utf-8',newline='\n')
