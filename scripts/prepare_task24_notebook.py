"""Generate Task24 from the validated Task23 supervisor notebook template."""
import json
from pathlib import Path
root=Path(__file__).resolve().parents[1]
n=json.loads((root/'notebooks/23_roadmap_calcium_pair_composition.ipynb').read_text(encoding='utf-8'))
for cell in n['cells']:
    cell['id']=cell['id'].replace('task23','task24')
    cell['source']=[line.replace('Task23','Task24').replace('task23','task24').replace('calcium_pair','sodium_family').replace('task24_preparation_authorized','task25_preparation_authorized') for line in cell['source']]
n['cells'][0]['source']=['# 🧩 GIADA Task24 originale — famiglia sodio\n','NaTa_t + NaTs2_t + Nap_Et2: reti indipendenti, trunk condiviso con head separate, rete condizionata. 18 sistemi vettorizzati, nessun dataset da montare.\n']
n['cells'][2]['source']=['## 🔬 Contratto\n','Audit dei tre canali contro NEURON prima del training. Sei gate, correnti individuali e somma; due width, budget crescenti, freeze prima di fresh. Voltaggio imposto, non dinamica autonoma. Percorsi rapidi (25ms) e lenti (2500ms), held-V fino a10s per Nap. LUT come riferimento numerico delle transizioni; latenza frozen misurata separatamente, nessuna selezione sui tempi.\n']
(root/'notebooks/24_roadmap_sodium_family_composition.ipynb').write_text(json.dumps(n,ensure_ascii=False,indent=1)+'\n',encoding='utf-8',newline='\n')
