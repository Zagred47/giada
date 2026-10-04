"""Generate aligned Task25 from the tested isolated-worker template."""
import json
from pathlib import Path
root=Path(__file__).resolve().parents[1]
nb=json.loads((root/'notebooks/24b_sodium_control_diagnosis.ipynb').read_text())
for cell in nb['cells']:
    cell['id']=cell['id'].replace('task24b','task25')
    cell['source']=[line.replace('Task24b','Task25').replace('task24b','task25').replace('sodium_control_diagnosis','heterogeneous_mechanisms').replace('primary_repair_passed','composition_passed').replace('task25_preparation_authorized','task26_preparation_authorized').replace('training_and_evaluation_seconds','evaluation_seconds') for line in cell['source']]
nb['cells'][0]['source']=['# 🔬 GIADA Task25 — meccanismi eterogenei\n','Cinque meccanismi calcio/sodio, dieci stati espliciti. Matrice2×2 di moduli congelati:4bundle×3seed. Nessun dataset da montare, nessun training.\n']
nb['cells'][2]['source']=['## 🧪 Composizione, non ancora modello joint\n','Stessi nuovi input e percorsi per tutti: errori per canale e corrente totale, cancellazione, identità, held1–10000ms, traiettorie a0.025/1ms, probeFP64 e formula/LUT non selezionabili. V imposto: non è un neurone autonomo. Gate primario: controllo indipendente3/3. Task26–28 restano da fare.\n']
(root/'notebooks/25_roadmap_heterogeneous_mechanisms.ipynb').write_text(json.dumps(nb,ensure_ascii=False,indent=1)+'\n',encoding='utf-8',newline='\n')
