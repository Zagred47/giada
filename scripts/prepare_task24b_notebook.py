"""Reuse the successful bounded-output native/GPU supervisor notebook."""
import json
from pathlib import Path
root=Path(__file__).resolve().parents[1]
n=json.loads((root/'notebooks/24_roadmap_sodium_family_composition.ipynb').read_text())
for cell in n['cells']:
    cell['id']=cell['id'].replace('task24','task24b')
    cell['source']=[line.replace('Task24','Task24b').replace('task24','task24b').replace('sodium_family','sodium_control_diagnosis').replace("'sharing_confirmed','per_family_passed'","'primary_repair_passed','per_arm_passed'") for line in cell['source']]
n['cells'][0]['source']=['# 🔬 GIADA Task24b — diagnosi del controllo sodio\n','Sottotask della Task24 originale: matrice2×2 LR costante/decrescente × clipping aggregato/percanale, width16/32, treseed (24sistemi). Nessun dataset da montare.\n']
n['cells'][2]['source']=['## 🧪 Ipotesi e controllo\n','Stessi dati e inizializzazioni; checkpoint fino60k. Soglie invariate, nuovo fresh dopo freeze. Selezione solo development, include held-V composto esplicitamente distinto dal rollout iterativo. Probe frozen inf/tau e FP32/FP64 non selezionabili. I modelli condivisi precedenti restano congelati. Promozione solo se il braccio primario decay_channel supera3/3.\n']
(root/'notebooks/24b_sodium_control_diagnosis.ipynb').write_text(json.dumps(n,ensure_ascii=False,indent=1)+'\n',encoding='utf-8',newline='\n')
