"""Mechanically generate the aligned Task26 notebook from tested Task25 template."""
import json
from pathlib import Path
root=Path(__file__).resolve().parents[1]
nb=json.loads((root/'notebooks/25_roadmap_heterogeneous_mechanisms.ipynb').read_text())
for cell in nb['cells']:
    cell['id']=cell['id'].replace('task25','task26')
    cell['source']=[line.replace('Task25','Task26').replace('task25','task26').replace('heterogeneous_mechanisms','joint_vs_independent').replace('composition_passed','comparison_passed').replace('per_arm_passed','per_family_passed').replace('task26_preparation_authorized','task27_preparation_authorized').replace("'evaluation_seconds'","'training_and_evaluation_seconds'") for line in cell['source']]
nb['cells'][0]['source']=['# 🔬 GIADA Task26 — joint vs independent\n','Cinque canali, dieci stati m/h. Famiglia × width, tre seed vettorizzati, ladder fino60k. Nessun dataset da montare.\n']
nb['cells'][2]['source']=['## 🧪 Condivisione, capacità e budget\n','Training da zero, stesso fit/minibatch/LR, selezione solo development e freeze prima del fresh. Probe gradienti, rate, gate, correnti individuali/totale, helditerativo e path veloci/lenti. Corrente analitica, V imposto: Task27/28 e closedloop restano separate.\n']
destination=root/'notebooks/26_roadmap_joint_vs_independent.ipynb'
destination.write_text(json.dumps(nb,ensure_ascii=False,indent=1)+'\n',encoding='utf-8',newline='\n')
