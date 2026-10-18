"""Generate Task39d notebook from the established Task39c launch scaffold."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
nb=json.loads((ROOT/'notebooks/giada_task39c_dynamic_identification.ipynb').read_text(encoding='utf-8'))
for cell in nb['cells']:
    cell['source']=[line.replace('39C','39D').replace('39c','39d').replace('dynamic_identification','current_timing') for line in cell['source']]
    if cell['cell_type']=='code':cell['outputs']=[];cell['execution_count']=None
nb['cells'][0]['source']=['# 🧪 GIADA Task39d — Correnti, memoria e timing\n',
    '12 bracci × 3 seed: 36 modelli in due coorti vettorizzate, stesso CUDA Graph.\n',
    'Fattoriale: scala correnti × storia/esposizione × bilanciamento loss; sidecar eventi e bias lineare appreso.\n',
    'Prima del training: oracoli congelati39c e audit crossing; soglie Task37 invariate.\n',
    '18 stati, dt0.1ms, stesso readout; nessuna cinetica canonica in inferenza. Task40 resta subordinata alla qualità.\n',
    'Nessun input Kaggle aggiuntivo; revisione completa fissata dal launcher.']
path=ROOT/'notebooks/giada_task39d_current_timing.ipynb'
path.write_text(json.dumps(nb,ensure_ascii=False,indent=1)+'\n',encoding='utf-8',newline='\n')
print(path.name)
