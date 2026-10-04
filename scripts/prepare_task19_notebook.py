"""Mechanically adapt the verified Task18d notebook to original Task19."""
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
notebook = json.loads((root / 'notebooks/18d_roadmap_potassium_architecture.ipynb').read_text(encoding='utf-8'))
for cell in notebook['cells']:
    if cell['cell_type'] == 'code':
        cell['source'] = [line.replace('giada_task18d_', 'giada_task19_').replace('run_roadmap_task18d.py', 'run_roadmap_task19.py')
            .replace('giada_task19_potassium_', 'giada_task19_single_gate_')
            .replace("'potassium_passed','selected','task19_authorized'", "'single_gate_transfer_passed','per_channel_transfer','task20_authorized'")
            .replace("'training_seconds'", "'training_and_evaluation_seconds'") for line in cell['source']]
        cell['outputs'] = []
        cell['execution_count'] = None
notebook['cells'][0]['source'] = ['# 🧬 GIADA — Task19 originale: Ih e Im\n',
    '24 modelli indipendenti: due meccanismi a un gate, due obiettivi, due capacità, tre seed. Audit nativo, LUT, stato estremo, rollout1000 e latenzaGPU. Nessun Dataset da montare.\n']
notebook['cells'][2]['source'] = ['## ⚗️ Contratto preregistrato\n',
    'V e parametri canonici costanti; input V,x,dt. Stream appaiati e optimizer indipendenti; checkpoint0/1k/5k/15k/30k. Selezione solo development, comune ai3seed. Test fresh dopo freeze; OOD separato. La Task20 richiede entrambi i canali e tutti i seed entro le soglie registrate, inclusi i rollout.\n']
path = root / 'notebooks/19_roadmap_single_gate_transfer.ipynb'
path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
for cell in notebook['cells']:
    if cell['cell_type'] == 'code':
        compile(''.join(cell['source']), str(path), 'exec')
print('Task19 notebook generated and all code cells syntax-checked')
