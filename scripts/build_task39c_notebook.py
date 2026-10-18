"""Use the established pinned-commit subprocess and optional Blob scaffold."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
nb=json.loads((ROOT/'notebooks/giada_task39b_architecture_diagnosis.ipynb').read_text())
for cell in nb['cells']:
    cell['source']=[line.replace('39B','39C').replace('39b','39c').replace('architecture_diagnosis','dynamic_identification') for line in cell['source']]
    if cell['cell_type']=='code':cell['outputs']=[];cell['execution_count']=None
nb['cells'][0]['source']=['# 🧪 GIADA Task39c — Identificazione dinamica e learnability\n',
    'Oracoli sui checkpoint39b, poi 8 bracci informazione×supporto×durata e4 confronti architetturali,3seed:36modelli.\n',
    'Tre coorti vettorizzate in un unico CUDA Graph. Ancore held fino10s e sequenze variabili appaiate. Budget fino120k con checkpoint.\n',
    'Soglie Task37,18stati,dt0.1ms; selezione development con finestre lunghe. Nessuna formula cinetica nell’inferenza appresa.\n',
    'Input Kaggle aggiuntivi:0. Revisione completa fissata dal launcher.']
# Batch jobs must not emit a base64 archive into the notebook output.
body=nb['cells'][-1]['source']
nb['cells'][-1]['source']=["# Per il download nel browser impostare GIADA_BROWSER_DOWNLOAD=1 e rieseguire.\n",
    "if os.environ.get('GIADA_BROWSER_DOWNLOAD') == '1':\n"]+['    '+line.rstrip('\n')+'\n' for line in body]+[
    "else:\n", "    print({'archive_ready':archive,'automatic_browser_download':False})\n"]
nb['cells'][-1]['metadata']['tags']=['optional-browser-download']
path=ROOT/'notebooks/giada_task39c_dynamic_identification.ipynb'
path.write_text(json.dumps(nb,ensure_ascii=False,indent=1)+'\n',encoding='utf-8',newline='\n')
print(path.name)
