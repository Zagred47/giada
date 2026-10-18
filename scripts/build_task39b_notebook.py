"""Create the Task39b notebook with the tested isolated worker/Blob scaffold."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
nb=json.loads((ROOT/'notebooks/giada_task39_learnability.ipynb').read_text())
for cell in nb['cells']:
    cell['source']=[line.replace('TASK39','TASK39B').replace('Task39','Task39b').replace('task39','task39b')
        .replace('learnability','architecture_diagnosis').replace("'task':38","'task':'39b'") for line in cell['source']]
    if cell['cell_type']=='code':cell['outputs']=[];cell['execution_count']=None
nb['cells'][0]['source']=['# 🧪 GIADA Task39b — Training × bias architetturali\n',
    '8 bracci fattoriali + 4 ancore architetturali, 3 seed = 36 membri indipendenti sulla stessa GPU.\n',
    'Orizzonte8/64 tick × collocazione off/on × loss relativa off/on; teste raggruppate e attivazione multiscala sulle due ancore.\n',
    'Stessi18 stati, nessuna formula cinetica in inferenza, soglie Task37 immutate. Selezione development8/512 tick; valutazione finale fino a10s.\n',
    'Nessun Input Kaggle aggiuntivo. Il launcher imposta GIADA_TASK39B_REVISION al commit completo pubblicato.']
path=ROOT/'notebooks/giada_task39b_architecture_diagnosis.ipynb'
path.write_text(json.dumps(nb,ensure_ascii=False,indent=1)+'\n',encoding='utf-8',newline='\n')
print(path.name)
