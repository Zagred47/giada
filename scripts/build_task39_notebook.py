"""Generate Task39 notebook from the tested Task38 execution/download scaffold."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
nb=json.loads((ROOT/'notebooks/giada_task38_joint_transition.ipynb').read_text())
for c in nb['cells']:
    c['source']=[line.replace('Task38','Task39').replace('task38','task39').replace('TASK38','TASK39')
        .replace('joint_transition','learnability').replace('Six paired GPU members','Thirty paired GPU members') for line in c['source']]
    if c['cell_type']=='code':c['outputs']=[];c['execution_count']=None
nb['cells'][0]['source']=['# 🧪 GIADA Task39 — Learnability diagnosis\n',
    '2×2×2 factorial + two loss anchors, three seeds:30 vectorized members. Same eight supervised targets per update; dataset coverage, recurrent exposure and physical dependency bias isolated.\n',
    'Budget ladder0/3k/10k/30k/60k. Task37 thresholds unchanged. Task40 remains blocked unless quality passes and the result is audited.\n',
    'The launcher sets GIADA_TASK39_REVISION to the full pushed commit. No new Kaggle Input datasets required.']
# Do not print the large per-arm summary in notebook output.
nb['cells'][2]['source']=[line.replace("'error','summaries'","'error','passing_arms'") for line in nb['cells'][2]['source']]
path=ROOT/'notebooks/giada_task39_learnability.ipynb'
path.write_text(json.dumps(nb,ensure_ascii=False,indent=1)+'\n',encoding='utf-8',newline='\n')
print(path.name)
