"""Post-freeze probes and historical analogies; no additional model selection."""
import hashlib
import json
import sqlite3
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'experiments/results/task39d_colab_v1'
def main():
    gradients=[]
    for x in json.loads((OUT/'gradient_probes.json').read_text()):
        if x['step']!=120000:continue
        weighted=np.array(x['term_norms'])*np.array(x['weights'])
        gradients.append(dict(architecture=x['architecture'],weighted_norms=weighted.tolist(),
            auxiliary_to_trajectory=(weighted[:,2:].sum(1)/weighted[:,:2].sum(1).clip(1e-30)).tolist(),
            trajectory_vs_inf_cosine=np.array(x['cosines'])[:,0,2].tolist(),
            limits='One fixed FIT probe; norm sums not gradient of total; conflicts do not prove causal interference.'))
    codes=['finding-task2b-h-physical-tau-identifiability-repaired-v1',
        'finding-task3d-voltage-support-conditional-shape-repair-v1',
        'findings-giada-roadmap-task26-gradient_interference-v1','findings-giada-ie-p6-outcome-v1']
    sql='SELECT * FROM findings WHERE "Codice stabile"=?'
    with sqlite3.connect(f'file:{ROOT/"research_memory/data/research_memory.sqlite"}?mode=ro',uri=True) as db:
        db.row_factory=sqlite3.Row
        precedents=[dict(db.execute(sql,(code,)).fetchone()) for code in codes]
    result=dict(valid=True,scope='post-freeze, exploratory; no stress selection',gradients=gradients,
        precedent_query=sql,precedent_codes=codes,precedents=precedents,
        sources={n:hashlib.sha256((OUT/n).read_bytes()).hexdigest() for n in ['gradient_probes.json','paired_contrasts.json','result_audit.json']})
    (OUT/'residual_diagnosis.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    print(json.dumps(dict(gradients=[{k:v for k,v in x.items() if k!='weighted_norms'} for x in gradients],
        precedents=[{k:p.get(k) for k in ['Nome','Risultato','Limitazioni']} for p in precedents]),ensure_ascii=False,indent=2))
if __name__=='__main__':main()
