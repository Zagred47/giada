"""Generate Task18d fixture and orchestration from verified Task18c templates."""
import hashlib,json,zipfile
from pathlib import Path
root=Path(__file__).resolve().parents[1];src=root/'artifacts/giada_task18c_potassium_c1a8e0b_e6158d0f.zip'
assert hashlib.sha256(src.read_bytes()).hexdigest()=='e16025c95e7395bd5baa0c6b50afd96a9c7d2358e9d96360d7250e4c4a2dd985'
dest=root/'experiments/fixtures/task18c_frozen_parent.zip'
with zipfile.ZipFile(src) as z,zipfile.ZipFile(dest,'w',compression=zipfile.ZIP_DEFLATED) as out:
    for n in ('selection_freeze.json','checkpoint_step30000.pt','final_report.json'):
        info=zipfile.ZipInfo(n,(2026,10,4,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED;out.writestr(info,z.read(n))
cfgpath=root/'experiments/task18d_potassium_architecture.json';cfg=json.loads(cfgpath.read_text(encoding='utf-8'));cfg['parent_fixture_sha256']=hashlib.sha256(dest.read_bytes()).hexdigest();cfgpath.write_text(json.dumps(cfg,indent=2)+'\n',encoding='utf-8')
runner=(root/'scripts/run_roadmap_task18c.py').read_text(encoding='utf-8').replace('Task18c','Task18d').replace('hh_potassium_local_repair','hh_potassium_architecture').replace('task18c_potassium_local_repair.json','task18d_potassium_architecture.json').replace('run_roadmap_task18c.py','run_roadmap_task18d.py').replace('fixtures/task18b_frozen_parent.zip','fixtures/task18c_frozen_parent.zip')
runner=runner.replace("'scripts/run_roadmap_task18d.py'","'scripts/run_roadmap_task18d.py','src/giada_teacher/hh_potassium_local_repair.py','src/giada_teacher/hh_potassium_diagnosis.py','experiments/fixtures/task18c_frozen_parent.zip'")
(root/'scripts/run_roadmap_task18d.py').write_text(runner,encoding='utf-8')
nb=json.loads((root/'notebooks/18c_roadmap_potassium_local_repair.ipynb').read_text(encoding='utf-8'))
for c in nb['cells']:
    if c['cell_type']=='code':c['source']=[s.replace('giada_task18c_','giada_task18d_').replace('run_roadmap_task18c.py','run_roadmap_task18d.py') for s in c['source']]
nb['cells'][0]['source']=['# 🧬 GIADA Task18d — struttura di mTau\n','9modelli: MLP liscia, feature cuspide, due head mTau. Tre seed e checkpoint0/5k/15k/30k. Parent18c incluso. Nessun Dataset da montare.']
nb['cells'][2]['source']=['## ⚗️ Confronto e conferma\n','Stessa funzione iniziale, dati, minibatch, optimizer e loss. Capacità effettive entro3%. Tre domini fresh e griglia della regione critica con stati0/1 dopo freeze. Soglie originali e nuovo requisito denso preregistrato.']
(root/'notebooks/18d_roadmap_potassium_architecture.ipynb').write_text(json.dumps(nb,ensure_ascii=False,indent=1)+'\n',encoding='utf-8')
launch=(root/'research_memory/register_task18c_launch.py').read_text(encoding='utf-8').replace('18c','18d').replace('18d_roadmap_potassium_local_repair','18d_roadmap_potassium_architecture').replace('24 modelli','9 modelli').replace('24 modelli indipendenti','9 modelli indipendenti').replace('182','183')
(root/'research_memory/register_task18d_launch.py').write_text(launch,encoding='utf-8')
reg=(root/'research_memory/register_task18b_result_18c.py').read_text(encoding='utf-8')
reg=reg.replace('task18c','TASK_NEXT').replace('Task18c','TASK_NEXT_TITLE').replace('task18b','task18c').replace('Task18b','Task18c').replace('TASK_NEXT','task18d').replace('task18d_TITLE','Task18d')
reg=reg.replace('giada_task18c_potassium_0a24443_54e99b74.zip','giada_task18c_potassium_c1a8e0b_e6158d0f.zip').replace('543b1197a0364a829f80ae1bfcb3045d4d0d0665948fd2c4697a3a6783b1424c','e16025c95e7395bd5baa0c6b50afd96a9c7d2358e9d96360d7250e4c4a2dd985').replace('task18c_kaggle_0a24443','task18c_kaggle_c1a8e0b').replace(",'tail_probe_metrics.json'",'').replace('task18d_potassium_local_repair.json','task18d_potassium_architecture.json').replace('task18c-w32-tail_enriched-tau0.1','task18c-tail_and_kink-top0-lr0.003')
reg=reg.replace('3/3seed RMSEuniform<.001 e coda passa; massimo uniform.0141–.0158>.01. Probe: tutti massimi m aV−60.284, dt25; tauoracle abbatte errori<.0008.','3/3uniform e coda passano; kink2/3. Seed17 max.010521>.01. Probe tutti m aV−59.931 dt25; tauoracle errore<.000258. Parent appaiato migliora uniform/coda/kink; OOD negativo peggiora e resta diagnostico.')
reg=reg.replace('Local coverage x worst5% loss x learning rate;24models; frozen-parent initialization; Adam reset matched.','Smooth vs cusp feature vs branch heads;9models; frozen-parent initialization; Adam reset matched.').replace('Distinguere copertura della cuspide, ottimizzazione robusta e rifinitura da semplice budget.','Distinguere struttura della curva da ulteriore ottimizzazione; controllo estremi stato.').replace('Local mTau error; preservare tail and Ca/Na; no blindscaling.','Errore mTau locale; confronto strutturale e griglia stati estremi.')
reg=reg.replace('w32 step60000 tail_enriched tau0.1; common development selection','w32 step30000 tail_and_kink top0 lr0.003; common development selection').replace('Fresh18b ora consumato','Fresh18c ora consumato').replace("'Nome':'18c prima di19'","'Nome':'18d prima di19'")
(root/'research_memory/register_task18c_result_18d.py').write_text(reg,encoding='utf-8')
