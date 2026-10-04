"""Generate immutable minimal parent fixture and notebook from tested templates."""
import hashlib,json,zipfile
from pathlib import Path
root=Path(__file__).resolve().parents[1]
src=root/'artifacts/giada_task18b_potassium_0a24443_54e99b74.zip'
assert hashlib.sha256(src.read_bytes()).hexdigest()=='543b1197a0364a829f80ae1bfcb3045d4d0d0665948fd2c4697a3a6783b1424c'
dest=root/'experiments/fixtures/task18b_frozen_parent.zip';dest.parent.mkdir(exist_ok=True)
with zipfile.ZipFile(src) as z,zipfile.ZipFile(dest,'w',compression=zipfile.ZIP_DEFLATED) as out:
    for n in ('selection_freeze.json','checkpoint_w32_step60000.pt','final_report.json'):
        info=zipfile.ZipInfo(n,(2026,10,4,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED;out.writestr(info,z.read(n))
cfgpath=root/'experiments/task18c_potassium_local_repair.json';cfg=json.loads(cfgpath.read_text());cfg['parent_fixture_sha256']=hashlib.sha256(dest.read_bytes()).hexdigest();cfgpath.write_text(json.dumps(cfg,indent=2)+'\n',encoding='utf-8')
runner=(root/'scripts/run_roadmap_task18b.py').read_text().replace('Task18b','Task18c').replace('hh_potassium_diagnosis','hh_potassium_local_repair').replace('task18b_potassium_diagnosis.json','task18c_potassium_local_repair.json').replace('scripts/run_roadmap_task18b.py','scripts/run_roadmap_task18c.py').replace("run(output,config,revision)","run(output,config,revision,ROOT/'experiments/fixtures/task18b_frozen_parent.zip')")
(root/'scripts/run_roadmap_task18c.py').write_text(runner,encoding='utf-8')
nb=json.loads((root/'notebooks/18b_roadmap_potassium_diagnosis.ipynb').read_text(encoding='utf-8'))
for c in nb['cells']:
    if c['cell_type']=='code':c['source']=[s.replace('giada_task18b_','giada_task18c_').replace('run_roadmap_task18b.py','run_roadmap_task18c.py') for s in c['source']]
nb['cells'][0]['source']=['# 🧬 GIADA Task18c — riparazione locale K_Pst\n','24 modelli appaiati: copertura locale × top5% loss × learning rate. Checkpoint0/5k/15k/30k. Parent congelato18b incluso; nessun Dataset da montare.']
nb['cells'][2]['source']=['## ⚗️ Contratto\n','Stessi pesi iniziali per seed e Adam azzerato per tutti. Coda negativa conservata; metà bracci aggiunge25%V in[-65,-55]mV. Loss media o media+top5%; lr0.003/0.0003. Freeze development prima di fresh uniform/coda/kink. Soglie invariate; OOD diagnostico.']
(root/'notebooks/18c_roadmap_potassium_local_repair.ipynb').write_text(json.dumps(nb,ensure_ascii=False,indent=1)+'\n',encoding='utf-8')
launch=(root/'research_memory/register_task18b_launch.py').read_text(encoding='utf-8').replace('18b','18c').replace('18c_roadmap_potassium_diagnosis','18c_roadmap_potassium_local_repair').replace('60k','30k').replace('181101/181102; fresh181401/181402/181403','182101/182102/182103; fresh182401/182402/182403/182404')
(root/'research_memory/register_task18c_launch.py').write_text(launch,encoding='utf-8')
