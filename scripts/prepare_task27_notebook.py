"""Generate Task27 notebook from the bounded-output, browser-ZIP Task26 template."""
import json
from pathlib import Path

root=Path(__file__).resolve().parents[1]
notebook=json.loads((root/'notebooks/26_roadmap_joint_vs_independent.ipynb').read_text(encoding='utf-8'))
for cell in notebook['cells']:
    cell['id']=cell['id'].replace('task26','task27')
notebook['cells'][0]['source']=[
    '# 🔬 GIADA Task27 — correnti individuali e totale\n',
    'Cinque canali, dieci stati espliciti, V imposto. Due architetture × quattro loss appaiate × tre seed. Nessun dataset da montare.\n']
notebook['cells'][2]['source']=[
    '## 🧪 Loss di corrente e controllo del mascheramento\n',
    'La corrente resta analitica. Confrontiamo none, individual, total e both con stessi esempi, seed e inizializzazione. '
    'Freeze prima del fresh; ogni canale e ogni pannello sono valutati separatamente. Task28 resta successiva.\n']
notebook['cells'][1]['source']=[line.replace('task26','task27') for line in notebook['cells'][1]['source']]
notebook['cells'][3]['source']=[
    "OUTPUT=Path('/kaggle/working/artifacts')/('giada_task27_current_supervision_'+REVISION[:7]+'_'+uuid.uuid4().hex[:8])\n",
    "completed=subprocess.run([sys.executable,'-u',str(REPO/'scripts/run_roadmap_task27.py'),'--teacher',str(TEACHER),'--output',str(OUTPUT)],check=False)\n",
    "path=OUTPUT/('failure_report.json' if (OUTPUT/'failure_report.json').exists() else 'final_report.json')\n",
    "report=json.loads(path.read_text()) if path.is_file() else {'valid':False,'error':f'worker interrupted: {completed.returncode}'}\n",
    "display({k:report.get(k) for k in ('valid','per_arm_passed','current_supervision_material_benefit','primary_median_worst_current_gain','task28_preparation_authorized','fresh_used_for_selection','training_and_evaluation_seconds','error')})\n",
    "print({'output':str(OUTPUT),'exit':completed.returncode})\n"]
destination=root/'notebooks/27_roadmap_current_supervision.ipynb'
destination.write_text(json.dumps(notebook,ensure_ascii=False,indent=1)+'\n',encoding='utf-8',newline='\n')
