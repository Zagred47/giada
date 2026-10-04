"""Generate Task20 supervisor/notebook from their tested orchestration template."""
import json
from pathlib import Path
root=Path(__file__).resolve().parents[1]
runner=(root/'scripts/run_roadmap_task19.py').read_text(encoding='utf-8')
runner=runner.replace('Task19','Task20').replace('single_gate_transfer import','calcium_gate_transfer import')
runner=runner.replace('task19_single_gate_transfer.json','task20_calcium_gate_transfer.json')
runner=runner.replace('task18d_kaggle_38e6431','task19_kaggle_9f5d49f').replace("prior['task19_authorized']","prior['task20_authorized']").replace('Task18d GO','Task19 GO')
runner=runner.replace("'src/giada_teacher/single_gate_transfer.py',", "'src/giada_teacher/calcium_gate_transfer.py', 'src/giada_teacher/single_gate_transfer.py',")
runner=runner.replace("'experiments/task19_single_gate_transfer.json'","'experiments/task20_calcium_gate_transfer.json'").replace('run_roadmap_task19.py','run_roadmap_task20.py')
(root/'scripts/run_roadmap_task20.py').write_text(runner,encoding='utf-8')
nb=json.loads((root/'notebooks/19_roadmap_single_gate_transfer.ipynb').read_text(encoding='utf-8'))
for n,c in enumerate(nb['cells']):
    c['id']=f'task20-{n}'
    if c['cell_type']=='code':
        c['source']=[s.replace("'single_gate_transfer_passed','per_channel_transfer','task20_authorized'","'calcium_gate_passed','per_channel_transfer','task21_authorized'").replace('giada_task19_','giada_task20_').replace('run_roadmap_task19.py','run_roadmap_task20.py').replace('single_gate_','calcium_gate_') for s in c['source']]
        compile(''.join(c['source']),'task20','exec')
nb['cells'][0]['source']=['# 🧬 GIADA Task20 originale — SK_E2\n','36modelli:calcio log/lineare,soloV×dueobiettivi×duecapacità×treseed. Calcio imposto, non CaDynamics. Nessun input da montare.\n']
nb['cells'][2]['source']=['## ⚗️ Contratto\n','Calcio in mM, tau canonica1ms e corrente analitiche. Freeze prima del fresh; dominio/OOD distinti; controlloV non riceve calcio. Promozione solo logcalcium/rate3seed più percorsi imposti.\n']
(root/'notebooks/20_roadmap_calcium_gate_transfer.ipynb').write_text(json.dumps(nb,ensure_ascii=False,indent=1)+'\n',encoding='utf-8')
print('Task20 supervisor and notebook generated; cells compile')
registry=(root/'research_memory/register_task19_preregistration.py').read_text(encoding='utf-8')
registry=registry.replace('Task19','Task20').replace('task19','task20').replace('Ih/Im','SK_E2').replace('Ih/Im','SK_E2')
registry=registry.replace('task20_single_gate_transfer.json','task20_calcium_gate_transfer.json')
registry=registry.replace('decisions-giada-task18d-to19-v1','decisions-giada-roadmap-task19-to20-v1')
registry=registry.replace('24modelli:2canali×2obiettivi×2width×3seed','36modelli:3encoding×2obiettivi×2width×3seed')
registry=registry.replace('single_gate_transfer.measurements.','calcium_gate_transfer.measurements.')
registry=registry.replace('1e-5','1e-6').replace('E_Ih−45mV,E_Im−85mV','ek−85mV; V incolonna3')
registry=registry.replace('513/2049 match', '513/2049 match').replace('[-135,75]mV','log10cai[-7,-2]mM').replace('Vcostante','calciocostante').replace('v1;','v1;')
registry=registry.replace('Ih/Im single-gate','SK_E2 calcium-gate').replace('Trasferibilità gate singolo; identificabilità rate; mini scaling; stabilità aVcostante.','Sufficienza inputcalcio; log/lineare; controlloV; rate; scaling; calcioimposto.')
registry=registry.replace("'Ruolo': 'Trattamento' if objective == 'rate_supervised' else 'Controllo negativo'", "'Ruolo': 'Controllo negativo' if channel == 'voltage_only' else 'Trattamento'")
registry=registry.replace('Stesse tuplefresh; unità dichiarate','Stesse tuplefresh; calcio mM incolonna0 log10, V mV incolonna3; unità dichiarate')
registry=registry.replace('128nuoveV×2statiiniziali; tutti3seed/canale','128nuovicai×2statiiniziali; tutti3seed/canale').replace('128nuoveV','128nuovicai')
registry=registry.replace('Interpolazione lineare percanale su','Interpolazione lineare su griglia logcalcio per SK_E2; dominio')
registry=registry.replace('Trasferibilità gate singolo; identificabilità rate; mini scaling; stabilità acalciocostante.','Sufficienza calcio; encoding; rate supervision; mini scaling; calcio imposto.')
registry=registry.replace("cfg = json.loads(path.read_text())", "cfg = json.loads(path.read_text(encoding='utf-8'))")
registry=registry.replace('from .register_task18d_result import StagedMirror','from .validated_batch import ValidatedBatchMirror as StagedMirror')
(root/'research_memory/register_task20_preregistration.py').write_text(registry,encoding='utf-8')
launch=(root/'research_memory/register_task19_launch.py').read_text(encoding='utf-8').replace('Task19','Task20').replace('task19','task20').replace('19_roadmap_single_gate_transfer','20_roadmap_calcium_gate_transfer').replace('24modelli','36modelli').replace('data190xxx','data200xxx')
launch=launch.replace("dict(cell_type='code', execution_count=None", "dict(id='task20-pin', cell_type='code', execution_count=None")
launch=launch.replace('from .register_task18d_result import StagedMirror','from .validated_batch import ValidatedBatchMirror as StagedMirror')
(root/'research_memory/register_task20_launch.py').write_text(launch,encoding='utf-8')
