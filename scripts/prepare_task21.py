"""Mechanical generation from tested Task19/20 orchestration, not result data."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
cfg=json.loads((ROOT/'experiments/task19_single_gate_transfer.json').read_text(encoding='utf-8'))
cfg.update(schema_version='giada-roadmap-task21-preregistration-v1',stage='Task21 originale, Atto III: dinamica lenta Nap_Et2 h',
 channels=['short_dt','multiscale_dt'],dt_values_ms=[.025,.1,.5,1,5,25,100,500,1000,5000,10000],
 short_dt_values_ms=[.025,.1,.5,1],rollout_horizons_steps=[1,10,100,1000,10000],
 data_seeds={'fit':210101,'fit_regions':210102,'development':[210201,210202],'minibatches':190301,'fresh':[210401,210402,210403,210404,210405]},
 fit_sampling='50% uniform[-135,75],50%[-75,-25]mV. Same V/state tuples, seed, initialization and minibatch indices; only registered training dt support differs.',
 selection='Common width/checkpoint per duration-support/objective across3seeds by worst-seed worst-domain normalized sum gateRMSE,gateMAX,infRMSE,logtauRMSE. Development uses common multiscale dt for every arm. Freeze before fresh. No rollout selection.',
 promotion='Task22 authorized only if multiscale_dt/rate_supervised all3seeds pass fresh in_support,negative_tail,state_extrema and all registered repeated1ms horizons through10000ms. Other arms diagnostic, not fallback candidates. OOD diagnostic only.',
 native_contract='Canonical Nap_Et2 h; qt2.3^1.3; exact local voltage mutations -38,-17,-64.4 preserved, including preceding m-rate branch. Zero conductance. RANGE-only hInf/hTau instrumentation. Native rates/gate errors<1e-7, section drift<1e-8mV.',
 architecture='MLP1->w->w->2 SiLU; hInf=sigmoid,tau=exp(clamped logtau); convex exponential gate update. Independent models. h only, no current claim.',
 limits='Isolated slow h gate, held imposed voltage. Not entire Nap_Et2 m^3*h channel; no current, endogenous voltage, full-neuron, shared-weight or speedup claim. Short vs multiscale duration is intentional information intervention. Long labels exact only at held voltage; simulation rollout remains1ms.',
 performance='Vectorized independent24models onCUDA; CPU double oracle separate native process. Float32 formula rollout floor, persistence and learned single macro update diagnostic; no speedup measurement.',
 hypotheses={
 'slow_gate_transfer':'The constrained slow h cell with multiscale duration support and rate supervision passes all registered gates in all3seeds.',
 'duration_identifiability':'Longer training durations expose slow kinetics better than short-only transitions, holding V/state/initialization/minibatch streams fixed; paired development score reduction>=10% at30k supports practical benefit.',
 'rate_identifiability':'Rate supervision improves hidden inf/logtau and long rollout over paired transition-only models; >=10% worst-development score reduction at30k supports benefit.',
 'capacity_budget':'Paired width16/32 and checkpoints distinguish capacity/budget; >=10% score reduction supports practical benefit, monotonic improvement not assumed.',
 'state_rollout':'Repeated1ms updates up to10000ms reveal errors masked at1ms. Compare persistence, formula float32 floor and learned macro update; report separately without fresh selection.',
 'numerical_reference':'LUT513/2049 voltage-grid baselines measure approximation reference; not a timing benchmark.'})
cfg['required_artifacts'] += ['paired_duration_contrasts.json']
(ROOT/'experiments/task21_slow_gate_transfer.json').write_text(json.dumps(cfg,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
runner=(ROOT/'scripts/run_roadmap_task20.py').read_text(encoding='utf-8').replace('Task20','Task21').replace('task20','task21').replace('calcium_gate_transfer','slow_gate_transfer').replace('task19_kaggle_9f5d49f','task20_kaggle_f94993f').replace('Task19 GO','Task20 GO')
(ROOT/'scripts/run_roadmap_task21.py').write_text(runner,encoding='utf-8')
nb=json.loads((ROOT/'notebooks/20_roadmap_calcium_gate_transfer.ipynb').read_text(encoding='utf-8'))
for n,c in enumerate(nb['cells']):
 c['id']=f'task21-{n}'
 if c['cell_type']=='code':
  c['source']=[s.replace('task20','task21').replace('calcium_gate','slow_gate').replace('task21_authorized','task22_authorized') for s in c['source']]
  compile(''.join(c['source']),'task21','exec')
nb['cells'][0]['source']=['# 🐢 GIADA Task21 originale — dinamica lenta Nap_Et2 h\n','24 modelli indipendenti vettorizzati: durate brevi/multiscala × due obiettivi × due capacità × tre seed. Nessun input Kaggle da montare.\n']
nb['cells'][2]['source']=['## 🔬 Contratto\n','Gate h isolato, non intero canale Nap. Rollout1ms fino10s; float32 oracle e persistence diagnostici. Rate, durata, capacità e budget nello stesso run. Freeze prima del fresh; niente selezione OOD o claim speedup.\n']
(ROOT/'notebooks/21_roadmap_slow_gate_transfer.ipynb').write_text(json.dumps(nb,ensure_ascii=False,indent=1)+'\n',encoding='utf-8')
reg=(ROOT/'research_memory/register_task19_preregistration.py').read_text(encoding='utf-8').replace('Task19','Task21').replace('task19','task21').replace('single_gate_transfer','slow_gate_transfer').replace('Ih/Im','Nap_Et2 h')
reg=reg.replace('from .register_task18d_result import StagedMirror','from .validated_batch import ValidatedBatchMirror as StagedMirror').replace('decisions-giada-task18d-to19-v1','decisions-giada-roadmap-task20-to21-v1')
reg=reg.replace('2canali','2supportidt').replace('latenzaCUDA','floorfloat32/persistence').replace('cfg = json.loads(path.read_text())',"cfg = json.loads(path.read_text(encoding='utf-8'))")
reg=reg.replace('Trasferibilità gate singolo; identificabilità rate; mini scaling; stabilità aVcostante.','Dinamica lenta; supportodurate; identificabilitàrate; mini scaling; precisionenumerica.')
reg=reg.replace('1/10/100/1000 passi','1/10/100/1000/10000 passi').replace('dt.025/.1/.5/1/5/25/100ms','dt.025/.1/.5/1/5/25/100/500/1000/5000/10000ms')
reg=reg.replace('Stessi input/streamseed/minibatch;','Stesse V/stato/streamseed/minibatch; dt differisce solo nel fattore registrato;')
reg=reg.replace('Tuple numeriche comuni tra canali/obiettivi/width/seed.','V/stato comuni; dt breve o multiscala secondo braccio. Fresh comune, mai modificato per condizione.')
reg=reg.replace('openRMSE=gateRMSE per singologate.','openRMSE è alias errore h, NON errore apertura totale m^3*h.').replace('correntegateanalitica;','h-only, nessuna corrente;')
reg=reg.replace('    mirror.commit_batch()', '    from .task20_reference_followup import register\n    register(mirror)\n    mirror.commit_batch()')
# No diagnostic current is defined for the h-only scope.
start=reg.index("    current = put('metrics'")
end=reg.index('    arms = []',start)
reg=reg[:start]+reg[end:]
(ROOT/'research_memory/register_task21_preregistration.py').write_text(reg,encoding='utf-8')
launch=(ROOT/'research_memory/register_task20_launch.py').read_text(encoding='utf-8').replace('Task20','Task21').replace('task20','task21').replace('20_roadmap_calcium_gate_transfer','21_roadmap_slow_gate_transfer').replace('36modelli','24modelli').replace('data200xxx','data210xxx')
(ROOT/'research_memory/register_task21_launch.py').write_text(launch,encoding='utf-8')
print('Task21 configuration, supervisor, notebook and registrars generated')
