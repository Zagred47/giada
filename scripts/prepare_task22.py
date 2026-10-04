"""Generate Task22 orchestration and mirror preregistration from tested templates."""
import json
from pathlib import Path
R=Path(__file__).resolve().parents[1]
cfg=json.loads((R/'experiments/task21_slow_gate_transfer.json').read_text(encoding='utf-8'))
cfg.update(schema_version='giada-roadmap-task22-preregistration-v1',stage='Task22 originale: condivisione controllata',channels=['Ih','Im','Nap_h'],families=['independent','shared_heads','conditioned'],objectives=['rate_supervised'],
 data_seeds={'fit':220101,'fit_regions':220102,'development':[220201,220202],'minibatches':220301,'fresh':[220401,220402,220403,220404,220405]},
 architecture='Three scalar gates Ih/Im/Nap_h. Independent:3 separate1-w-w-2MLPs. Shared_heads:one1-w-wtrunk and3two-outputheads. Conditioned:4-w-w-2MLP with voltage+onehot mechanism id. State and dt enter only analytic convex exponential solver. Inf sigmoid,tau exp(clampedlogtau). Three independent seed bundles.',
 fit_sampling='50% uniform[-135,75],25%[-120,-95],25%[-75,-25]; common V,state,dt tuples for all3mechanisms/architectures/widths/seeds. Fixed multiscale durations.',
 selection='One common width/checkpoint per architecture across3seeds and3mechanisms by worst-seed worst-mechanism worst-domain normalized gate/rate score on development. Freeze before fresh; no rollout selection.',
 promotion='Sharing confirmed only if independent control passes3/3 and a shared family passes3/3 on every mechanism/domain/horizon with fewer selected parameters per three-gate bundle. No posthoc thresholds. Task23 preparation authorized if independent reference passes; scientific sharing decision separate.',
 native_contract='Fresh subprocess native audits for canonical Ih,Im,Nap_Et2 h using RANGE-only diagnostics. Reuse verified double-oracle equations; fixed teacher074c466. No joint native interpreter libraries.',
 limits='Isolated held-voltage gates, not entire Nap channel, current or coupled neuron. SK_E2 excluded to avoid calcium-input confounding. Known mechanism identity available through routing for separate/head models and onehot for conditioned. Widths are matched, parameter counts NOT equal; explicit parameter/accuracy tradeoff, not matched-parameter superiority claim.',
 performance='18 three-gate systems:3families×2widths×3independentseeds. Seed bundles vectorized; Adam/clipping independent perseed. Each family uses equal mean loss across3mechanisms and same norm1 bundle clipping. No measured inference speedup.',
 hypotheses={
 'sharing_learnability':'At least one shared architecture passes all registered scalar-gate criteria with fewer parameters than the independently trained bundle.',
 'identity_use':'Swapping mechanism identity/head routing increases in-support gateRMSE at least10fold in median over3seeds; confirms identity matters, not promotion criterion.',
 'negative_transfer':'Sharing can degrade worst-mechanism accuracy even if average error improves. Report permechanism, not pooled success.',
 'capacity_budget':'Width16/32 and checkpoints quantify capacity/optimization under sharing. >=10% worst-development score improvement at30k supports capacity benefit; no monotonicity assumed.',
 'long_rollout':'All3mechanisms remain bounded and accurate through10000 repeated1ms updates; OOD voltage kept diagnostic only.'})
cfg['gates'].pop('open_rmse');cfg.pop('short_dt_values_ms',None)
cfg['required_artifacts']=['native_audit.json','run_contract.json','code_provenance.json','equivalence_preflight.json','development_ladder.json','selection_freeze.json','fresh_metrics.json','final_report.json','checkpoint_*.pt','fit_development.npz','fresh.npz']
cfg['required_artifacts'].append('paired_architecture_contrasts.json')
(R/'experiments/task22_controlled_gate_sharing.json').write_text(json.dumps(cfg,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
s=(R/'scripts/run_roadmap_task21.py').read_text(encoding='utf-8').replace('Task21','Task22').replace('task21','task22').replace('slow_gate_transfer','controlled_gate_sharing').replace('task20_kaggle_f94993f','task21_kaggle_ea1f758').replace('Task20 GO','Task21 GO')
# Include all equation implementations in provenance.
s=s.replace("'src/giada_teacher/controlled_gate_sharing.py',", "'src/giada_teacher/controlled_gate_sharing.py', 'src/giada_teacher/slow_gate_transfer.py',")
(R/'scripts/run_roadmap_task22.py').write_text(s,encoding='utf-8')
nb=json.loads((R/'notebooks/21_roadmap_slow_gate_transfer.ipynb').read_text(encoding='utf-8'))
for n,c in enumerate(nb['cells']):
 c['id']=f'task22-{n}'
 if c['cell_type']=='code':
  c['source']=[s.replace('task21','task22').replace('slow_gate','controlled_sharing').replace("'slow_gate_passed','per_channel_transfer','task22_authorized'","'sharing_confirmed','per_family_passed','task23_authorized'") for s in c['source']]
  # Replacement order moves authorization to task23 explicitly.
  c['source']=[s.replace("'controlled_sharing_passed','per_channel_transfer','task22_authorized'","'sharing_confirmed','per_family_passed','task23_authorized'") for s in c['source']]
  compile(''.join(c['source']),'task22','exec')
nb['cells'][0]['source']=['# 🧩 GIADA Task22 originale — condivisione controllata\n','Ih, Im, Nap_h: modelli separati / trunk+head / identità onehot.18sistemi vettorizzati, stessi dati, due capacità e budget crescenti. Nessun input da montare.\n']
nb['cells'][2]['source']=['## 🔬 Contratto\n','Accuratezza per gate e seed, rollout1ms fino10s, identità scambiata. Width allineate, parametri non identici: misurare tradeoff esplicito. Non canali completi, non speedup.\n']
(R/'notebooks/22_roadmap_controlled_gate_sharing.ipynb').write_text(json.dumps(nb,ensure_ascii=False,indent=1)+'\n',encoding='utf-8')
reg=(R/'research_memory/register_task21_preregistration.py').read_text(encoding='utf-8').replace('Task21','Task22').replace('task21','task22').replace('slow_gate_transfer','controlled_gate_sharing').replace('decisions-giada-roadmap-task20-to21-v1','decisions-giada-roadmap-task21-to22-v1')
reg=reg.replace("for channel in cfg['channels']:","for channel in cfg['families']:")
reg=reg.replace('24modelli:2supportidt×2obiettivi×2width×3seed','18sistemi:3architetture×2width×3seed,3gate ciascuno').replace('Dinamica lenta; supportodurate; identificabilitàrate; mini scaling; precisionenumerica.','Condivisione pesi; tradeoff parametri/accuratezza; negative transfer; identità; scaling.')
reg=reg.replace("'Ruolo': 'Trattamento' if objective == 'rate_supervised' else 'Controllo negativo'","'Ruolo': 'Baseline' if channel == 'independent' else 'Trattamento'")
reg=reg.replace('Pesi indipendenti; stessi dati,minibatch,seed,inizializzazioneentrocapacità.','Seed indipendenti; stesso stream e loss perbundle; condivisione interna secondo famiglia. Parametri perbundle espliciti.')
reg=reg.replace('V/stato comuni; dt breve o multiscala secondo braccio. Fresh comune, mai modificato per condizione.','V/stato/dt comuni a tutte famiglie. Routing/head o onehot danno identità nota in tutte architetture.')
reg=reg.replace('    from .task20_reference_followup import register\n    register(mirror)\n','')
reg=reg.replace('Nap_Et2 h','Ih/Im/Nap_h').replace('audit, LUT, estremi stato, rollout e floorfloat32/persistence.','audit, estremi stato, rollout e identità scambiata.').replace('controlled_gate_sharing.measurements.','controlled_gate_sharing.metrics.')
reg=reg.replace('Non finito o tau≤0 sempre fallimento; openRMSE è alias errore h, NON errore apertura totale m^3*h.','Non finito o tau≤0 sempre fallimento; ogni gate valutato separatamente.').replace('dt differisce solo nel fattore registrato;','dt identico in tutte famiglie;')
reg=reg.replace('    mirror.commit_batch()', '    from .task21_reference_followup import register\n    register(mirror)\n    mirror.commit_batch()')
a=reg.index('    for size in (513, 2049):');b=reg.index('    for index, (key, text)',a);reg=reg[:a]+reg[b:]
(R/'research_memory/register_task22_preregistration.py').write_text(reg,encoding='utf-8')
launch=(R/'research_memory/register_task21_launch.py').read_text(encoding='utf-8').replace('Task21','Task22').replace('task21','task22').replace('21_roadmap_slow_gate_transfer','22_roadmap_controlled_gate_sharing').replace('24modelli','18sistemi').replace('data210xxx','data220xxx')
(R/'research_memory/register_task22_launch.py').write_text(launch,encoding='utf-8')
print('Task22 artifacts generated')
