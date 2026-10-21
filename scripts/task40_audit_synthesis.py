"""Persist reproducible descriptive queries and audit interpretation."""
import json,sqlite3,sys
from pathlib import Path
from collections import Counter
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
OUT=ROOT/'experiments/results/task40_audited_v1'
def main():
    read=lambda n:json.loads((OUT/n).read_text())
    c=sqlite3.connect('file:'+str(ROOT/'research_memory/data/research_memory.sqlite').replace('\\','/')+'?mode=ro',uri=True);c.row_factory=sqlite3.Row
    queries={
      'prior_findings':("SELECT * FROM findings WHERE \"Codice stabile\" IN (?,?,?,?)",['findings-giada-task39j-result-sufficient_package-v1','findings-giada-task39j-result-freeze_contrast-v1','findings-giada-task39j-result-remaining_stress-v1','findings-giada-iv-b-v2-result-pass-v1']),
      'prior_lesson':("SELECT * FROM lessons WHERE \"Codice stabile\"=?",['lessons-giada-task39j-result-protect-learned-function-v1']),
      'hypotheses':("SELECT * FROM claims WHERE \"Codice stabile\" IN (?,?)",['claims-giada-task39i-result-objective_precision-v1','claims-giada-task39e-result-residual-metric_or_precision-v1'])}
    result={k:dict(sql=sql,parameters=params,rows=[dict(r) for r in c.execute(sql,params)]) for k,(sql,params) in queries.items()};c.close()
    (OUT/'prior_memory_queries.json').write_text(json.dumps(result,indent=2)+'\n')
    failed=[r for r in read('failure_details.json') if r['model']=='J1' and r['mode']=='autonomous'];a=read('audit.json');ctrl=read('attribution_audit.json')
    stats=dict(case_seed_failures=len(failed),unique_cases=len(set(r['case'] for r in failed)),
        by_seed=dict(Counter(r['seed'] for r in failed)),by_source=dict(Counter(r['metadata']['source_index'] for r in failed)),
        by_protocol=dict(Counter(r['metadata']['protocol'] for r in failed)),
        all_nrmse_failed=all('nrmse' in ch['failed_metrics'] for r in failed for ch in r['channels']),
        absolute_rmse_range=[min(ch['rmse_ma_cm2'] for r in failed for ch in r['channels']),max(ch['rmse_ma_cm2'] for r in failed for ch in r['channels'])],
        nrmse_range=[min(ch['nrmse'] for r in failed for ch in r['channels']),max(ch['nrmse'] for r in failed for ch in r['channels'])])
    findings={
        'outcome':'Task40 technical execution valid; J1 264/264 imposed and autonomous trajectory checks pass, but 22/168 autonomous intervention case-seed checks fail. No GO to Task41.',
        'localization':'All 22 failures concern calcium channels: 18 Ca_LVAst, 4 Ca_HVA. I3 and J1 fail identical cases. All are early_high protocols: 10 distinct intervention/source cases, not 22 independent protocols.',
        'numerical_floor':'All 672 saved intervention verdicts reproduced. Recalculation maximum numeric discrepancy: 5.59e-10. Casting before subtraction changes no verdict. Formula FP32 versus FP64 passes 56/56 differential controls; candidates still fail 22/168 against FP64. Numerical roundoff alone is not supported.',
        'causal_channel_control':'Frozen CPU replay matches 168/168 GPU differential flags. Learned calcium-only has 22 failures; learned all-except-calcium has 0. Restoring only Ca_LVAst leaves 4, only Ca_HVA leaves 18. Canonical substitutions are diagnostic oracles, not eligible candidates.',
        'kinetic_control':'At fixed reference V/Ca and common initial state, learned Ca coefficients fail 22; oracle u still 22; oracle z gives 0; oracle m-z alone leaves 3. In this diagnostic, correcting relaxation coefficients suffices to remove failures; correcting equilibrium alone does not. This is a sufficiency result, not proof of a unique cause or learnability.',
        'readout_control':'Reference V in readout leaves 22; reference m leaves 2; reference h leaves 17; reference m+h leaves 0. Dominant activation kinetics with a secondary inactivation contribution, not evidence that h can be ignored.',
        'why_J1_did_not_fix':'Task39j learned corrections target only gates 8,9,12,13 (Nap_Et2,K_Pst). Calcium gates 0..3 remain I3. Their residual failure is not a regression introduced by 39j. Its atomic frozen-function method is a promising hypothesis to transfer, not proof of Ca repair.',
        'scope':'The 32 historical native paths and 56 formula interventions are opened diagnostics, not fresh NEURON confirmation. Imposed mode mixes native base V/Ca with formula intervention V/Ca; consistent formula-path coefficient controls also reproduce 22 failures. Original NO-GO preserved.',
        'stress':'Both known stress failures remain: Ca_LVAst offset (seed17, same_v_history case1) and Ca_HVA NRMSE (seed29, activation_boundary case0). Shared channels motivate testing a common repair but do not prove an identical cause.',
        'decision':'Propose Task40b calcium kinetic precision diagnosis, not Task41. No training launched and no current threshold relaxed.'}
    hypotheses=[
      dict(id='relaxation_precision',priority='high',statement='Small learned calcium relaxation errors, especially m, cause the differential failures.',evidence='oracle_z22to0;oracle_u22to22',test='FrozenJ1 pluslearnedCa-only logit-z correction; allotherfunctionsunchanged.'),
      dict(id='sensitivity_objective',priority='medium_high',statement='Pointwise coefficient accuracy does not sufficiently constrain local response slopes.',evidence='Trajectorygatespass butdifferentialgatesfail; notyet isolated fromcoefficientprecision.',test='2x2objective(coefficient-onlyvscoefficient+pairedresponse)xindependentanchorcount(1024vs4096),samearchitectureandupdatebudget.'),
      dict(id='coverage_budget',priority='medium',statement='Independent FIT coverage or optimization budget is insufficient in sensitive calcium regimes.',evidence='Notattributedbycurrentaudit; needsmini-scalinglaws.',test='Save1000/3000/10000/30000updates ineachrun;compare1024/4096anchors,reportexposureandwalltime.'),
      dict(id='local_capacity',priority='medium',statement='Residual correction resolution limits achievable kinetic accuracy.',evidence='Canonicalzrescue showsneededfunction;doesnotshowoptimizerlearnability.',test='Matched32vs64basiscontrol,coefficientobjective,uniformsupport;compareFITandholdoutcurves.'),
      dict(id='coupled_u_z',priority='lower_but_test',statement='Joint equilibrium and relaxation corrections help learnability despite oracle-z sufficiency.',evidence='Oracleu alonefails;oraclez alonepasses,so urepair notdemonstratednecessary.',test='Samecapacity u+z correction versus z-only,protectparentfunction andreportparametercount.'),
      dict(id='m_only_sufficiency',priority='unlikely_universal',statement='Correctingactivation alone suffices for every failure.',evidence='Oraclem-z leaves3 failures;testasrestrictednegativecontrol.',test='m-onlylearnedz correction vsbothm/hz correction.'),
      dict(id='common_stress_cause',priority='open',statement='The same kinetic repair resolves the two long-path stress failures.',evidence='Samechannels,notyet samecausalproof.',test='Frozen replaybothstressfamiliesafterfreeze;nouseforselection;retainfailuresifunresolved.')]
    plan=dict(status='proposed_not_preregistered_or_launched',task='40b',thresholds_changed=False,
        baseline='FrozenJ1',candidate_scope='LearnedCa_HVA/Ca_LVAst coefficient corrections only;noanalyticalgatefallback inpromotablecandidate.',
        arms=[
          dict(id='B0',variant='FrozenJ1 no-training control'),
          dict(id='B1',variant='z-only m+h;32basis;1024independentuniformFITanchors;coefficientloss'),
          dict(id='B2',variant='B1 + independentpairedresponse/sensitivityloss'),
          dict(id='B3',variant='B1 with4096FITanchors'),
          dict(id='B4',variant='B2 with4096FITanchors'),
          dict(id='B5',variant='B1 with64basis'),
          dict(id='B6',variant='B1 butu+z;capacitymatchiffeasibleotherwiseexplicitparameterconfound'),
          dict(id='B7',variant='B1 butm-onlyz negative/restrictedcontrol')],
        primary_contrasts=['B1/B2andB3/B4:objective','B1/B3andB2/B4:coverage','B1/B5:capacity','B1/B6:jointcorrection','B1/B7:m-onlyrestriction'],
        checkpoints=[0,1000,3000,10000,30000],seeds=[17,29,43],
        safeguards=['SharedFITstreams;independentholdout;noexactfailedpathtraining','Samefrozenparent;nochangeabstractionlevel','Coefficient,heldrelaxation,pairedresponseandautonomousmetrics','Original40criteriaand72developmentchecksunchanged','Canonicaloraclesdiagnosticonly','Bothstressfamiliesafterfreeze;nonewauthorizationfromaverages','Task41independentconfirmationonlyaftercausalqualification','GPUvectorizedarmsifparitypreflightpasses;costandexposurereported'])
    output=dict(valid=True,post_hoc=True,statistics=stats,findings=findings,hypotheses=hypotheses,next_experiment=plan,task41_authorized=False)
    (OUT/'synthesis.json').write_text(json.dumps(output,indent=2)+'\n')
    lines=['# GIADA Task40 — audit dei 22 fallimenti','', 'Audit post hoc; soglie e checkpoint invariati. Nessun nuovo training.','', '## Risultati','']
    for k,v in findings.items():lines.append(f'- **{k}**: {v}')
    lines+=['','## Tutti i 22 confronti falliti','', '| Caso | Seed modello | Stimolo | Intervento | Canale | NRMSE | NMAX | Metriche fallite |','|---|---|---|---|---|---|---|---|']
    for r in failed:
        for ch in r['channels']:lines.append(f"| {r['case']} | {r['seed']} | source {r['metadata']['source_index']} | {r['metadata']['intervention']} | {ch['channel']} | {ch['nrmse']:.6f} | {ch['nmax']:.6f} | {', '.join(ch['failed_metrics'])} |")
    lines+=['','## Proposta Task40b (da preregistrare)','',json.dumps(plan,indent=2),'','I file JSON contengono tutte le metriche, incluse quelle che passano, i controlli e i limiti.']
    (OUT/'AUDIT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(stats,indent=2));print('Query rows:',{k:len(v['rows']) for k,v in result.items()})
if __name__=='__main__':main()
