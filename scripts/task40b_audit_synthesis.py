"""Interpret frozen audit results with explicit opened-domain and capacity limits."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.giada_teacher.task37_ionic_playground import save_json
F=ROOT/'experiments/results/task40b_audited_v1'
def main():
    read=lambda n:json.loads((F/n).read_text())
    audit=read('audit.json');assert audit['valid'] and audit['cpu_B6_passed']==dict(development=72,stress_diagnostic=72)
    details=read('stress_details.json');ablations=read('stress_ablation.json')
    findings={
      'outcome':'B1-B6 repair all22Task40 autonomous intervention failures:146to168/168;imposed148to168/168. All8arms pass264/264 trajectories inbothmodes and72/72development. B7 m-only retains3Ca_LVAst differential failures inbothmodes.',
      'stress':'B0/J1 reproduces70/72stress;B1-B5 reach71/72;B6alone72/72;B7 remains70/72. Allmodelseeds preserved;24uniqueprotocols perrole,not72independentprotocols.',
      'timing_repair':'same_v_history case1 seed17 Ca_LVAst offset0.5to0.1ms forB1andB6,unchangedlimit0.3ms. Relaxationcorrection resolves this timingresidual.',
      'equilibrium_repair':'activation_boundary case0 seed29 Ca_HVA NRMSE0.01178899to0.01205136forB1(stillfails)and0.00145030forB6(limit0.01). Jointu+z repairs the activation-boundary residual.',
      'ablation':'FrozenB6 z-only passes11/12focusedstress,losingCa_HVA boundary;u-only passes11/12,losingCa_LVAst offset. Bothcomponents required for this frozenpackage ontheseopenedcases,notproofthatallpossiblemodelsneedbothcorrections.',
      'contrast':'No registered incremental contrast meets the20percent medianseverityrule inbothdomains. Globaldevelopmentseverity canbe dominatedbyunchangedchannelsandexcludes timing. This doesnotnegatethecase-specificB6repair. Extra pairedloss,4096anchors,64basisdo not establish addedregisteredbenefit here;notgeneralproofthatdataorcapacityneverhelp.',
      'nomination':'NominateB6 withall3frozenmodelseeds17/29/43,notonebestseed. 32basis,1024FITanchors,coefficientloss,u+zCa-onlycorrection;256activeweights/seedversusB1 128. Parentbitwiseunchanged,18FP32states,sameabstractionlevel,noanalyticalgatefallback.',
      'audit':'Allsavedcheckpoints/Adamandautonomoustracesfinite;freezehash,checkpointSHA,per-memberargminandchosenweightsverified;independentatomicrecipeandselectionlossrecomputed. 2112autonomoustrajectoriesand1344interventionmetricsrecomputedfromsavedtraces;CPU144recurrenceverdictsmatchB6GPU. Imposedtracesnotsaved,soimposedauditchecksallstorednumericcomponents,notfreshfulltrace replay.',
      'membrane':'B6autonomousworstVrmse0.0128588mVandCa3.22957e-7mM;0spike-countmismatches,0spike-timingerror,0domainexits. AbsoluteVqualitydoesnotreplaceionicdifferentialgates.',
      'provenance_erratum':'ExecutedpreregistrationlimitscopiedaTask40lineassertingnotraining/noselection. Explicit40btraining/budget/selectionfieldsandcheckpointsshowactualexecution. Separateerratumpreserved;DBeffectiveprocedureannotatedwithoutchangingexecutedpreregbytes,hash,thresholdsorrun.',
      'decision':'GO to prepareTask41independentlevel1confirmationofB6,notGOtodeploymentormulticompartment. Newprotocols,seeds,parametricpanelsandmatchedcostuncertaintyarerequired;freezeB6andthresholdsbeforefreshaccess. NoB6speedupclaim;Task41mayconcludeaccuratebutslow.'}
    evidence=[
      dict(id='relaxation_precision',outcome='Sostiene',argument='Learnedz-onlym+hcorrectionremoves22autonomousdifferentialfailures;frozenB6u-onlylosestimingstress.'),
      dict(id='sensitivity_objective',outcome='Indeterminata',argument='B1alreadyqualifiescausally;B2/B4donotmeetregistered20percentcontrast.Necessityofpairedobjectiveunsupportedinthisdomain.'),
      dict(id='coverage_budget',outcome='Indeterminata',argument='1024anchorsandearlycheckpointsalreadyqualify;4096doesnotcloseCa_HVAstress.Largercoverage/exposuremaymatterelsewhere.'),
      dict(id='local_capacity',outcome='Indeterminata',argument='32to64basisz-onlydoesnotclosestressorpassregisteredcontrast.B6hasmoreactiveoutputs,acapacityconfoundexplicitlyretained.'),
      dict(id='coupled_u_z',outcome='Sostiene',argument='B6onlyall-domainpass;frozenablationu-removalreopensCa_HVAstress.Scopeisfrozenpackage,notcapacity-matcheduniversalmechanism.'),
      dict(id='m_only_sufficiency',outcome='Contraddice',argument='B7retains3Ca_LVAstresponsefailuresandbothstressresiduals. Activation-onlyisnotuniversallysufficienthere.'),
      dict(id='common_stress_cause',outcome='Indeterminata',argument='Onejointpackagefixesboth,butablationsseparateequilibrium-sensitiveboundaryfromrelaxation-sensitiveoffset.Auniquesharedcauseisnotsupported.')]
    output=dict(valid=True,findings=findings,evidence=evidence,preferred_arm='B6',post_hoc_nomination=True,
        frozen_checkpoint_sha256=audit['frozen_sha256'],task41_preparation_authorized=True,task41_executed=False,thresholds_changed=False,
        ablation_summary={k:dict(passed=sum(r['metrics']['passed'] for r in rr),total=len(rr)) for k,rr in ablations['results'].items()},
        next_step=dict(task=41,scope='Independent level1 confirmation;no furtherfitting',candidate='B6allthreefrozenseeds',
           requirements=['Newprotocols/seedsandparametricpanels,notrelabelledhistoricalfixtures','Sealresultsuntilcandidateandanalysisfrozen',
           'Pairedimposed/autonomouscurrent,state,V,Ca,spikeandinterventionmetrics;retainallrareandworstcases',
           'Matchedaccuracy/workionicandintegratedcost,uncertainty;accurate-but-slowallowed',
           'NoautomaticmulticompartmentGOornewabstractionlevel;stoponfreshfailureanddiagnosewithoutthresholdchange']))
    save_json(F/'synthesis.json',output)
    lines=['# GIADA Task40b — audit e scelta per Task41','','Audit post hoc; nessun training, nessuna modifica delle soglie.','',
        '| Variante | Development | Stress | Risposte autonome | Risposte imposte |','|---|---:|---:|---:|---:|']
    for name,r in audit['summaries'].items():lines.append(f"| {name} | {r['development']['passed']}/72 | {r['stress_diagnostic']['passed']}/72 | {r['autonomous']['differential_passed']}/168 | {r['imposed']['differential_passed']}/168 |")
    lines+=['','Tutte le varianti:264/264 traiettorie in ciascuna modalità. Sono casi×seed, non altrettanti protocolli indipendenti.','']
    for name,value in findings.items():lines.extend(['## '+name,'',value,''])
    lines+=['## Due stress originali','','| Variante | Ca_LVAst offset (ms; limite0.3) | Ca_HVA NRMSE (limite0.01) |','|---|---:|---:|']
    for arm in ('B0','B1','B6'):
        rr=[r for r in details['original_two_cases'] if r['arm']==arm]
        offset=next(r for r in rr if r['family']=='same_v_history')['metrics']['channels'][1]['event']['offset_error_ms']
        error=next(r for r in rr if r['family']=='activation_boundary')['metrics']['channels'][0]['nrmse']
        lines.append(f'| {arm} | {offset:.1f} | {error:.8f} |')
    lines+=['','GO alla preparazione della conferma indipendente Task41 con B6. Nessuna conferma indipendente già acquisita; nessuno speedup B6 dimostrato.']
    (F/'AUDIT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
if __name__=='__main__':main()
