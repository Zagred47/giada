"""Evidence-linked interpretation, not a preregistration or model selection."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.giada_teacher.task37_ionic_playground import save_json
OUT=ROOT/'experiments/results/task41_audited_v1'

def main():
    read=lambda n:json.loads((OUT/n).read_text())
    audit=read('audit.json');assert audit['valid'] and audit['stress_probe_passes']['three_local_coefficient_oracles']==36
    nats=read('nats2_coefficient_interventions.json')['interventions']
    assert nats['oracle_z_gate7']['17']['metrics']['offset_error_ms']==0
    findings={
        'outcome':'Task41 valida dopo la correzione tecnica della semantica sinaptica: B6 congelata passa527/528 traiettorie in ciascuna modalità,336/336 risposte differenziali in ciascuna modalità e33/36 stress. Le176 traiettorie e12 storie stress sono ripetute sui3 seed di modello, non528/36 protocolli indipendenti. NO-GO preservato.',
        'localization':'Un solo caso/seed di traiettoria fallisce in entrambe le modalità: delayed_doublet,canonical,inputseed104117,Viniziale-56mV,iniezione1.01,modello17,NaTs2_t. Tre stress activation_boundary con inizializzazione-42mV: Nap_Et2 input104173 seed17/43 eIm input104143 seed43. Tutti falliscono solo offset; correnti/carica/gate finiti e qualificati, nessuna uscita dal dominio o errore di conteggio spike nei casi accoppiati.',
        'secondary_crossing':'NaTs2_t non mostra una traslazione di7.3ms: riferimento e candidato hanno lo stesso intervallo principale7.2–9.7ms. Solo B6 seed17 supera di2.035e-10mA/cm2 la soglia relativa al campione17ms del secondo impulso; il riferimento resta sotto di4.537e-10. La definizione last-active produce il salto9.7→17ms. L’errore di corrente al campione è6.571e-10; il massimo sull’intera traccia2.058e-8. Nessun filtro, durata minima o modifica della soglia viene applicato.',
        'nats_mechanism':'Replay congelato: FP64 dello stato eoracolo del solo equilibrio non risolvono NaTs2_t. Sostituire solo z del gate7(h) con il coefficiente canonico porta offset7.3→0ms; sostituire z del gate6(m) non basta. Anche lo swap della sola traiettoria h risolve, quello m no. Questo localizza una sensibilità al recupero/inattivazione appresa, non prova la necessità di più stato nascosto.',
        'nap_mechanism':'Nap_Et2 offset0.5/0.7ms conNRMSE4.11e-5/5.50e-5. Correggere solo u del gate8(m,attivazione rapida) porta a0.1/0.2ms. Correggere solo u del gate9(h,lento) o z non basta. Un piccolissimo bias del gate veloce riscalando m^3h altera il crossing della coda lenta: non ogni errore su tempi lunghi è causato dal gate lento o da memoria insufficiente. Al voltaggio di coda-45.516mV il bias u_m è1.27e-5/1.63e-5.',
        'im_mechanism':'Im seed43 offset0.6ms. Oracolo del solo z gate11 porta a0ms; solo u porta a0.7ms. Nel plateau-40.932mV z è sottostimato di1.12%,tau appresa43.9326ms contro43.4401ms. La pendenza piccola della corrente rende questo bias visibile nel timing.',
        'numerical_floor':'Tre fallimenti stress persistono con coefficienti appresi FP32 e stato FP64(33/36). Formule canoniche con coefficienti/stato FP32 passano36/36, sia delta sia combinazione convessa; il controllo Torch analitico32 passa36/36 suCPU. Il roundoff da solo non spiega questi residui. Il controllo stress FP32 mancava nel report originale ed è ora aggiunto come diagnostica post-hoc, non come nuova conferma CUDA.',
        'three_functions':'Oracoli locali su u(Nap.m),z(NaTs2.h),z(Im) passano36/36 stress; z(NaTs2.h) risolve separatamente il caso accoppiato aV imposto. Sono interventi diagnostici su dati aperti, non un candidato apprendibile già ottenuto né una prova di guarigione closed-loop complessiva.',
        'comparison40b':'40b B6 aveva264/264 traiettorie,168/168 risposte e72/72 stress sufixture storiche aperte. Task41 usa nuovi protocolli/pannelli/storie: non è una regressione misurata sugli stessi casi. I residui non sono nei due canali calcio riparati in40b; tutte le loro verifiche di corrente passano sul nuovo pannello. Il nuovo problema è precisione trasferibile suNaTs2_t,Nap_Et2,Im.',
        'cost':'SuT4 il risparmio mediano B6 vsformule è65.9%/52.9% nel core ionico(batch1/642perseed) e43.9%/35.0% integrato. A4096perseed B6 èpiùlenta di8.8%/5.8%; èpiùlenta dellaLUT in tutti i6scenari. Nessuno speedup qualificato per qualità e nessuna generalizzazione hardware; aggiungere correttori senza misurare il costo può peggiorare il limite.',
        'decision':'Non promuovere Task42. Proporre Task41b diagnostica mirata, senza ridurre l’astrazione, senza formule canoniche in inferenza e senza cambiare le soglie. Task41 ormai aperta resta una regressione diagnostica; dopo eventuale riparazione serve una nuova conferma indipendente congelata.'}
    hypotheses=[
        dict(id='nats_h_rate',statement='Il bias del coefficiente di rilassamento NaTs2_t.h causa il falso crossing del secondo impulso nel caso aperto.',status='Supportata nel dominio',evidence='nats_mechanism',falsifier='Il solo oracolo z_gate7 non elimina il crossing in replay appaiato; nuovi protocolli potrebbero mostrare cause differenti.'),
        dict(id='nap_fast_equilibrium',statement='Il bias di equilibrio diNap_Et2.m domina i due offset della coda lenta osservati.',status='Supportata nel dominio',evidence='nap_mechanism',falsifier='Il solo oracolo u_gate8 non porta entrambi entro0.3ms; non implica sufficienza su altri regimi.'),
        dict(id='im_relaxation',statement='Il bias di z_Im domina il residuo Im seed43 osservato.',status='Supportata nel dominio',evidence='im_mechanism',falsifier='Oracolo z non ripara offset mentre altri fattori sì; verifica già positiva solo sul caso aperto.'),
        dict(id='roundoff_only',statement='I residui Task41 sono spiegati dal solo arrotondamento dello stato FP32.',status='Contraddetta nel dominio',evidence='numerical_floor',falsifier='Stato FP64 con identici coefficienti appresi mantiene gli stessi fallimenti, mentre formule32passano.'),
        dict(id='coverage',statement='L’esposizione FIT alle regioni ad alta sensibilità è insufficiente a imparare precisione trasferibile.',status='Aperta',evidence='three_functions',falsifier='A uguale N/budget/basis, stratificazione FIT-only non migliora coefficienti holdout, margini eventi e gate completi nei3seed.'),
        dict(id='capacity',statement='La risoluzione locale della correzione appresa limita la precisione dei tre coefficienti.',status='Aperta',evidence='three_functions',falsifier='Raddoppiare risoluzione a parità di dati/obiettivo non riduce il residuo holdout dopo convergenza; considerare costo e non solo parametri.'),
        dict(id='objective_conditioning',statement='La loss puntuale media sottopesa piccoli errori molto amplificati dalla pendenza di coda o dal margine del secondo impulso.',status='Aperta',evidence='secondary_crossing',falsifier='Una loss FIT di correnti condizionate controllata per gradienti non migliora timing rispetto ai coefficienti soli, o introduce regressioni.'),
        dict(id='budget',statement='A supporto/basis fissati, il residuo è ancora riducibile aumentando il budget di ottimizzazione.',status='Aperta',evidence='three_functions',falsifier='Curve multi-checkpoint e residui FIT/holdout saturano senza riduzione della severità locale e senza miglioramento tra seed.')]
    proposed=dict(name='Task41b: local kinetic precision and threshold-margin diagnosis',status='proposal_not_preregistered_not_launched',
        frozen_control='B6,all3originalseeds; parent bitwise frozen, zero-initialized bounded learned residuals.',
        targets={'u':[8],'z':[7,11]},
        factorial_axes={'support':['uniform independent FIT','sensitivity-stratified independent FIT at equal sample count'],
            'local_resolution':[32,64],
            'objective':['coefficient supervision','same coefficient loss plus bounded conditioned-current loss']},
        members='8pairedarms x3seeds + frozenB6control; optional joint-u/z sidecar only if fit-only preflight shows a clear diagnostic need.',
        budgets='Increasing checkpoints along the same optimizer trajectory; exact budgets/learningrate fixed using FIT-only preflight, not Task41 outcomes.',
        data='Independent atomic anchors, held steps, slow tails and double/triple pulses with randomized recovery interval and initial state. Stratification from canonical FIT sensitivity only; no copying Task41 exact voltages/seeds intoFIT.',
        shared='Paired initializations,anchor identifiers where possible,minibatch schedules and evaluations;report sample exposures,walltime and parameter count.',
        contrasts='Coverage at fixed resolution/objective;resolution at fixed support/objective;objective at fixed support/resolution;budget within each trajectory. Report all4conditional contrasts per axis, not winner-only attribution.',
        diagnostics='u/z errors pergate andV,tail slope,error/slope estimate,eventthresholdmargin,all connected active intervals,first/secondpulse ratio,history dependence;eventwise matching is supplementary and never replaces original last-active score.',
        selection='New development split with preregistered all-case quality and per-seed guards;old39j/40b/41panels only opened diagnostic/regression sets. Do not select checkpoints on Task41stress and relabel independent.',
        gates='Exact Task37/40/41 thresholds;no smoothing or excluded failures. No regression in calcium,SK,otherchannels;autonomous responses,state,current,spikes anddomain checks unchanged.',
        runtime='Vectorized independent members onGPU if equivalence/isolation preflightpasses;CPU/GPUnumericalfloor and no-canonical-fallback tests;measure identical resident formula/LUT/B6/candidate at all3batches.',
        success='All unchanged diagnostic/development gates plus frozen new independent confirmation before Task42;quality andspeed remain separate. Three diagnostic oracle functions being sufficient does not guarantee SGD can learn them.',
        backup='SQLite local only;runtime logs/checkpoints copied outsideVM;noDriveDBupload.')
    evidence=[dict(id='1',outcome='Contraddice',state='Contraddetta nel dominio',finding='outcome',argument='Strict all-case native generalization fails one case-seed in each mode;527/528 is strong partial coverage,notcompletequalification.'),
        dict(id='2',outcome='Sostiene',state='Supportata nel dominio',finding='outcome',argument='All336pairedresponsespass in eachmode onthe112interventions x3frozenmodelseeds;notuniversalityandnotwaiverofabsoluteoffsetfailure.'),
        dict(id='3',outcome='Contraddice',state='Contraddetta nel dominio',finding='localization',argument='Only33/36freshlonghistoriespass;threeoffsetfailures localized byposthocoracles.'),
        dict(id='4',outcome='Sostiene',state='Supportata nel dominio',finding='cost',argument='Costs are separately measured and quality fails;claim is separation ofqualification,not universal acceleration.')]
    links=[dict(code='findings-giada-task39j-result-sufficient_package-v1',lesson='A frozen parent with a small learned local correction previously worked; motivates local repair,notproofitwillworkagain.'),
        dict(code='findings-giada-task39j-result-freeze_contrast-v1',lesson='Preserve previously learned functions and measure regressions;pastunfreezingharmwasbudget/objective-specific.'),
        dict(code='findings-giada-task40b-result-ablation-v1',lesson='Different mechanisms require different u/z corrections;avoidoneuniversalrelaxationhypothesis.'),
        dict(code='lessons-giada-task39f-result-temporal-v1',lesson='A temporal objective previously suffered gradient domination;boundedlossaloneisnotenough,measuregradientnormsandmatchedobjectivecontrasts.'),
        dict(code='lessons-giada-task39h-result-temporal-v1',lesson='Allgatechecks andsmallwaveformerror donotguaranteetiming;reusepergateoraclesbutretestonthecurrentcandidate.')]
    save_json(OUT/'synthesis.json',dict(valid=True,findings=findings,hypotheses=hypotheses,original_claim_evidence=evidence,
        memory_links=links,next_experiment=proposed,training_performed=False,thresholds_changed=False,task42_authorized=False))
    print('Task41 evidence-linked synthesis saved')

if __name__=='__main__':main()
