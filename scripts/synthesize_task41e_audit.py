"""Synthesize saved 41e evidence; propose, but do not launch, Task41f."""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'experiments/results/task41e_audited_v1'


def main():
    probe=json.loads((OUT/'residual_probe.json').read_text(encoding='utf-8'))
    external=ROOT/'artifacts/task41e_reserve_external'
    audit=json.loads((external/'final_detailed_audit.json').read_text())
    final=json.loads((external/'final_report_recovered.json').read_text())
    backup=json.loads((external/'final_recovery_receipt.json').read_text())
    limits=('Audit post hoc su pannelli già aperti, non conferma indipendente. Soglie originali immutate. '
        'Tre seed del modello condividono i protocolli: non sono repliche biologiche indipendenti. '
        'Il freeze originale resta Q0; H10 è una variante diagnostica, non una nuova selezione. '
        'Nessun GO a Task42, deployment o accelerazione qualificata end-to-end. '
        'Reranking e rescoring sono analisi di sensibilità, non sostituzioni retroattive del risultato.')
    findings={
      'outcome':'41e terminata a 60000 aggiornamenti, exit0, valid=True, TRANSFER_DIAGNOSIS_NO_GO, nessun braccio supera tutti i gate. Tutti gli11 checkpoint/artefattiTorch recuperati hanno tensori finiti. Il problema HTTP500 era di accesso ai risultati, non evidenza di training bloccato.',
      'recovery':'Archivio finale completo recuperato fuori runtime:206 membri verificati con dimensione/CRC contro inventario finale remoto e ZIP integro. Ricompressione di base+delta: contenuto finale identico, byteZIP diversi dal remoto. Nessun riavvio/interruzione del worker, retraining o rivalutazione scientifica per il recupero.',
      'quality':'Sul medesimo pannello41c: P110334/336 risposte autonome e31/36 stress; Q0 nominata334/336 e33/36; H10 diagnostica335/336 e33/36, vecchi stress36/36. D_C01 già335/336 e32/36, D_combined334/336 e32/36 con vecchi stress34/36. H10 combina un miglioramento rispetto al singolo C01 sugli stress con assenza delle regressioni della composizione. Non è un GO.',
      'regressions':'Tutti i nuovi bracci:528/528 traiettorie nei pannelli41 e41c in entrambe modalità,336/336 risposte imposte e336/336 autonome41; storico40b264/264 traiettorie e168/168 risposte, recurrence72/72 development e72/72 stress. B6 controllo resta527/528 traiettorie e riproduce il risultato storico. A01 introduce stress41c caso0 seed43; H01 introduce vecchio stress41 caso1 seed17. Conteggi uguali possono nascondere scambi di identità.',
      'residuals':'Q0: development35/36, unico fallimento Ca_HVA offset0.4ms caso2 seed43; stress41c33/36, caso1 slow_boundary_return tutti3seed con offset0.8/0.7/0.8ms contro0.3ms; NRMSE corrente Ca5.56e-5/5.82e-5/6.37e-5. Risposte autonome: Nap caso122 seed17 NRMSE0.0144197 e caso108 seed43 0.0100642 contro0.01. H10: stesso development, stessi3stress offset0.9/0.7/0.8, risolve caso122 seed17 ma non108 seed43. Q0 peggiora il margine122 rispetto aP1100.011357 pur migliorando gli stress.',
      'selection':'25/36 membri selezionano step0. Il ranking ricostruisce il fallimento con severity>1, ma il valutatore canonico applica la tolleranza numerica già esistente1e-10ms sul timing. Valori0.30000000000000004 risultano quindi falsamente falliti nel ranking. Disaccordi su4-10 membri percheckpoint nonzero. Reranking diagnostico usando i predicati ufficiali cambia5/36checkpoint: A10seed43,H10seed43,H01seed43,J10seed43,J11seed17. H10/H01/J11 arriverebbero36/36 nel development già salvato. Q0 e il suo verdetto restano invariati; nessun nuovo test o candidato promosso.',
      'reference':'La suite ufficiale usa riferimento canonico TorchFP32; i probe di accoppiamento usano riferimento canonicoFP64 e batch ridotto. Nelle6 risposte108/122 i4controlli FP32 passano6/6 controFP64, ma4/6,5/6,5/6,4/6 contro il riferimento originale, riproducendo i conteggi dei fallimenti. Quindi il pass dei probe non prova la riparazione. Le stesse varianti ricalcolateFP64 nei probe passano6/6 anche contro il riferimento originale. Il floor canonicoFP32vsNumPy sulle risposte108/122 vale0.0080072/0.0076503 NRMSE su soglia0.01. Forte evidenza di sensibilità numerica; non autorizza a cancellare fallimenti né a cambiare riferimento primario post hoc.',
      'feedback':'A60000 la loss comune40tick di selezione è superiore all iniziale in tutti36membri (non necessariamente nei checkpoint selezionati). A00seed29 peggiora45.29x; A01 circa1.05-1.11x. Il bilanciamento dinamico riduce molto la deriva delle norme, ma non basta al miglioramento globale. Le nuove loss/code/supporti non dimostrano un vantaggio uniforme:0/15 contrasti registrati superati. Il difetto di ranking limita anche la lettura di questi contrasti.',
      'representation':'41e può modificare solo u0 e u8, cioè gli equilibri dei gate m diCa_HVA eNap_Et2. Tutti i coefficienti di rilassamento z e il gate h diCa_HVA restano congelati. È una restrizione identificabile: aumentare il budget sulla stessa correzione u non garantisce di correggere errori di rilassamento/timing. Non è ancora dimostrato che z/h siano la causa.',
      'fusion':'D_combined->D_fused: parent.raw da3chiamate a1, raw bitwise identici e errori di rolloutV/Ca/gate/corrente tutti0. Riduzione tempo integrato mediano31.87%,41.16%,53.49% ai batch1,642,4096 perseed. È una ottimizzazione equivalente reale; non un nuovo successo di qualità.',
      'speed_scope':'Il precedente circa8x è reale nel benchmark residente isolato: Task39c circa7.843-8.010x,80tick catturati in un unico CUDAgraph,V/Ca imposti, singolo modello perseed. La41e usaCUDAgraph perstep, cicloPython,copie perstep,3seed insieme, Caautonomo e diversi carichi/readout; lo scopeintegrated aggiorna ancheV. Non è un confronto appaiato, quindi non si può attribuire la differenza alla sola astrazione o al solo accoppiamento. Nuovo benchmark deve separare fisica e granularità di esecuzione.',
      'cost':'41e batch642perseed integrato80tick: formula106.031ms,P11081.477,D_combined174.900,D_fused102.889,Q0101.255,J11101.499. A4096 formula126.294ms,Q0237.065ms. Le correzioni aggiungono costo rispetto aP110; tuttiquality_qualified=False. Nessun8x dimostrato sul compartimento completo.',
    }
    hypotheses=[
      dict(id='numerical_response',evidence='reference',status='Aperta',priority='Alta',
        statement='I residui Nap sulle minuscole differenze causali contengono una quota numerica dominante rispetto all errore appreso.',
        test='Stessi pesi,stessi casi,stesso batch,stessa fase: candidatoFP32/mixed/FP64 e riferimentoFP32/FP64/NumPy; congelare il riferimento ufficiale per tutti i pass/fail. Isolare precisione diV,gate,readout e sottrazione; conversione aFP64 dopo la simulazione non recupera bit persi. Misurare errori con segno,ULP,floor e margine di ciascun caso. I riferimenti alternativi sono diagnostici.',
        falsifier='Il fallimento resta con simulazione/riferimento convergenti o supera largamente il floor; la parte appresa resta necessaria anche dopo precisione controllata.'),
      dict(id='relaxation_identifiability',evidence='representation',status='Aperta',priority='Alta',
        statement='L offset Ca_HVA non è identificabile correggendo soltanto l equilibrio m: può richiedere rilassamento z o inibizione h.',
        test='Oracoli esclusivamente diagnostici per u_m,z_m,u_h,z_h, singoli e combinazioni mirate sui replay esatti. Errori di crossing con segno, pendenza locale, occupancy e residui dei coefficienti. Poi confrontare correzioni apprese u-only e u+z, lasciando18stati,solver,readout e astrazione invariati; eventuale h solo con motivazione preregistrata.',
        falsifier='u-only diagnostico elimina tutto il residuo mentre z/h non aggiungono vantaggio; oppure il floor numerico spiega il timing residuo.'),
      dict(id='selection_semantics',evidence='selection',status='Aperta',priority='Certa incoerenza tecnica; impatto generale da testare',
        statement='La diversa semantica del confine0.3ms scarta checkpoint validi e amplifica la scelta step0.',
        test='Un unico predicato canonico nel ranking e nella valutazione, test di bordo0.3/0.4ms e regressione su tutti i JSON salvati. Il confronto retrospettivo resta diagnostico; nuova selezione fissata prima di dati nuovi.',
        falsifier='Dopo allineamento predicati e nuovo development, checkpoint e qualità non migliorano. Non spiega da solo il NO-GO diQ0, rimasta identica nel reranking.'),
      dict(id='rare_tail_support',evidence='residuals',status='Aperta',priority='Media',
        statement='Le code lente e le risposte minuscole richiedono supporto/budget mirati, ma il supporto stratificato41e non era sufficientemente allineato al bersaglio numericamente affidabile.',
        test='Supporto ampio/arricchito da soli protocolliFIT indipendenti con code e crossing lenti. Stesso numero ancore iniziale,poi2livelli dati e checkpointbudget lungo la stessa traiettoria; registrare esposizione perregime e compute, non solo update. Nuovi test separati; non riciclare i casi aperti per il fit.',
        falsifier='La copertura aumenta e le learningcurves si saturano senza migliorare le metriche di coda/risposta a obiettivo affidabile: il limite non è soltanto campionamento/budget.'),
      dict(id='objective_transfer',evidence='feedback',status='Aperta',priority='Media-alta',
        statement='Ridurre errori puntuali non dà una direzione utile per la risposta accoppiata; loss e gradienti misurano o inseguono anche residui numerici.',
        test='Su stessi batchFIT: gradienti per loss, coseni, effetto finito di singoli passi e loss su finestre disgiunte. Confrontare coefficient-only e loss di risposta coerente sul nucleo numericamente auditato, con frequenza/budget bilanciati. Se la loss di risposta non scende nemmeno nelFIT controllato, fermare lo scaling.',
        falsifier='LossFIT/holdout e direzioni migliorano coerentemente ma il fallimento resta circoscritto al supporto nuovo: prevale generalizzazione o capacità.'),
      dict(id='execution_granularity',evidence='speed_scope',status='Aperta',priority='Alta per spiegare il trasferimento8x',
        statement='Una quota del mancato trasferimento della velocità dipende da confini dei CUDAgraph,copie e batching, non dalla funzione appresa in sé.',
        test='Stessi pesi,dtype,batch,80tick e GPU: whole-loopgraph/per-stepgraph x ingressiimposti/sistemaaccoppiato, formule e rete entrambi ottimizzati allo stesso modo; verificare equivalenza prima dei tempi. Riportare walltime eCUDAevents,costo ionic ecompleto, parent/correzioni/readout. Nessuna promessa8x.',
        falsifier='Il vantaggio resta assente dopo confronto equivalente eprofiling mostra costoaritmetico/banda dominante nelle correzioni anziché overhead esecuzione.'),
    ]
    old_evidence=[]
    for key in ('objective','gradient_balance','feedback_path','horizon_support','joint_protection','selection','cost'):
        finding={'objective':'feedback','gradient_balance':'feedback','feedback_path':'reference','horizon_support':'feedback','joint_protection':'quality','selection':'selection','cost':'fusion'}[key]
        old_evidence.append(dict(id=key,state='Supportata nel dominio' if key=='cost' else 'Aperta',
            outcome='Sostiene' if key=='cost' else 'Indeterminata',finding=finding,
            argument=('Il riuso equivalente del padre riduce il tempo, senza cambiare numeri.' if key=='cost' else
                'Evidenza aggiornata41e: '+findings[finding]+' Nessun vantaggio generale dimostrato per il contrasto originario.')))
    proposed=dict(name='Task41f — precisione, rilassamento e trasferimento',status='Proposta; non preregistrata né avviata',
        stage0='Preflight congelato: riprodurre esattamente i fallimenti originali prima di applicare interventi. Unificare predicati di soglia già esistenti. Distinguere FP32riferimento/candidato e dimensionebatch. Nessun cambiamento dei criteri di accettazione.',
        frozen_matrix='Precisione/riferimenti e clamping sui medesimi replay; oracoli separati di equilibrio/rilassamento m/h solo perattribuzione. Controlli congelati P110,Q0,H10,C01,D_combined,D_fused. Tutti i risultati originali preservati.',
        training_matrix='Se i residui restano apprendibili dopo audit numerico:6bracci,3parametrizzazioni x2supporti. u-only con64basi;u-only con128basi;u+z con64basi (controllo capacità a budgetparametri confrontabile). SupportoFIT ampio/arricchito di code-risposte minuscole,stesso numeroancore. Targetgate da motivare con probeFIT/preregistrazione,non scegliere sul test.18stati e stessa astrazione;eventuale h ècorrezione appresa,nonfallbackcanonico. Checkpointprogressivi e learningcurves perregime;numeroesatto update dopo microbenchmark.',
        objective_probe='Diagnostica gradienti e passi appaiati coefficient-only/feedback senza moltiplicare indiscriminatamente trainingcompleti. Integrare confronto completo solo seilFITprobe distingue leipotesi e il budget resta sensato. Nessuna interpretazione di un fallimento dellottimizzatore come prova di impossibilità.',
        execution_matrix='Stessi modelli congelati:CUDAgraph perstep/interoloop x fisicaimposta/accoppiata. Formule e rete trattate simmetricamente; equivalenza numericaobbligatoria prima delbenchmark. Separare velocità da GOqualitativo.',
        freeze='Nuovi seedFIT/selection/development; regressioni note dopo freeze,stress invariati,confermaindipendente successiva. H10 non viene promossa sui test aperti. Identità dei casi e margini,non solo conteggi.',
        thresholds_changed=False,task42_authorized=False,DB_backup='locale',launched=False)
    out=dict(valid=True,limits=limits,findings=findings,hypotheses=hypotheses,original_claim_evidence=old_evidence,
        next_experiment=proposed,memory_links=probe['memory_query']['parameters'],backup=backup,
        final_decision=final['decision'],summaries=final['summaries'],thresholds_changed=False,
        contrasts=audit['contrasts'])
    (OUT/'synthesis.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(dict(findings=len(findings),hypotheses=len(hypotheses),new_run=False)))


if __name__=='__main__':main()
