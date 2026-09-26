# GIADA Task 16 originale — conferma embedded congelata Ca_HVA

Stato: **preregistrato, run Kaggle non ancora eseguito**. Notebook `notebooks/16_roadmap_embedded_frozen_confirmation.ipynb`. Segue la selezione Task 15c (report SHA-256 `ffde98ea986cc6dc293688b0e9335efb6af60639fe95281440df17523c4b2fc9`, codice `fb8a55cdb038628af6dd35cac9b576dfb4b3ff99`): candidato **LUT lineare 513** Task 5, congelato. Non è una conferma della MLP Task15b né una sostituzione causale (Task17).

## Domanda e interfaccia

La domanda è se la LUT che ha passato il ponte validation mantenga gate e corrente Ca_HVA quando viene eseguita come **shadow durante il replay nativo** del teacher NEURON completo a 642 segmenti. Ogni transizione è riprodotta dallo snapshot, con sinapsi, RNG, CVode e morfologia autentici. A ogni campione di 0,025 ms vengono letti V, m/h nativi, ECa e gbar dei siti preregistrati e viene aggiornato immediatamente lo stato shadow della LUT. Il candidato riceve il midpoint dei due campioni V del sottopasso; quindi usa un path teacher-forced **dopo** che il teacher ha avanzato quel sottopasso. Non è un input interamente disponibile all'inizio del millisecondo, né il candidato può modificare la membrana. L'aggiornamento LUT nel callback Python usa float32 e deve superare un preflight di equivalenza numerica (`max abs error ≤5e-7`) con `gpu_lut_predict` Task5 su otto casi fissati; la GPU è impiegata per questa verifica, mentre l'aggiornamento shadow durante la simulazione è CPU perché NEURON avanza sequenzialmente.

Non confrontare con `segment.ica`, che è corrente ionica aggregata. Per isolare Ca_HVA, confrontare la corrente analitica `gbar·m²·h·(V−ECa)` calcolata dai gate nativi e dagli stati shadow sullo stesso V/ECa/gbar. Il controllo formula integra separatamente i gate lungo lo stesso path midpoint; così il suo errore misura il floor d'interpolazione, distinto dall'errore LUT.

## Fonte, campionamento e replay

Dataset targeted v1.1 base originale (HDF5 SHA-256 `3fef415544a82b55801461e3cec069ed292faca0075f1f9f431e9dce8f5ea6d8`), più snapshot nativi immutabili. Solo split **`deterministic_test`**, non usato dalla Task15c. Seed di campionamento 16017; siti 0, 387, 460, 469; regimi quiet, rising, spike, falling classificati dalla microtraccia V. Fino a 8 coppie sito×transizione per gruppo, minimo 4 per tutti i 16 gruppi. Le transizioni uguali selezionate per più siti vengono rigiocate una volta sola. Nessun modello viene scelto né riaddestrato sul test.

Prima di valutare la LUT: verifica SHA fonte, Task5 e Task15c; confronto completo degli stati e delle sequenze RNG del replay contro HDF5 (tolleranza `1e-5`); confronto di ciascuna traccia V e dei gate m/h nativi finali contro lo store (tolleranza `1e-5`). Se il replay fallisce, produrre **NO-GO diagnostico** e non attribuire l'errore al candidato. Copiare soltanto gli snapshot necessari nella directory di lavoro; non modificare la fonte Kaggle.

## Metriche e decisione fissate ora

Misurare per gruppo e globale: RMSE endpoint m/h e `m²h`, violazioni di occupanza; RMSE sul path di corrente Ca_HVA analitica in mA/cm², formula midpoint e LUT rispetto ai gate nativi; supporto, provenienza e massimo errore del replay. Il gate embedded teacher-forced passa soltanto se:

1. tutti i 16 gruppi hanno ≥4 coppie sito×transizione;
2. il replay completo passa;
3. per ogni gruppo, sia RMSE m sia h della formula midpoint verso il teacher sono ≤`0.005`;
4. per ogni gruppo, sia RMSE m sia h della LUT verso il teacher sono ≤`0.005`, con zero violazioni di occupanza;
5. per ogni gruppo, RMSE della corrente Ca_HVA shadow lungo i 41 campioni è ≤`0.0001 mA/cm²`.

Riportare i valori continui e i gruppi anche se una soglia fallisce; non cambiare soglie dopo il test. Un passaggio qui conferma solo il candidato **shadow e teacher-forced** sul test indipendente. Non certifica input causali disponibili al confine di 1 ms, sostituzione della membrana, accuratezza di spike/voltaggio in rollout né Gate C. Queste domande appartengono alla Task17 e ai gate successivi della roadmap.
