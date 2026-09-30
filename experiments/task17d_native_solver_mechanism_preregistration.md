# GIADA Task 17d — meccanismo della sensibilità numerica nativa

Stato: protocollo diagnostico prima di aprire i risultati Task 17d. La Task17a
resta NO-GO. Task17c ha già mostrato che la griglia dei rate m conta, ma la
traiettoria nativa cambia con le tolleranze CVode. Questa task non seleziona
LUT, rete o nuove soglie Gate C.

## Condizioni e interventi

Teacher canonico a 642 segmenti, protocollo NMDA della Task17c, siti
0/387/460/469, condizioni note seed 170029 × gbar 1,0; seed 170083 × gbar
1,5; controllo debole seed 170029 × gbar 0,5. Finestra 60 ms per osservare
anche la risposta dopo i 40 ms della Task17c. Stesso SaveState, schedule,
Random123 e ordine degli interventi per ogni condizione.

Bracci nativi per tutte le condizioni:

1. `default`: CVode del teacher (`atol=0,001`, `rtol=0` attesi).
2. `tight`: `atol=1e-5`, `rtol=1e-6`.
3. `ultra`: `atol=1e-7`, `rtol=1e-8`.
4. `calcium_scaled`: tolleranze default, `atolscale` dello STATE cai
   effettivamente identificato a runtime moltiplicato per `1e-4`.
5. `voltage_scaled`: tolleranze default, `atolscale('v')=1e-2`.

I due bracci selettivi sono sonde di attribuzione numerica, non candidate
settings di produzione. Un audit runtime registra nome degli STATE, scale
prima/dopo e interrompe il run se il target cai non è identificabile o se la
scala non viene applicata. Non si inferisce la scala dal solo file MOD.

Controllo sul driver: per i due casi critici, un replay nativo `ultra` a
campionamento 0,005 ms, contro 0,025 ms, con stessi input. Il campionamento
richiede chiamate `CVode.solve` aggiuntive e potrebbe pertanto modificare
anche la traiettoria: la differenza è un effetto del driver, non mero errore
di osservazione.

## Predizioni discriminanti

- Se lo STATE del calcio è il principale limite di scala, `calcium_scaled`
  sposta la traiettoria verso `ultra` più di `voltage_scaled` nei casi critici.
- Se domina la precisione del voltaggio, si osserva il pattern opposto.
- Se domina l'amplificazione vicino alla soglia, piccole differenze precoci
  possono produrre timing o plateau tardivi molto diversi; il controllo
  debole dovrebbe essere meno sensibile. Non si dichiara una biforcazione
  formale sulla base di queste tre condizioni.
- Se il driver è rilevante, `ultra` cambia al variare della cadenza di solve.
- Se i rilasci differiscono fra bracci, non si attribuisce la divergenza
  soltanto all'integrazione della membrana.

Metriche: V, m, h, cai, stato SK quando esposto, correnti locali; RMSE raw,
massimi, attraversamenti di 0 mV, ampiezza e timing dei picchi, risposta
20–60 ms, allineamento temporale solo diagnostico. La metrica raw resta il
criterio primario. Hash degli esiti di rilascio e verifica di frontiera per
ogni millisecondo. Ripetere il nativo a impostazioni identiche in preflight.
Reportare anche costo CPU; nessuna inferenza su speedup GPU.

Stop per teacher non canonico, mismatch SaveState, 642 segmenti assenti,
replay identico fallito, scala non applicata, rilascio causale non verificato,
NaN/Inf. In caso di stop, salvare failure report e log tramite supervisore.

L'esito resta diagnostico anche se un braccio selettivo coincide con `ultra`:
prima di cambiare il teacher o validare un candidato servono convergenza
verificata e protocollo indipendente. I risultati Task17d vanno nel solo
mirror SQLite dopo verifica dell'artefatto; Airtable resta congelato.
