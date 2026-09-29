# GIADA Task 17a — emendamento tecnico dopo il preflight di compilazione

Data: 2026-09-29. Questo emendamento non cambia ipotesi, seed, protocolli,
perturbazioni, soglie o scelta LUT-513 del protocollo preregistrato (SHA-256
`9eff15aef204d1175e2c61776d72c2e1d2b7903424bf4fa28163e561ea0a8cc6`).

Il primo run Kaggle della revisione `4dd37ab` si è fermato **prima della
costruzione del teacher e di qualsiasi trial**: `nrnivmodl` di NEURON 8.2.7
ha tradotto la copia-formula, ma il traduttore NMODL è andato in segmentation
fault sul file LUT che usava `TABLE`. Nessun errore di voltaggio, gate o effetto
causale è stato misurato. Non attribuire questo crash alla learnability o
alla correttezza biologica della LUT.

L'implementazione viene cambiata da `TABLE` intrinseco a **interpolazione
lineare NMODL esplicita**, generata dalla stessa matrice float32 di 513 nodi
usata nella Task 16. Un albero bilanciato seleziona uno dei 512 intervalli;
nel singolo intervallo si interpola ciascuna delle quattro rate. Fuori dal
dominio [-135, 75] mV si usano i nodi estremi, come nel candidato congelato.
Restano identici `USEION ca READ eca WRITE ica`, `cnexp`, m/h, gbar e gli
ingressi causali. La prova numerica sul meccanismo compilato, con tolleranza
preregistrata 2e-5, resta un preflight obbligatorio. Se la nuova traduzione o
la prova di equivalenza fallisce, il microcanary resta NO-GO tecnico.

La revisione correttiva non rende retroattivamente valido il run fallito e non
autorizza Gate C. Nel mirror SQLite locale il protocollo conserva l'hash
originale e rimanda a questo emendamento come traccia della modifica.

## Secondo preflight: compilazione risolta, prima del primo trial

L'utente ha fornito lo ZIP
`giada_roadmap_task17_causal_replacement_retry_5ce7c8e.zip`, SHA-256
`4ed7f94a81a2085b8842c03d05a7c15e314bedf6068b4bbc0a4ebfed0618efe6`.
Il report della revisione `5ce7c8e` conferma
che i meccanismi espliciti compilano e il preflight sui dieci nodi campionati
ha errore assoluto massimo **zero**. Il teacher e il burn-in sono proseguiti
fino alla preparazione del primo trial. Lì il codice Python ha assunto
erroneamente che `audit.synapse_records` fosse un dizionario e ha chiamato
`.items()` su una lista: `AttributeError`. Nessun trial nativo/formula/LUT è
stato completato, quindi non esistono metriche causali.

La correzione usa `enumerate(synapse_records)` e, nell'audit statico
contestuale, reimposta esplicitamente `gCa_HVAbar` canonico prima di ogni
braccio: SaveState non è un contratto per la restaurazione dei parametri di
densità e una moltiplicazione cumulativa renderebbe non appaiati i livelli
×0,5/×1/×1,5. Il nuovo preflight include anche nodi intermedi della LUT.
Queste sono correzioni del driver e di validità sperimentale, non nuovi dati
per modificare il modello o le soglie preregistrate.

## Terzo guasto: SaveState dopo rimozione/reinserimento di meccanismi

Il tentativo Kaggle `114e4f7` non ha scritto né final_report né failure_report;
la cella di recupero ha poi mascherato il problema con FileNotFoundError.
Dal solo log Kaggle non era possibile conoscere il codice di uscita nativo.
Il driver precedente è stato quindi eseguito in Ubuntu/WSL locale, Python
3.12.3 e NEURON 8.2.7, con il teacher canonico compilato in una copia temporanea.
**Riprodotto SIGSEGV** in `SaveState.restore`, chiamato dal trial nativo di
ritorno dopo il braccio formula (riga 441 del driver precedente). Il problema
non è un fallimento biologico della LUT.

La nuova politica non ripristina mai un SaveState creato prima di una modifica
dei meccanismi, nemmeno quando il suffisso torna ad avere lo stesso nome:

1. Si completa il blocco nativo con il suo snapshot di equilibrio.
2. Si ripristina l'equilibrio ancora compatibile, si trasferisce il canale
   una volta e si salva un **nuovo** snapshot della struttura formula.
3. Si completano i 27 trial formula; ogni confronto resta appaiato al nativo.
4. Il ritorno al nativo genera un altro snapshot, non riapre quello originale;
   il controllo di ritorno deve ancora rispettare 1e-5 mV.
5. Si passa allo stesso modo alla LUT e si eseguono i 27 trial LUT.

Il trasferimento verifica la conservazione del confine ai siti registrati;
gbar canonico viene reimpostato prima di ogni episodio. Nessun teacher-state
successivo al confine entra nei rollout. Il cambiamento è nell'ordine di
esecuzione e nella gestione di SaveState, non nella matrice, nei seed, nelle
soglie o nella LUT. Questo emendamento sostituisce il dettaglio operativo
«ripristinare lo snapshot nativo prima di ogni braccio» della preregistrazione,
preservandone l'intento di condizioni iniziali appaiate.

Il runner ora ha un processo supervisore distinto dal worker NEURON:
`process.log`, `process_status.json`, `last_phase.json` e `failure_report.json`
restano disponibili anche se il worker termina per segnale. Un finale scritto
prima di un'uscita anomala non basta a dichiarare successo. I singoli episodi
completati vengono salvati progressivamente. Notebook e celle di recupero
usano output univoci senza cancellare i precedenti e non leggono report assenti.

### Verifica reale, non soltanto mock

- Due esecuzioni locali complete della matrice corretta: 81 episodi e quattro
  episodi di preflight ciascuna, uscita normale; la seconda è inclusa nella
  suite di integrazione.
- Suite Linux: **14 test, tutti passati**, 110.720 s complessivi, incluso
  SIGSEGV deliberato in un processo figlio per provare il recupero dei report.
- Copia-formula: massimo RMSE V **0**, massimo errore gate **0** nella matrice.
- LUT: massimo RMSE V **0.5933959189530174 mV** (limite 2), massimo errore gate
  **0.01835138564840766** (limite 0.01). Contrasto gbar valido, 11 effetti
  identificabili. Quindi **CAUSAL_MICROCANARY_NO_GO**, non un crash e non un GO.
- Questa è una validazione locale della matrice preregistrata, non una nuova
  replica con nuovi seed, non un risultato Kaggle e non una promozione Gate C.
  I due run ripetuti non aumentano il numero di repliche indipendenti.

Il report completo versionato è
`experiments/results/task17a_native_validation/final_report.json`, SHA-256
`99d9bd64d3b7b657242899195c2266d96674b9f62ddd46e567d762346c6b5666`.
La provenance registra HEAD di base `114e4f7`, working tree modificato e SHA-256
dei tre file eseguiti: non attribuire il fix al vecchio commit da solo.
Artefatti integrali locali: `artifacts/giada_task17_native_validation_20260929`.
Lo ZIP compatto conserva report, log, metriche e tracce a 1 ms; nessuna soglia
è stata adattata in base al risultato.

Riproduzione Linux (teacher già compilato con la versione fissata):

```bash
GIADA_TASK17_NATIVE_TEST=1 \
GIADA_TASK17_TEST_OUTPUT=/tmp/giada-task17-new-unique-run \
GIADA_NATIVE_TEACHER=/path/to/pinned/neuron_as_deep_net \
GIADA_TASK15C_ARTIFACT=/path/to/giada_roadmap_task15c_embedded_interface_bridge.zip \
python -m unittest tests.test_native_process tests.test_roadmap_causal_cahva_replacement -v
```
