# GIADA Task 17b — attribuzione causale dei rate Ca_HVA

Stato: follow-up **diagnostico** del NO-GO della Task 17a. Non modifica la
soglia 17a (errore massimo gate 0,01), non riabilita la LUT e non autorizza
Gate C. Il teacher resta quello nativo a 642 segmenti, con `USEION ca WRITE
ica`, cinetica `cnexp`, sinapsi autentiche, stessa sequenza di eventi e
perturbazione di `gCa_HVAbar` applicata a tutti i compartimenti pertinenti.

## Domande distinguibili

1. La deviazione è causata dai rate di attivazione m, da quelli di
   inattivazione h o da un'interazione fra essi?
2. Per m, è dominante l'errore di `mInf`, di `mTau` oppure il loro effetto
   congiunto non lineare nel circuito voltaggio–corrente–gate?
3. L'effetto emerge solo vicino a certi regimi/seed e intensità della
   conduttanza, anziché essere un errore uniforme del lookup?

## Matrice appaiata

- Protocollo fissato: `task17_nmda-tuft-n10-b2-r1-unpaired-branch-w400`, il
  regime in cui la Task 17a ha superato il limite gate.
- Seed diagnostico noto: `170029`. Nuovi seed dopo l'emendamento qui sotto:
  `170083`, `170097`.
- Moltiplicatori globali di `gCa_HVAbar`: `0.5`, `1.0`, `1.5`.
- Otto bracci: native, clone formula, LUT completa, solo `mInf` LUT, solo
  `mTau` LUT, coppia `mInf+mTau` LUT, solo `hInf` LUT, solo `hTau` LUT.
- Sample 0,025 ms, durata 40 ms; stessi quattro siti preregistrati in 17a:
  0, 387, 460, 469.

Il clone formula deve restare entro 0,05 mV di RMSE e 0,002 di errore massimo
dei gate su ogni confronto. La mancata corrispondenza è un **fallimento del
controllo**, non una prova contro la LUT. La compilazione e il probe della
tabella congelata precedono i trial. Ogni cambio di meccanismo crea una nuova
generazione di `SaveState`; non si ripristina uno snapshot di una struttura
precedente.

Contrasti primari: errore massimo m/h e RMSE di V di ogni braccio contro il
native, per seed × conduttanza × sito. La coppia `mInf+mTau` distingue
l'interazione m dalle due perturbazioni singole. I bracci h sono controlli
ortogonali. L'errore dei bracci in closed loop non è additivo: non si devono
sommare i massimi per dedurre causalità quantitativa.

## Emendamento trasparente, prima della seconda matrice

Una prima matrice **esplorativa** con seed `170029`, `170059`, `170071` e i
quattro bracci singoli (senza la coppia m) ha mostrato che i bracci h hanno
effetti minimi, ma il full LUT può amplificare la perturbazione più di
ciascun braccio m isolato. L'emendamento aggiunge il braccio `mInf+mTau`;
`170059` e `170071` sono ormai dati di sviluppo e **non** vengono presentati
come conferma nuova. I seed `170083` e `170097` sono stati fissati prima
della matrice emendata. Anche questi sono solo verifica diagnostica, non un
test sealed indipendente per la selezione finale.

Non sono previsti retraining, tuning della griglia o scelta di architettura
basata sui risultati di questa matrice. Un futuro candidato corretto richiede
nuova preregistrazione e test indipendenti.
