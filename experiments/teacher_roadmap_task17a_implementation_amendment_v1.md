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
