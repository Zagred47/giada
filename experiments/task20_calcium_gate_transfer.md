# 🧬 Task 20 originale — gate dipendente dal calcio SK_E2

## 🎯 Domanda

Trasferiamo la cella vincolata a un meccanismo che legge `cai`, non V.
Il MOD canonico ha `zInf = 1/[1 + (0.00043/cai)^4.8]` e `zTau = 1 ms`.
Manteniamo tau nota e la corrente `gbar*z*(V-ek)` analitiche: la rete impara
solo zInf. Il ramo `ca < 1e-7` aggiunge 1e-7 al calcio locale della rate;
non va sostituito silenziosamente con un clamp continuo.

## ⚗️ Matrice unica

36 modelli indipendenti: 3 input × 2 obiettivi × 2 capacità × 3 seed.

- Calcio logaritmico: candidato primario preregistrato.
- Calcio lineare: stessa informazione, diversa rappresentazione.
- Solo voltaggio: controllo negativo, calcio indipendente e non visibile.

Stream, inizializzazioni e minibatch sono appaiati. Checkpoint a 0/1k/5k/15k/30k
sulla stessa traiettoria. Parametri 321/1153, tau non appresa. La promozione
richiede il candidato primario in tutti i seed: nessun ripiego scelto dal fresh.

## 🔬 Audit e conferma

Prima del training, un processo solo NEURON verifica rate e transizioni, ramo
sotto soglia, tau e indipendenza da V. Un processo GPU separato evita la
coesistenza nativa che in passato causava crash. Equivalenza vettorizzata/Adam
indipendente verificata prima del training.

Selezione solo development e freeze hash prima del fresh. Conferma su dominio
logaritmico, coda a basso calcio, stati 0/1, rollout a calcio costante e sequenze
di calcio imposto (dwell 1/5/25/100 ms). LUT logaritmiche 513/2049 come riferimento.
Le soglie numeriche sono nel JSON preregistrato; OOD separato.

## ⚠️ Limiti

Il calcio imposto non è il calcio endogeno: non abbiamo testato CaDynamics,
feedback calcio-voltaggio né il neurone completo. Un percorso costante durante
ogni ms non dimostra sufficienza di un campionamento ms per percorsi biologici
variabili dentro quel ms. Task 21 riguarda la dinamica lenta e Task 22 la
condivisione: non sono anticipate da questa Task 20.

Task 19 ha confermato accuratezza ma il MLP eager era più lento di formule/LUT.
Non si assume speedup e non si usa una misura di accuratezza come sua prova.
