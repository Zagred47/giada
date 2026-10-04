# Task17f — verifica di sviluppo prima della consegna Kaggle

Eseguita il 4 ottobre 2026 in Linux locale (WSL Ubuntu), NEURON8.2.7.
Il teacher è il commit canonico `074c4666300a8ad246601dab179a97a6942f0f29`.
L'implementazione era nel working tree sopra `8a56e60`: il campo code_revision
dei report indica quindi la base Git, non un commit che già includesse 17f.
Questa directory conserva i report numerici; i risultati non sono una conferma
indipendente e non autorizzano Task18.

## Risultato osservato

Pilot completato in 27 episodi. Primo livello qualificato in ciascuna ladder:

| Evento | Stimolo | Seed pilot positivi |
|---|---|---|
| somatic_spike | 2 impulsi da 0,9 ms a 4 nA | 3/3 |
| calcium_spike | 12 sinapsi, 3 burst, 4 eventi/sinapsi/burst, senza assistenza soma | 3/3 |
| nmda_spike | 10 sinapsi cluster tuft, 2 burst, 2 eventi/sinapsi/burst | 3/3 |

I livelli precedenti sono salvati anche quando negativi o misti. Non sono
stati cambiati pesi NetCon, soglie evento o il candidato in seguito ai risultati.

Il confronto dei detector sulla stessa traiettoria identifica anche un caso
concreto: `nmda_spike_r1_soma0`, seed171519, contiene uno spike NMDA al centro
del cluster ma zero al probe legacy nexus. Non significa che tutti gli eventi
persi in17e siano spiegati dal probe: qui è un contrasto locale di sviluppo.

Integrazione: quiete e tre stimoli sul solo seed pilot171501, gbar1,
quattro bracci nativo/precisione maggiore/formula/LUT, più una replica attiva.
16 episodi nella matrice e 12 confronti; tutti passati, così come la replica.
Peggior RMSE V della LUT nei quattro siti Ca_HVA: `0.000874028118` mV.
`candidate_smoke_passed=true`, `confirmation_accessed=false`,
`gate_c_authorized=false`, `task18_authorized=false`.

La prova esercita concretamente NEURON, restore degli snapshot, corrente
somatica, eventi ripetuti, detector localizzato, compilazione NMODL,
sostituzione causale, rilascio e serializzazione. Le condizioni indipendenti
su tre seed e tre gbar restano da eseguire con il notebook 17f.

Test automatici: 12 passati tra contratti eventi/supporto, rilasci e solver;
JSON del notebook e sintassi di tutte le celle verificati. Questa validazione
non dimostra ancora che tutti i casi confermativi passeranno.
