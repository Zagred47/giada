# ⚡ GIADA Task 6 — stress del gate su percorsi di voltaggio noti

Congeliamo i vincitori della Task 5: LUT lineare a 513 punti e physical-τ
di larghezza 32, tutti e tre i seed. Nessun riaddestramento.

In un solo run confrontiamo sei famiglie di percorsi da 1 ms: step in salita e
discesa, rampa, impulso bifasico, intorno a -27 mV e estremi di tensione.
Ogni caso inizia con occupazioni `m,h` indipendenti dall'equilibrio. Il teacher
di riferimento è la composizione della formula `.mod` a 8 sottopassi; un pilot
la confronta con il meccanismo NEURON compilato autentico.

Le due domande ortogonali sono: (1) quanta informazione si perde dando soltanto
il voltaggio iniziale `V_t`? (2) a informazione uguale, LUT o physical-τ
mantiene meglio la precisione? Perciò ciascuno è valutato sia con `V_t`
costante per l'intero ms, sia con il percorso esogeno noto. L'oracle formula
con `V_t` costante misura il limite informativo di quel contratto.

La fase development comprende 256 casi per famiglia. Dopo il freeze vengono
generati 512 casi sigillati indipendenti per famiglia. Nessun risultato
sigillato cambia candidato o soglia. Metriche separate per `m`, `h`, `m²h`,
violazioni d'occupazione e dispersione fra seed.

⚠️ Il percorso futuro è noto perché imposto dal voltage clamp. Non equivale
a conoscere il futuro voltaggio *endogeno* del neurone. Il test di feedback
membrana–gate resta una fase successiva.
