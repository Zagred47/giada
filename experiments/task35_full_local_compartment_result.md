# GIADA Task 35 — compartimento locale completo dichiarato

**Esito:** confermato nei 32 casi indipendenti preregistrati. Questa è una
conferma di generalizzazione *locale* a variazioni controllate delle
conduttanze, non una simulazione multicompartmentale o della cellula intera.

Il run Kaggle ha usato GIADA `e1a6efd78bbed9e28a160c75e33f7371cdf72d9a`
e il teacher canonico `074c4666300a8ad246601dab179a97a6942f0f29`.
Protocollo, codice e notebook provenivano dalla stessa revisione. La
calibrazione separata è passata prima di aprire i casi di conferma. Nessun
checkpoint è stato riaddestrato o selezionato sul nuovo test.

## Contrasto e misure

La Task 33 aveva confermato 11 canali, CaDynamics, membrana passiva e quattro
sinapsi stocastiche in un cilindro attivo *solo* con conduttanze canoniche.
Qui lo stesso inventario locale viene riprodotto con seed e tempi di eventi
nuovi, in quattro pannelli: canonico, calcio×4, sodio×4 e potassio×4. Due
seed, due tensioni iniziali e due protocolli per pannello producono 32 casi
di 40 ms a passo 0,1 ms. Eventi presinaptici e stream Random123 sono
osservati; i rilasci realizzati e le correnti future del teacher non entrano
nel modello.

| Controllo | Risultato |
|---|---:|
| Floor formula–NEURON, peggior V tra i pannelli | 0,002833 mV |
| Floor formula–NEURON, peggior Ca | 8,00×10⁻⁸ mM |
| Peggior modello/pannello, V pooled | 0,010606 mV |
| Peggior modello/pannello, V sul singolo episodio | 0,025443 mV |
| Peggior modello/pannello, Ca | 2,764×10⁻⁶ mM |
| Effetto sinaptico del controllo zero-weight | ≥21,3 mV nei pannelli |

Il caso più difficile per V e Ca è `independent`, seed 43, pannello
`calcium_x4`. Anche questo supera gli stessi limiti numerici preregistrati
in Task 33 (V pooled ≤1 mV, V episodio ≤2 mV, Ca ≤10⁻⁵ mM). Tutti e sei i
checkpoint passano **in ciascuno** dei quattro pannelli; non si è usata una
media aggregata per nascondere fallimenti. Shadow sinaptico e floor nativo
passano; i controlli zero-weight/RNG, eventi programmati ≠ rilasci e
presenza dell'effetto sinaptico passano.

## Interpretazione e limite

Il risultato rende più solida la componente locale: i gate appresi, il solver
analitico V/Ca e l'interfaccia sinaptica causale continuano a funzionare
insieme anche quando i gruppi principali di conduttanze sono amplificati
separatamente. Le moltiplicazioni ×4 sono pannelli di stress preregistrati,
non una dichiarazione che rappresentino distribuzioni fisiologiche complete.
La Task 34 aveva mostrato che la maggior parte dello scarto grezzo delle
correnti rispetto allo stato nativo non era attribuibile direttamente ai
gate neurali; la Task 35 non risolve né reinterpreta quel limite diagnostico.

Il compartimento resta un singolo cilindro con `E_Ca` fissato. Non sono
testati corrente assiale, geometria/morfologia, eterogeneità spaziale,
predizione della cellula intera o speedup. La successiva Task 36 deve
introdurre e validare l'interfaccia assiale tra due compartimenti, con il
prerequisito IV-D1; non è autorizzata a ereditare un claim di cellula completa.

Artefatti: `experiments/results/task35_final_report.json`,
`experiments/results/task35_premodel_report.json`,
`experiments/results/task35_calibration_report.json` e
`experiments/results/task35_native_shadow_traces.npz`.
