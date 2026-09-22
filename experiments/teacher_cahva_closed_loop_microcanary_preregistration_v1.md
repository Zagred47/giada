# 🔁 GIADA Task 7 — microcanary causale Ca_HVA

Prima di aggiungere una rete per il voltaggio, testiamo se la LUT sopravvive
quando riceve **il voltaggio prodotto dal proprio circuito**, non la traccia
futura del teacher. Un compartimento NEURON autentico contiene `Ca_HVA` e
`pas`; la corrente Ca_HVA rimane calcolata con la formula economica
`gbar*m*m*h*(V-E_Ca)`. Il solver del voltaggio è fisico e deterministico, non
ancora una rete: questo isola il feedback gate–corrente–voltaggio.

Bracci appaiati: teacher NEURON, formula esatta dei gate + solver semi-implicito,
LUT-513 congelata + lo stesso solver, physical-τ congelato nei tre seed +
lo stesso solver. Tutti iniziano dallo stesso stato; dopo il tempo zero ogni
braccio usa solo il proprio V. Dodici traiettorie da 20 ms: due condizioni
iniziali, 1×/4× conduttanza, riposo/impulso/doppio impulso. Il 4× è uno stress
artificiale, non il valore canonico.

La formula esatta + solver misura il **mismatch numerico di base con NEURON**.
Se supera 0,25 mV di RMSE V o 0,005 RMSE gate in un episodio, non attribuiamo
automaticamente lo scarto degli altri bracci alla LUT o alla rete: salviamo
comunque il report e diagnostichiamo il solver. Metriche per episodio:
voltaggio, gate, corrente e violazioni d'occupazione. Nessun retraining,
selezione del seed o uso del futuro teacher.

⚠️ Un compartimento non rappresenta il neurone Hay a 642 segmenti. Non sono
testati calcio dinamico, sinapsi, altri canali, coupling assiale o un updater
del voltaggio appreso. L'architettura ibrida completa resta un'ipotesi.
