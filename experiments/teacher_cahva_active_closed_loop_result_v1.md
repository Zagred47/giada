# 🔁 GIADA Task 7b — risultato e riallineamento necessario

ZIP esatto: SHA-256
`be673bf2bc065f8f6b5c88bbc36713de943dd6283166c57de7b5f1d642e79ae7`.
Tutti i 24 episodi sono completi, il controllo a `gbar=0` è pulito, sei delle
otto coppie canoniche superano il gate di esposizione e il voltaggio della
formula coincide con NEURON a circa `2e-12 mV`. Ca_HVA modifica quindi V in
modo misurabile, fino a oltre 5 mV nel confronto 1×/0×.

⚠️ La successiva ispezione forense delle tracce mostra che la metrica dei gate
del report originale è sfasata di un sottopasso. Il teacher aggiorna V con i
gate di bordo, aggiorna poi i gate usando il nuovo V e registra `ica` del
vecchio stato. La formula originale dava gate a `n+1` identici ai gate teacher
di `n`, non ai gate teacher di `n+1`. L'RMSE m circa 0,0026 non è quindi un
errore di rappresentazione della formula. Con ordine corretto, la formula
riproduce tutti gli episodi in V, m, h e corrente a precisione numerica.

La Task 7c ricalcola i candidati congelati sullo stesso ZIP, senza rieseguire
NEURON e senza trattare questa rianalisi come conferma indipendente. Il
vantaggio di V della LUT nel report 7b resta una misura su un rollout con stato
semanticamente sfasato; non autorizza ancora una scelta architetturale.
