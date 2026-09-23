# ✅ Task 9c — replay NEURON denso

ZIP SHA-256: `32f7472d0a41a24c8f6a6d8666b8feed6dc5f1dc4f8017bfb0602d4dad0e1bd4`.
Il run della revisione `395f61b` è valido su 12 transizioni **train** già
congelate. Il replay della traccia originale a 40 campioni differisce al massimo
di `3,8125e-6 mV`; lo stato finale completo e le sequenze RNG coincidono
esattamente. Nessun blocker di riproducibilità.

L'integrazione della formula Ca_HVA sul voltaggio autentico a `0,001 ms`
produce RMSE globale di `m = 0,000134906` contro `0,000151628` usando il
midpoint sui 40 campioni e `0,000145998` con interpolazione lineare dei 40
campioni. La differenza massima tra risultati a `0,005` e `0,001 ms` è solo
`1,998e-6` per i gate. Tutti gli otto gruppi sito/regime passano il limite
registrato di `0,005`.

Il miglioramento addizionale rispetto all'interpolazione della Task 9b è
**modesto**, mentre il grande miglioramento precedente derivava soprattutto
dal non trattare ogni intervallo come se il voltaggio fosse costante al suo
estremo destro. Resta un piccolo floor formula/teacher, dell'ordine di
`1e-4` per `m` nel campione selezionato: non dichiararlo nullo né attribuirlo
alla rete. Il risultato riguarda 12 casi train già aperti, non un test
indipendente e non un rollout di voltaggio autonomo.
