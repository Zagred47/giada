# 🧭 GIADA Task 7c — riallineamento dei bordi temporali

È una **rianalisi forense post hoc** dei dati 7b già aperti, non un nuovo test
indipendente. Nessun teacher NEURON viene rieseguito e nessun modello viene
riaddestrato. Le due componenti dello ZIP 7b sono fissate con SHA-256 nel
contratto JSON accanto a questo documento.

L'evidenza diretta delle tracce è che NEURON usa i gate al bordo `n` per
aggiornare il voltaggio, registra `ica` dal medesimo stato vecchio e poi
aggiorna i gate con il voltaggio `V_(n+1)`:

```text
V_(n+1)       = G(V_n, m_n, h_n, U_n)
ica_(n+1)     = gbar * m_n² * h_n * (V_n - E_Ca)
(m,h)_(n+1)   = F(V_(n+1), m_n, h_n, dt)
```

🔬 La formula esatta deve riprodurre simultaneamente voltaggio, gate e corrente
campionata sul teacher salvato. Solo dopo sono comparabili le metriche di
boundary state di LUT e physical-τ congelati. Vengono riportati tutti i 24
episodi e tutti i tre seed; non selezioniamo un vincitore usando questi dati.

⚠️ Le tracce 7b non sono un fresh test. Anche un ottimo risultato qui non
dimostra ancora transizioni da snapshot arbitrari, teacher Hay completo,
calcio dinamico o un updater di voltaggio appreso.
