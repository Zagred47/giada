# 🧠 GIADA Task 9 — percorsi fisiologici del teacher completo

La Task 8 ha usato percorsi inventati e imposti. Ora prendiamo le microtracce
`V(t)` già registrate dal teacher Hay autentico a 642 segmenti nel dataset
targeted v1.1. Non rigeneriamo il dataset e non addestriamo nuovi modelli.

🔒 La sorgente è fissata con SHA-256; leggiamo soltanto lo split `train`,
lasciando intatti validation e test. Campioniamo deterministicamente fino a
4.000 transizioni e fino a 24 percorsi per ciascuna combinazione fra quattro
siti (soma, hot zone, due punti tuft) e quattro regimi (quiet, salita, spike,
discesa). Eventuali celle vuote restano visibili: non le riempiamo a posteriori.

⏱️ Dal boundary state autentico `(m_t,h_t)` applichiamo formula, LUT-513 e i
tre seed physical-τ congelati usando (a) solo `V_t`, (b) otto campioni, o
(c) tutti i 40 sottopassi. Seguendo il contratto temporale 7c, il sottopasso
usa `V_(n+1)` della microtraccia. Valutiamo sia rispetto alla formula composta
sul percorso fine, sia rispetto al vero stato `m,h` al bordo successivo.

⚠️ Il teacher usa CVode e le microtracce sono float32: la formula esatta su
40 campioni potrebbe non riprodurre esattamente il gate finale. Misuriamo
quindi questo *reference floor*. Se il massimo RMSE di `m,h` supera `0,005`
in qualsiasi gruppo sito/regime non vuoto, non attribuiamo l'errore
candidato–teacher al candidato; resta valido soltanto il confronto numerico
candidato–formula. Nessun seed viene scelto sui dati.

Questa è ancora una diagnosi **teacher-forced**: il percorso futuro del teacher
non diventa un input causale disponibile al modello autonomo.
