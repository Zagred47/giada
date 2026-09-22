# 🌊 GIADA Task 8 — rampe e chirp a velocità controllata

La Task 6 aveva già testato una rampa a otto campioni; quindi qui non la
replichiamo. Imponiamo a Ca_HVA quattro famiglie di percorsi continui di 1 ms:
rampa lenta, rampa rapida (variazione concentrata in 0,25 ms), chirp a bassa
frequenza e chirp ad alta frequenza. Ogni percorso è campionato a 0,025 ms;
occupazioni iniziali `m,h` sono indipendenti dall'equilibrio.

🔬 Il target è la formula `.mod` composta sui 40 sottopassi. Un pilot appaiato
con il meccanismo NEURON compilato verifica 12 casi per famiglia. A parità di
percorso confrontiamo LUT-513 e physical-τ width-32, seed 17/29/43, **tutti
congelati** dalla Task 5. Ogni candidato riceve alternativamente soltanto
`V_start`, otto campioni o tutti i 40 campioni. Anche la formula esatta è
valutata con le prime due viste: questo misura l'informazione temporale persa
senza attribuirla erroneamente alla capacità di una rete.

📏 Riportiamo separatamente per famiglia RMSE di `m`, `h`, `m²h`, score
normalizzato, violazioni d'occupazione e dispersione fra seed. Development:
128 casi/famiglia; sealed: 256 nuovi casi/famiglia. Dopo il development si
scrive il freeze; nessuna scelta del candidato o delle soglie usa il sealed.

⚠️ Il voltaggio futuro è noto perché imposto dall'esterno. Il test non fornisce
al modello chiuso un input causale futuro e non dimostra validità su 642
segmenti, calcio dinamico o updater di voltaggio appreso.
