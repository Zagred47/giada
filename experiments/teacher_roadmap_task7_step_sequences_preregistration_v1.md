# ⚡ Roadmap Task 7 — sequenze di voltage step esogene

Questa è la **Task 7 originale della roadmap**, non il microcanary closed-loop
storicamente etichettato `07/07b/07c`. La Task 6 ha già coperto step singoli a
0,5 ms e impulso bifasico a 0,25–0,75 ms, con formula e pilot NEURON validati.
Qui completiamo solo il contrasto mancante: ordine, tempo e durata degli step
variabili con identici estremi e identici gate iniziali.

Quattro famiglie appaiate: salita precoce/tardiva, impulso precoce/tardivo,
impulso breve/lungo e sequenza alto-basso/basso-alto. Due profili per coppia
con stesso V iniziale e finale, stessi m/h iniziali. Il path ha otto intervalli
piecewise-constant da 0,125 ms. Per famiglia: 256 development, 512 sealed;
seed distinti. Congeliamo prima del sealed la formula `.mod`, LUT-513 e
physical-τ width-32 con seed 17/29/43 della Task 5. Nessun retraining.

Il pilot verifica 24 percorsi/famiglia contro Ca_HVA compilato (errore massimo
gate ≤1e-9). Su development e sealed misuriamo RMSE di m, h e m²h, score
normalizzato, violazioni d'occupazione e contrasto within-pair del teacher.
Confrontiamo input solo V iniziale con l'intero path **esogeno noto**. Il sealed
si apre dopo freeze con hash e overlap zero. Gate di interpretazione: pilot
valido, valori finiti, split disgiunti, LUT path-aware con score ≤0,02 in tutte
le famiglie; almeno una famiglia deve mostrare differenza teacher entro coppia
≥0,001 per il gate m o h. In caso contrario risultato inconclusivo, non
asserzione che la storia non conti.

Il voltaggio futuro è noto solo perché imposto dal clamp: non è un input
causalmente disponibile al neurone autonomo. Questa task non valida il
closed-loop e non sostituisce la Task 10 sulla sufficienza degli input.
