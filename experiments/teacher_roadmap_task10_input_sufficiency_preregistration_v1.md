# 🧭 Roadmap Task 10 — sufficienza dell'ingresso Ca-HVA

Questa è la **Task 10 originale della roadmap**, non lo studio supplementare
`TG-01` sui sottopassi di un MLP full-state. Riusiamo esclusivamente i 350
percorsi train già congelati dalla Task 9 sul teacher a 642 segmenti. Non
generiamo nuovi dati, non alleniamo modelli e non apriamo lo split test.

## Matrice appaiata

Per ogni percorso e per gli stessi m/h iniziali, componiamo la formula Ca-HVA
con 1, 2, 4, 8 e 40 aggiornamenti interni per ms. A parità di integratore
confrontiamo otto viste del voltaggio: solo V iniziale (causale), media
intra-ms (oracle), estremi (oracle), estremi più media (oracle), 5 campioni,
9 campioni, 21 campioni e tutti i 41 campioni originali (oracle). Le viste
campionate sono interpolate linearmente prima dell'integrazione. Il braccio
41 campioni × 40 sottopassi deve riprodurre entro `1e-12` il midpoint già
definito nella Task 9b. Le viste registrate sono `start_only`, `mean_oracle`,
`endpoints_oracle`, `endpoints_mean_oracle`, `five_oracle`, `nine_oracle`,
`twentyone_oracle`, `full41_oracle`.

Per ogni sito 0/387/460/469 e regime quiet/rising/spike/falling riportiamo
RMSE di m, h, m²h contro il full-path e contro il teacher autentico; la
differenza full-path–teacher è il floor del riferimento e non va attribuita
alla vista ridotta. Una vista campionata supera il gate esplorativo se, con
40 sottopassi, gli RMSE addizionali di m, h e m²h rispetto al full-path sono
tutti ≤`0.001` in ogni gruppo. Il minimo è scelto fra V iniziale, estremi,
5, 9, 21 e 41 campioni; le statistiche sono controlli separati, non una
scala di informazione totalmente ordinata.

⚠️ Solo `start_only` è disponibile causalmente all'inizio dell'intervallo.
Tutte le altre viste usano informazione futura del teacher e sono **oracle
diagnostici**, non input leciti al rollout autonomo. L'esito può stabilire
quale storia di V *sarebbe* sufficiente per il gate isolato, non come produrla
nel neurone accoppiato. I 350 percorsi sono già aperti: nessuna pretesa di
conferma fresh o generalizzazione indipendente.
