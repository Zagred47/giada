# 🧩 GIADA Task 4 — matrice di primitive appaiate

## 🎯 Domanda

Nel dominio `Ca_HVA m+h` held-voltage già validato, quale primitiva offre il
miglior compromesso tra accuratezza, struttura fisica, capacità, stabilità e
costo?

## 🧪 Bracci nello stesso run

| Classe | Braccio | Informazione imposta |
|---|---|---|
| Oracle | formula estratta | cinetica completa |
| Numerico | LUT nearest | valori tabulati |
| Numerico | interpolazione lineare | regolarità locale |
| Numerico | Chebyshev grado 16 | regolarità globale |
| Neurale | MLP diretto | nessuna dinamica esplicita |
| Neurale | GRU | ricorrenza generica |
| Neurale | physical-τ | rate e solver esponenziale |
| Neurale | direct-z | combinazione convessa |

I quattro bracci neurali e i tre seed vengono eseguiti contemporaneamente
sulla GPU. Ricevono gli stessi tuple e gli stessi minibatch. I checkpoint a
0/1k/3k/10k/30k/50k producono anche la prima mini scaling law della Task 5.

## 🔒 Separazione dei dati

Il sealed Task 3e non viene letto. La selezione del checkpoint avviene soltanto
su development. Un nuovo sealed Task 4 viene materializzato una sola volta,
dopo il freeze, e non modifica la selezione.

## 📏 Interpretazione

La formula è l'oracle di accuratezza, non un candidato appreso. Le famiglie
sono confrontate mediante RMSE di `m`, `h` e `m²h`, peggior strato normalizzato,
violazioni di occupazione, parametri e curva rispetto al budget.

Se physical-τ resta entro il 10% del miglior modello appreso conserva la
preferenza meccanicistica. Un vantaggio direct-z limitato a `dt` fissi non è
sufficiente: la decisione definitiva richiederà lo stress su `dt` della Task 6.
