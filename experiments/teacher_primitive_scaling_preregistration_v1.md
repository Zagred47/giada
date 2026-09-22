# 📈 GIADA Task 5 — scaling fisico e numerico

La Task 4 ha mostrato physical-τ come miglior modello appreso e la LUT lineare
come baseline più accurata. La Task 5 misura dove si trova il compromesso.

Un unico run addestra tre larghezze physical-τ (`16`, `23`, `32`) con tre seed
e gli stessi minibatch. Salva gli score ai passi `0/1k/3k/10k/30k/50k/75k`.
In parallelo valuta LUT da `33` a `513` punti e Chebyshev di grado `8/16/24`.

Latenza e throughput di physical-τ e LUT sono misurati sulla stessa GPU a
batch `1024` e `65536`. Il costo di memoria della tabella viene esplicitato.
Le misure riguardano implementazioni PyTorch float32, non un kernel ottimizzato
o un deployment definitivo.

Il sealed della Task 4 resta chiuso. Il checkpoint viene scelto sul development
per ogni larghezza. Anche la larghezza preferita e la risoluzione preferita di
LUT lineare, LUT nearest e Chebyshev sono scelte sul development e congelate
prima di aprire una volta un nuovo sealed Task 5 di
8.192 tuple. Questa fase resta limitata al voltage clamp costante.
