# 🧩 GIADA Task 4 — risultato della matrice appaiata

## Decisione

**Physical-τ è nettamente il miglior modello appreso, ma la LUT con
interpolazione lineare è la migliore primitiva non-oracle per accuratezza.**

La matrice è valida: selezione development-only, sealed Task 3e mai letto,
8.192 esempi sealed nuovi, nessuna sovrapposizione e parallelismo
famiglia×seed effettivamente usato.

## Ranking sealed

| Posizione | Primitiva | Score |
|---:|---|---:|
| 1 | Formula oracle | ~0 |
| 2 | LUT lineare | 0,02963 |
| 3 | Physical-τ | 0,40410 |
| 4 | LUT nearest | 1,11629 |
| 5 | Chebyshev-16 | 1,38298 |
| 6 | Direct-z | 1,70351 |
| 7 | GRU | 4,18906 |
| 8 | MLP diretto | 5,84207 |

Physical-τ supera direct-z di circa 4,2×, GRU di 10,4× e MLP di 14,5×. Ha
solo 694 parametri per seed e zero violazioni di occupazione. Questo è un forte
risultato a favore della decomposizione meccanicistica: una rete generica con
più parametri non recupera automaticamente la dinamica.

## La sorpresa importante: interpolazione lineare

La LUT lineare ottiene uno score circa **13,6× inferiore** a physical-τ. Non
possiamo però dichiararla globalmente dominante: è stata misurata su CPU,
mentre le reti sono state misurate su GPU. A batch 65.536 physical-τ elabora
circa 125,4 milioni di righe/s come ensemble di tre seed; la LUT lineare circa
8,24 milioni di righe/s su CPU. Sono dispositivi diversi, quindi il rapporto
non è una comparazione hardware equa.

## Mini scaling law

Physical-τ passa da score 258,5 all'inizializzazione a 22,87/17,34/3,92/1,41
e infine 0,404 ai checkpoint 1k/3k/10k/30k/50k. Non è ancora chiaramente
saturato. Direct-z raggiunge il minimo a 30k e poi peggiora; MLP e GRU
continuano a migliorare ma rimangono molto lontani.

## Informazione acquisita

1. La struttura physical-τ non è solo interpretabile: è causalmente utile
   rispetto a reti generiche e direct-z.
2. La ricorrenza generica della GRU non sostituisce il solver fisico noto.
3. Nel problema atomico 1D una baseline numerica ben scelta è estremamente
   forte e deve restare nel progetto come controllo o possibile componente.
4. Chebyshev grado 16 non è automaticamente competitivo: l'approssimazione
   globale degrada soprattutto il ramo `m`.
5. La Task 5 è solo parzialmente coperta: abbiamo la curva di budget, non
   ancora capacità, risoluzione LUT e grado polinomiale.

## Prossimo passo

Una Task 5 multifattoriale piccola deve incrociare:

- capacità e budget physical-τ;
- numero di punti della LUT lineare;
- grado e trasformazione del polinomio;
- memoria, throughput sullo stesso dispositivo e precisione numerica.

Subito dopo, la Task 6 deve stressare `dt`, code di tensione, stati iniziali e
rollout lunghi. Solo quello stabilirà se il vantaggio della LUT rimane quando
si esce dal contratto interpolativo più favorevole.
