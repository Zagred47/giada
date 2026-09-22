# GIADA Task 3d — risultato della matrice causale m+h

## Esito sintetico

La Task 3d ha ristretto nettamente il problema: il fallimento Task 3c è principalmente un problema di **supporto in tensione**, con un forte beneficio addizionale del prior monotono soltanto nel contesto informativo completo. Non emergono prove che servano un trunk indipendente o una rete più larga.

Il candidato development da confermare è `full_repair`, shared width 23, step 50.000. Tutti e tre i seed hanno score inferiore a 1 (`0,3323`, `0,3120`, `0,6135`). La Task 4 resta bloccata finché questo candidato non supera un nuovo test sealed.

## Integrità

- 30 candidati complessivi;
- 24 shared, 3 independent, 3 wide;
- circa 45,9 minuti complessivi;
- 545 candidate-step/s;
- nessun accesso al fresh Task 3c;
- checkpoint e report verificati tramite SHA-256;
- nessun test sealed usato per selezione.

## Correzione interpretativa necessaria

I training e le metriche dell'artefatto sono validi. Tuttavia alcune percentuali nel campo `causal_effect_fractions` sono state calcolate contro `baseline_current`, invece che contro il controllo locale indicato nel preregistro. I contrasti corretti sono stati ricalcolati dai valori grezzi immutabili:

| Contrasto preregistrato | Effetto corretto | Supportato ≥20% |
|---|---:|---|
| baseline → same-support dense | −11,2% | No |
| dense → balanced | +50,8% | Sì |
| balanced → expanded | +88,6% | Sì |
| balanced → multihorizon | −28,4% | No |
| balanced → shape | −80,6% | No |
| expanded → expanded+multihorizon | −37,7% | No |
| expanded+multihorizon → full repair | +75,5% | Sì |
| baseline → full repair | +97,9% | Sì |
| full shared → independent | +1,4% | No |
| full shared → width 46 | −66,1% | No |

Questo cambia la spiegazione rispetto alla sintesi automatica originale: il multi-orizzonte da solo non è una riparazione e il prior di forma da solo non è una riparazione. Il prior diventa molto efficace quando è combinato con supporto esteso e orizzonti multipli. È quindi presente un'interazione, non una somma di effetti indipendenti.

## Cosa abbiamo escluso

- più campioni nello stesso intervallo non bastano;
- una rete più larga non aiuta al budget registrato;
- il trunk condiviso non è il collo di bottiglia dominante;
- il solo multi-orizzonte non risolve la generalizzazione;
- il solo vincolo monotono non risolve la generalizzazione;
- prolungare ogni braccio indiscriminatamente non è corretto: molti controlli peggiorano dopo 30k.

## Cosa risulta supportato

1. bilanciare il supporto di tensione;
2. estenderlo esplicitamente nelle code;
3. usare il prior monotono nel contesto completo;
4. mantenere la cella shared compatta;
5. congelare il checkpoint preregistrato del candidato full repair prima del nuovo test.

## Decisione

Preparare una conferma minimale del candidato `full_repair shared width 23 @ 50k` su un nuovo sealed set indipendente. Il nuovo set deve contenere strati central, voltage-tail e long-horizon e non deve riutilizzare tuple o risultati del fresh Task 3c.
