# GIADA — risultato Task 10 originale: sufficienza dell'ingresso

## Provenienza e validità

- Artefatto: `giada_roadmap_task10_input_sufficiency.zip`.
- SHA-256 ZIP: `cb415191a9b2230b0224d2d7c9b20028493490688cca33e091a353dafac45b99`.
- Revisione eseguita: `732e236a5b287c9c61d528228edfef493dbd8dd4`.
- 350 path Task 9 già aperti, solo train; nessun nuovo training, nessun nuovo teacher, nessun sealed test aperto.
- Otto ricostruzioni del voltaggio × 1/2/4/8/40 sottopassi, stessi stati iniziali m/h.
- Report valido; riproduzione full41/40 del midpoint Task 9b: errore massimo 0.

## Risultati

RMSE **adimensionali**, non mV. La tabella misura l'errore aggiuntivo rispetto a full41/40, a 40 sottopassi.

| Vista del voltaggio | RMSE m globale | RMSE m²h globale |
| --- | ---: | ---: |
| Solo V iniziale, mantenuto costante | 0.221102 | 0.145690 |
| Media oracle, mantenuta costante | 0.037271 | 0.030119 |
| Estremi oracle, interpolazione lineare | 0.124358 | 0.099375 |
| Estremi + media oracle, ricostruzione quadratica | 0.012448 | 0.012100 |
| 5 campioni oracle | 0.011245 | 0.010903 |
| 9 campioni oracle | 0.001526 | 0.001642 |
| 21 campioni oracle | 0.000164 | 0.000163 |
| Tutti i 41 campioni oracle | 0 | 0 |

La prima vista campionata testata che soddisfa RMSE <=0.001 per m, h e m²h in **ogni gruppo sito/regime** è `twentyone_oracle`. Nel gruppo più difficile, soma/spike, gli errori m/m²h sono 0.000580/0.000611, contro 0.005560/0.006202 con nove campioni.

Il floor full41/40 contro teacher autentico è m=0.0002145852, h=0.00000122824, m²h=0.0001165483. Non confondere errore aggiuntivo con errore totale contro teacher.

Anche la quadratura conta: con full41 ma 1/2/4/8 sottopassi, RMSE m aggiuntivo è rispettivamente 0.078300/0.024260/0.009726/0.001002. Non sono stati testati tutti i conteggi intermedi.

## Interpretazione e correzione del perimetro

Il risultato dimostra limiti di specifiche ricostruzioni analitiche del path, non un limite universale delle reti o di tutti gli operatori a 1 ms. Non dimostra che 21 misure future oppure 40 sottopassi siano sempre necessari. Gli estremi+media, per esempio, sono stati decodificati con una particolare curva, non con ogni decoder possibile.

La Task 7 originale ha dimostrato con contrasti appaiati che percorsi esogeni diversi con stessi estremi e gate iniziali possono dare gate finali diversi. Questo non equivale a dimostrare che uno stato fisico completo e gli ingressi causali non possano determinare l'evoluzione endogena. Occorre specificare l'interfaccia: canale isolato guidato da un path esterno oppure sistema accoppiato che genera il proprio voltaggio.

Tutte le viste diverse da start_only contengono futuro teacher: sono oracle diagnostici, non ingressi disponibili online. La Task 10 conclude il confronto diagnostico preregistrato, **non chiude il contratto causale operativo del Gate B**, non valida il neurone completo e non dimostra accelerazione GPU.

Le ipotesi successive sono separate in `giada_post_task10_causal_operator_hypotheses.md`; nessun nuovo notebook o protocollo è preparato con questa registrazione.
