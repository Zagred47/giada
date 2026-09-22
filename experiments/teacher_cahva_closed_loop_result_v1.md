# 🔁 GIADA Task 7 — risultato e limite informativo

Lo ZIP `giada_cahva_closed_loop_microcanary.zip` (SHA-256
`90e7521de52ac9e7af28130701ef2bafde154097add76d1e71c5af715ef2d2d5`)
contiene 12 episodi completi; il report è valido e il solver formula+membrana
riproduce NEURON con worst RMSE V `9,80e-13 mV` e worst RMSE gate
`5,74e-6`.

⚠️ I protocolli scelti non hanno però esposto il canale: `max |ica|` nel teacher
è circa `7,65e-10 mA/cm²`, e il confronto 1×/4× cambia V al massimo di
`1,16e-6 mV`. LUT e physical-τ producono gate diversi ma identiche tracce V
salvate nel rollout float32. Quindi l'equivalenza del voltaggio non discrimina
la qualità dei modelli in una regione Ca_HVA attiva.

✅ Conclusione consentita: infrastruttura di confronto causale e solver di
riferimento calibrati in un compartimento. Non è un successo dell'architettura
nel circuito attivo. La Task 7b preregistrata aggiunge forcing depolarizzante e
controllo a conduttanza zero prima di formulare tale confronto.
