"""Register the complete Kaggle Task17d run, preserving attribution limits."""

from __future__ import annotations

import hashlib
import json

from .mirror import Mirror, ROOT, write_json


def main():
    path = ROOT.parent / "experiments/results/task17d_native_solver_5818cb3/README.md"
    payload = path.read_bytes()
    mirror = Mirror()

    def put(table, suffix, **fields):
        return mirror.local_upsert(table, {
            "Codice stabile": f"{table}-giada-task17d-result-{suffix}-v1",
            **fields,
        })["record_id"]

    matches = mirror.query(
        "SELECT table_key,record_id,stable_code FROM v_records "
        "WHERE stable_code IN ("
        "'blocks-giada-task17d-native-solver-paired-v1',"
        "'experiments-giada-task17d-native-solver-matrix-v1')"
    )["rows"]
    ids = {row["table_key"]: row["record_id"] for row in matches}
    if set(ids) != {"blocks", "experiments"}:
        raise RuntimeError("Task17d preregistration missing")
    artifact = put("artifacts", "analysis", **{
        "Nome": "Task17d — analisi del run Kaggle completo",
        "Tipo": "Report",
        "Percorso": str(path.relative_to(ROOT.parent)).replace("\\", "/"),
        "SHA-256": hashlib.sha256(payload).hexdigest(),
        "Dimensione byte": len(payload), "Versione": "v1",
        "Descrizione": "Archivio originale SHA-256 c5a31059964c5f355d0230906fa159983ad1cfda6e52f52ab064c80df38f128e; report, metriche, tracciati e hash in ZIP utente.",
    })
    run = put("runs", "kaggle-complete", **{
        "Nome": "Task17d 5818cb3 — matrice solver nativo completa",
        "Stato": "Completata", "Blocco": [ids["blocks"]],
        "Artefatti prodotti": [artifact], "Seed": "170029,170083",
        "Hardware e ambiente": "Kaggle; teacher NEURON a 642 segmenti; revisione 5818cb3287ac44d31e313deef92a03d1e72167b7",
        "Descrizione": "Tre condizioni note, cinque policy solver e due controlli ultra_dense; 17 confronti appaiati dichiarati nel report.",
        "Configurazione effettiva": "60 ms; campioni 0,025 ms, ultra_dense 0,005 ms; default 1e-3/0, tight 1e-5/1e-6, ultra 1e-7/1e-8; scale cai 1e-4 e v 1e-2.",
        "Validità tecnica": "Exit 0, valid=true, repeat preflight identico. Gli hash completi dei release outcome differiscono: il solo hash non distingue decisioni discrete da stati continui; non rivendicare input realizzati perfettamente appaiati.",
    })
    observations = []
    for seed, gbar, default, calcium, voltage in (
        (170029, "1.0", 43.2270, 43.2327, 0.1197),
        (170083, "1.5", 5.6381, 5.6372, 0.0041),
        (170029, "0.5", 0.0429, 0.0429, 0.0013),
    ):
        for policy, value in (("default", default), ("calcium_scaled", calcium), ("voltage_scaled", voltage)):
            observations.append(put("observations", f"{seed}-{gbar}-{policy}", **{
                "Nome": f"Task17d max RMSE V {seed}/{gbar}/{policy} vs ultra",
                "Valore": value, "Numerosità": 1,
                "Strato o sottogruppo": f"seed={seed}; gbar={gbar}; policy={policy}; max 4 sites",
                "Descrizione": "mV, valori arrotondati dal confronto preregistrato; rilascio discreto fra policy non ancora verificato.",
                "Run": [run], "Artefatti dettagliati": [artifact],
            }))
    put("findings", "voltage-dominance-limited", **{
        "Nome": "Task17d: scala V domina scala cai nei casi nativi noti",
        "Risultato": "Max RMSE V vs ultra: caso 170029/1.0 default 43.2270, calcium_scaled 43.2327, voltage_scaled 0.1197 mV; caso 170083/1.5: 5.6381, 5.6372, 0.0041 mV. Ultra_dense coincide con ultra ai numeri riportati.",
        "Descrizione": "Contrasto descrittivo a policy appaiate sui casi di sviluppo; non isola meccanisticamente ogni termine biofisico e non autorizza Gate C.",
        "Incertezza": "Solo tre condizioni già note; fingerprint dei release outcome comprende campi continui, quindi release_success appaiato non verificabile dall'archivio.",
        "Limitazioni": "Nessun nuovo seed sigillato, nessuna prova generale di convergenza biologica o sufficienza del passo di 1 ms.",
        "Esito": "Misto", "Osservazioni": observations, "Esperimenti": [ids["experiments"]],
    })
    write_json(ROOT / "data/airtable_snapshot.json", mirror.export_snapshot())
    assert mirror.verify()["valid"]
    return {"valid": True, "airtable_accessed": False, "run": run, "observations": len(observations)}


if __name__ == "__main__":
    print(json.dumps(main(), indent=2))
