"""Record the complete Task17d-b Kaggle result in the SQLite mirror."""

from __future__ import annotations

import hashlib
import json

from .mirror import Mirror, ROOT, write_json


def main():
    path = ROOT.parent / "experiments/results/task17db_solver_release_8bdfd0d/README.md"
    payload = path.read_bytes()
    mirror = Mirror()

    def put(table, suffix, **fields):
        return mirror.local_upsert(table, {
            "Codice stabile": f"{table}-giada-task17db-result-{suffix}-v1",
            **fields,
        })["record_id"]

    ids = {row["table_key"]: row["record_id"] for row in mirror.query(
        "SELECT table_key,record_id FROM v_records WHERE stable_code IN ("
        "'blocks-giada-task17db-solver-release-paired-v1',"
        "'experiments-giada-task17db-solver-release-matrix-v1')"
    )["rows"]}
    if set(ids) != {"blocks", "experiments"}:
        raise RuntimeError("Task17d-b preregistration missing")
    artifact = put("artifacts", "analysis", **{
        "Nome": "Task17d-b — analisi del run Kaggle completo",
        "Tipo": "Report",
        "Percorso": str(path.relative_to(ROOT.parent)).replace("\\", "/"),
        "SHA-256": hashlib.sha256(payload).hexdigest(),
        "Dimensione byte": len(payload), "Versione": "v1",
        "Descrizione": "ZIP originale SHA-256 bad9cb4603906a703cee38d406b69caa148b2e3daa188fc7afc56df3f31c2b522; metriche e release_decisions nell'archivio utente.",
    })
    run = put("runs", "kaggle-complete", **{
        "Nome": "Task17d-b 8bdfd0d — rilasci, solver e candidati",
        "Stato": "Completata", "Blocco": [ids["blocks"]],
        "Artefatti prodotti": [artifact], "Seed": "170029,170083",
        "Hardware e ambiente": "Kaggle, NEURON 8.2.7+, teacher 642 segmenti, revisione 8bdfd0d1453e0b7169de9cfc25804ecd3ebfec16",
        "Descrizione": "21 episodi sui tre casi noti; nove coppie native e nove coppie candidate; 420 outcome evento in totale, 20 per episodio.",
        "Configurazione effettiva": "60 ms; 0,025 ms; default, voltage_scaled, ultra, super_ultra; formula/LUT513/m-LUT2049 a ultra.",
        "Validità tecnica": "Exit 0, valid=true, verifica della frontiera valida; decisioni/draw/probabilità/quantità di rilascio identici in tutti i confronti; riferimento e controllo formula passano soglie diagnostiche.",
    })
    for suffix in ("discrete-release", "reference-stability", "matched-candidates"):
        mirror.local_upsert("claims", {
            "Codice stabile": f"claims-giada-task17db-solver-release-{suffix}-v1",
            "Stato": "Supportata nel dominio",
        })
    for suffix in ("calcium-scale", "driver-and-release"):
        mirror.local_upsert("claims", {
            "Codice stabile": f"claims-giada-task17d-native-solver-{suffix}-v1",
            "Stato": "Contraddetta nel dominio",
        })
    observations = []
    for seed, gbar, default, voltage, super_ultra, lut513, lut2049 in (
        (170029, "1.0", 43.226966, 0.119684, 0.006729, 0.177680, 0.010962),
        (170083, "1.5", 5.638085, 0.004086, 0.003001, 0.133836, 0.008718),
        (170029, "0.5", 0.042933, 0.001267, 0.000097, 0.000745, 0.000046),
    ):
        for arm, value in (("default", default), ("voltage_scaled", voltage),
                           ("super_ultra", super_ultra), ("lut513", lut513),
                           ("m_lut2049", lut2049)):
            observations.append(put("observations", f"{seed}-{gbar}-{arm}", **{
                "Nome": f"Task17d-b massimo RMSE V {seed}/{gbar}/{arm}",
                "Valore": value, "Numerosità": 1,
                "Strato o sottogruppo": f"seed={seed}; gbar={gbar}; arm={arm}; max 4 sites",
                "Descrizione": "mV; confronto col nativo ultra; numeri arrotondati dal report Kaggle.",
                "Run": [run], "Artefatti dettagliati": [artifact],
            }))
    put("findings", "attribution", **{
        "Nome": "Task17d-b: rilasci identici, riferimento stabile e m-LUT più precisa",
        "Risultato": "Tutti i 18 confronti hanno 20 eventi appaiati e decisioni/draw/quantità identici. Ultra-super_ultra max RMSE V 0,006729 mV. Clone formula 0; LUT513 max gate 0,011229 > 0,01; m-LUT2049 max gate 0,000731 e max RMSE V 0,010962 mV.",
        "Descrizione": "Sul set di sviluppo, la sensibilità del teacher a v non deriva da release flip; la maggiore risoluzione dei rate m riduce l'errore sotto solver uguale.",
        "Incertezza": "Tre casi già usati, nessuna conferma indipendente o misura di costo computazionale.",
        "Limitazioni": "LUT2049 è nuova sonda diagnostica, non candidato congelato. Gate C resta chiuso; tabella 513 ricostruita dalla formula.",
        "Esito": "Misto", "Osservazioni": observations, "Esperimenti": [ids["experiments"]],
    })
    write_json(ROOT / "data/airtable_snapshot.json", mirror.export_snapshot())
    assert mirror.verify()["valid"]
    return {"valid": True, "airtable_accessed": False, "run": run,
            "observations": len(observations)}


if __name__ == "__main__":
    print(json.dumps(main(), indent=2))
