"""Record the exact Task 17b Kaggle replica in the SQLite mirror only."""

from __future__ import annotations

import hashlib
import json

from .mirror import Mirror, ROOT, write_json


EXPECTED = {
    "final_report.json": "6af0eb906544478cae6d6288120fef818cef44bfc224218105c6a5085a586a80",
    "paired_metrics.json": "e8d5b960ec9cdaebad52ecd8bd047af46efaf459676887a575f79300a6d5cd62",
    "process_status.json": "1b803d603af17696e80f74599d4afaa3859e3a86720a90327ae9145a80ed609c",
}


def main():
    base = ROOT.parent / "experiments/results/task17b_kaggle_replication"
    original = ROOT.parent / "experiments/results/task17b_rate_attribution"
    for filename, digest in EXPECTED.items():
        if hashlib.sha256((base / filename).read_bytes()).hexdigest() != digest:
            raise RuntimeError(f"Kaggle Task17b evidence hash mismatch: {filename}")
    report = json.loads((base / "final_report.json").read_text(encoding="utf-8"))
    metrics = json.loads((base / "paired_metrics.json").read_text(encoding="utf-8"))
    local = json.loads((original / "paired_metrics.json").read_text(encoding="utf-8"))
    status = json.loads((base / "process_status.json").read_text(encoding="utf-8"))
    assert status["returncode"] == 0 and report["valid"]
    assert report["code_revision"] == "4ee6b574ac0daad520790888b5693c8efa146ec1"
    assert report["decision"] == "DIAGNOSTIC_ONLY" and not report["gate_c_authorized"]
    assert report["teacher_segment_count"] == 642 and len(metrics) == len(local) == 9
    assert max(x["voltage_max_error_mv"] for x in report["native_repeat_preflight"].values()) == 0
    numerical_differences = []
    for actual, reference in zip(metrics, local):
        assert (actual["seed"], actual["gbar_multiplier"]) == (reference["seed"], reference["gbar_multiplier"])
        for arm, sites in actual["metrics"].items():
            for site, row in sites.items():
                for key, value in row.items():
                    numerical_differences.append((key, abs(value - reference["metrics"][arm][site][key])))
    maximum_difference = max(value for _, value in numerical_differences)
    assert maximum_difference < 1e-5
    failures = [(row["seed"], row["gbar_multiplier"], site)
                for row in metrics for site, value in row["metrics"]["lut"].items()
                if max(value["m_max_error"], value["h_max_error"]) > 0.01]
    assert failures == [(170029, 1.0, "460"), (170083, 1.5, "0")]

    mirror = Mirror()

    def put(table, suffix, **fields):
        return mirror.local_upsert(table, {
            "Codice stabile": f"{table}-giada-task17b-kaggle-replication-{suffix}-v1",
            **fields,
        })["record_id"]

    artifact_ids = []
    for filename in EXPECTED:
        path = base / filename
        artifact_ids.append(put("artifacts", filename.split(".")[0], **{
            "Nome": f"Task17b — Kaggle {filename}", "Tipo": "Report",
            "Percorso": str(path.relative_to(ROOT.parent)).replace("\\", "/"),
            "SHA-256": EXPECTED[filename], "Dimensione byte": path.stat().st_size,
            "Versione": "2026-09-30-kaggle-v1",
            "Descrizione": "Estratto esatto dallo ZIP utente giada_task17b_rate_attribution_4ee6b57_5e2fac30.zip, SHA-256 ZIP d509f51b507a72c44bf00ef21fce759f0c4cc7abe90bd26c5a1bcded6dc3ef77.",
        }))
    run = put("runs", "same-seed-technical", **{
        "Nome": "Task17b — replica tecnica Kaggle del commit 4ee6b57",
        "Stato": "Completata", "Artefatti prodotti": artifact_ids,
        "Seed": "170029,170083,170097",
        "Descrizione": "Stessa matrice e stessi seed dell'esecuzione locale: verifica di portabilità tecnica, non nuova evidenza indipendente sul tasso di fallimento.",
        "Hardware e ambiente": "Kaggle; NEURON 8.2.7 dichiarato dal notebook; teacher 074c4666300a8ad246601dab179a97a6942f0f29; codice GIADA 4ee6b57; process_status exit 0.",
        "Configurazione effettiva": "8 bracci × 3 seed × 3 gbar, protocollo NMDA, 40 ms, campionamento 0,025 ms.",
        "Validità tecnica": f"valid=true; replay nativo e clone formula zero; due casi gate >0,01 invariati; massimo delta metrica locale–Kaggle {maximum_difference:.8g}.",
    })
    put("findings", "reproduced-diagnosis", **{
        "Nome": "Task17b: la diagnosi dei rate m si riproduce tecnicamente su Kaggle",
        "Risultato": "Run Kaggle exit 0 sul commit 4ee6b57, 642 segmenti; stesso pattern di due superamenti della soglia gate e m-pair quasi sovrapposto alla LUT completa. Massimo delta numerico di una metrica contro il run locale <1e-5.",
        "Descrizione": "Conferma tecnica cross-environment, non campione di seed indipendente e non promozione del candidato.",
        "Incertezza": "Stessi seed, protocollo e parametri del run locale; differenze di floating point minute. Non stima probabilità di generalizzazione.",
        "Limitazioni": "Nessun nuovo test sealed; Gate C resta false; solo NMDA e quattro siti.",
        "Esito": "Misto", "Esperimenti": ["rec3a71e7dd882fd5"],
    })
    write_json(ROOT / "data/airtable_snapshot.json", mirror.export_snapshot())
    assert mirror.verify()["valid"]
    return {"valid": True, "airtable_accessed": False, "run": run,
            "maximum_metric_difference": maximum_difference, "threshold_failures": failures}


if __name__ == "__main__":
    print(json.dumps(main(), indent=2))
