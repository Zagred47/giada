"""Record Task 17e Kaggle outcome in the SQLite mirror only."""

from __future__ import annotations

import hashlib
import json

from .mirror import Mirror, ROOT, write_json


def main():
    path = ROOT.parent / "experiments/results/task17e_independent_causal_confirmation_4800545/README.md"
    mirror = Mirror()

    def put(table, suffix, **fields):
        return mirror.local_upsert(table, {
            "Codice stabile": f"{table}-giada-task17e-result-{suffix}-v1",
            **fields,
        })["record_id"]

    ids = {row["table_key"]: row["record_id"] for row in mirror.query(
        "SELECT table_key,record_id FROM v_records WHERE stable_code IN ("
        "'blocks-giada-task17e-independent-causal-independent-v1',"
        "'experiments-giada-task17e-independent-causal-matrix-v1')"
    )["rows"]}
    if set(ids) != {"blocks", "experiments"}:
        raise RuntimeError("Task 17e preregistration missing")
    payload = path.read_bytes()
    artifact = put("artifacts", "analysis", **{
        "Nome": "Task17e — analisi della conferma indipendente",
        "Tipo": "Report", "Percorso": str(path.relative_to(ROOT.parent)).replace("\\", "/"),
        "SHA-256": hashlib.sha256(payload).hexdigest(),
        "Dimensione byte": len(payload), "Versione": "v1",
        "Descrizione": "ZIP originale SHA-256 ab29bebd85da9434e3b387b8aba3d4308c455dc5ee2082273d66ec00f93f57f7; metriche ed estratti diagnostici a 1 ms nell'archivio utente.",
    })
    run = put("runs", "kaggle-complete", **{
        "Nome": "Task17e 4800545 — conferma causale indipendente",
        "Stato": "Completata", "Blocco": [ids["blocks"]],
        "Artefatti prodotti": [artifact], "Seed": "171001,171019,171043",
        "Hardware e ambiente": "Kaggle, NEURON 8.2.7+, teacher 642 segmenti, revisione 480054532290c96800cb4312228335dd295bb6e8",
        "Descrizione": "135 episodi; processo exit 0; valid=true; Gate C no-go unicamente per mancanza di eventi nativi richiesti.",
        "Configurazione effettiva": "Cinque schedule, tre gbar, tre bracci, 60 ms, campionamento 0,025 ms; LUT m 2049 f64 congelata.",
        "Validità tecnica": "Hash tabella, compilazione, probe e replay nativo validi; formula, V, gate, corrente, rilascio, contrasto gbar passano; supporto eventi assente.",
    })
    for suffix, name, value, unit in (
        ("voltage", "Peggior RMSE V", 0.00200289621519927, "mV"),
        ("gate", "Peggior errore gate", 0.00011565913381789139, "frazione"),
        ("current", "Peggior RMSE corrente Ca_HVA", 3.5776294744503435e-06, "mA/cm2"),
        ("gbar-effect", "Peggior errore relativo gbar identificabile", 8.677551190909955e-05, "frazione"),
        ("event-support", "Eventi nativi richiesti rilevati", 0, "eventi"),
    ):
        put("observations", suffix, **{
            "Nome": f"Task17e: {name}", "Valore": value, "Numerosità": 1,
            "Strato o sottogruppo": "candidato vs nativo, 45 condizioni; " + unit,
            "Descrizione": f"Unità {unit}; soglie e interpretazione nel report.",
            "Run": [run], "Artefatti dettagliati": [artifact],
        })
    put("findings", "support-limited", **{
        "Nome": "Task17e: accuratezza sub-soglia confermata, Gate C non informativo sugli eventi",
        "Risultato": "Tutti i contrasti numerici passano, ma 0 eventi nativi per ciascuna delle tre classi richieste; event_support=false e Gate C no-go.",
        "Descrizione": "La LUT2049 non mostra una regressione sui casi visitati. Il mancato supporto rigenerativo impedisce la decisione causale sugli eventi.",
        "Incertezza": "Il comportamento durante eventi rigenerativi resta sconosciuto in questa conferma indipendente.",
        "Limitazioni": "Task 18 non autorizzata; nessuno speedup dimostrato; cinque schedule non bastano per il supporto evento.",
        "Esito": "Misto", "Esperimenti": [ids["experiments"]],
    })
    mirror.local_upsert("claims", {
        "Codice stabile": "claims-giada-task17e-independent-causal-bounded-gate-c-v1",
        "Stato": "Aperta",
        "Limiti": "La conferma 17e passa i contrasti numerici ma non visita eventi nativi; Gate C resta aperto.",
    })
    write_json(ROOT / "data/airtable_snapshot.json", mirror.export_snapshot())
    assert mirror.verify()["valid"]
    return {"valid": True, "airtable_accessed": False, "run": run}


if __name__ == "__main__":
    print(json.dumps(main(), indent=2))
