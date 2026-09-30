"""Record the incomplete first Kaggle Task17d attempt in the SQLite mirror."""

from __future__ import annotations

import hashlib
import json

from .mirror import Mirror, ROOT, write_json


def main():
    path = ROOT.parent / "experiments/results/task17d_initial_preflight_failure/README.md"
    payload = path.read_bytes()
    mirror = Mirror()
    block_code = "blocks-giada-task17d-native-solver-paired-v1"
    matches = mirror.query(
        "SELECT record_id FROM v_records WHERE table_key='blocks' "
        f"AND stable_code='{block_code}'"
    )["rows"]
    if len(matches) != 1:
        raise RuntimeError(f"Missing preregistered Task17d block: {block_code}")
    artifact = mirror.local_upsert("artifacts", {
        "Codice stabile": "artifacts-giada-task17d-initial-preflight-failure-v1",
        "Nome": "Task17d — analisi del primo arresto tecnico Kaggle",
        "Tipo": "Report",
        "Percorso": str(path.relative_to(ROOT.parent)).replace("\\", "/"),
        "SHA-256": hashlib.sha256(payload).hexdigest(),
        "Dimensione byte": len(payload),
        "Versione": "v1",
        "Descrizione": "Fonte: ZIP utente SHA-256 5606e296ce66e92a003181175ce219efa0eaf1ea60cab103e143a4f291dec0d3445; nessun final_report scientifico.",
    })["record_id"]
    run = mirror.local_upsert("runs", {
        "Codice stabile": "runs-giada-task17d-initial-preflight-failure-v1",
        "Nome": "Task17d 54a6aed — float32 atolscale, run incompleto",
        "Stato": "Fallita",
        "Blocco": [matches[0]["record_id"]],
        "Artefatti prodotti": [artifact],
        "Hardware e ambiente": "Kaggle, NEURON, revisione 54a6aed3672c9cec4fa5b6c695d5f0d378919531",
        "Descrizione": "Trial default/tight/ultra completati; arresto al preflight calcium_scaled per confronto troppo rigido di un valore float32. Nessun report finale.",
        "Validità tecnica": "Incompleto; nessun GO/NO-GO scientifico. Ipotesi Task17d ancora aperte.",
    })["record_id"]
    write_json(ROOT / "data/airtable_snapshot.json", mirror.export_snapshot())
    assert mirror.verify()["valid"]
    return {"valid": True, "airtable_accessed": False, "run": run}


if __name__ == "__main__":
    print(json.dumps(main(), indent=2))
