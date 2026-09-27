"""Record archive-integrity checks and provenance limits for each teacher run."""

from __future__ import annotations

import json

from .audit_teacher_sources import audit
from .mirror import Mirror, ROOT, write_json


def main() -> dict:
    source = audit()
    if source["missing_archives"] or source["crc_invalid"]:
        raise RuntimeError("Source audit failed")
    mirror = Mirror()
    with mirror.connect() as conn:
        runs = {r["stable_code"]: r["record_id"] for r in conn.execute(
            "SELECT stable_code,record_id FROM v_records WHERE table_key='runs'")}
        artifacts = {r["stable_code"]: r["record_id"] for r in conn.execute(
            "SELECT stable_code,record_id FROM v_records WHERE table_key='artifacts'")}
    count = 0
    for row in source["entries"]:
        experiment = row["experiment"]
        archive_name = row["archive"].replace("\\", "/").rsplit("/", 1)[-1]
        run_id = runs["run-teacher-source-" + experiment.removeprefix("experiment-") + "-v1"]
        artifact_id = artifacts["artifact-teacher-source-zip-" + archive_name[:-4].replace("_", "-") + "-v1"]
        mirror.local_upsert("quality", {
            "Nome": "Integrità dell'archivio — " + experiment,
            "Codice stabile": "quality-teacher-archive-" + experiment.removeprefix("experiment-") + "-v1",
            "Descrizione": "Controllo retrospettivo tecnico della fonte locale; non è una nuova verifica biologica o statistica.",
            "Tipo controllo": "SHA-256, CRC ZIP e decodifica JSON",
            "Regola": "Archivio esistente; CRC di tutti i membri valido; report JSON principali decodificabili; annotare separatamente se il documento originale riporta l'hash ZIP.",
            "Esito": "Superato" if row["archive_exists"] and row["zip_crc_valid"] and row["reports_parse"] else "Fallito",
            "Dettagli": f"ZIP {archive_name}, SHA-256 {row['archive_sha256']}; SHA nel documento originale: {bool(row.get('archive_hash_in_note'))}. Quando assente, il legame con il documento rimane semantico e non crittografico.",
            "Run": [run_id],
            "Artefatti diagnostici": [artifact_id],
        })
        mirror.local_upsert("quality", {
            "Nome": "Corrispondenza hash documento-ZIP — " + experiment,
            "Codice stabile": "quality-teacher-document-hash-" + experiment.removeprefix("experiment-") + "-v1",
            "Descrizione": "Controllo distinto dall'integrità interna dello ZIP. Un documento senza hash non permette una verifica crittografica della sua associazione allo ZIP.",
            "Tipo controllo": "SHA-256 ZIP dichiarato nel documento di risultato",
            "Regola": "L'hash SHA-256 esatto dell'archivio compare nel documento versionato originale.",
            "Esito": "Superato" if row.get("archive_hash_in_note") else "Non eseguito",
            "Dettagli": ("Corrispondenza esatta con il documento originale." if row.get("archive_hash_in_note") else
                         "Hash ZIP non riportato nel documento originale; associazione per nome, struttura e valori, non certificazione crittografica.") + f" ZIP SHA-256 {row['archive_sha256']}.",
            "Run": [run_id],
            "Artefatti diagnostici": [artifact_id],
        })
        count += 1
    write_json(ROOT / "data" / "airtable_snapshot.json", mirror.export_snapshot())
    status = mirror.verify()
    if not status["valid"]:
        raise RuntimeError("Mirror verification failed")
    return {"checks": count, "valid": status["valid"], "snapshot_hash": status["snapshot_hash"]}


if __name__ == "__main__":
    print(json.dumps(main(), ensure_ascii=False, indent=2))
