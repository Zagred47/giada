"""Index Tasks 0.4–0.6 without promoting setup contracts to experiments."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import zipfile

from .mirror import Mirror, ROOT, write_json


PREPARATORY = (
    ("0-4-cahva-double-oracle", "teacher_double_oracle_result_v1.md", "teacher_double_oracle_result_v1.json", "artifact-giada-cahva-double-oracle-result-v1"),
    ("0-5-atomic-domain-splits", "teacher_atomic_domain_splits_v1.md", "teacher_atomic_domain_splits_v1.json", "artifact-giada-atomic-domain-splits-v1"),
    ("0-6-common-gpu-baseline", "teacher_common_gpu_baseline_v1.md", "teacher_common_gpu_baseline_v1.json", "artifact-giada-common-gpu-baseline-v1"),
)


def main() -> dict:
    mirror = Mirror()
    with mirror.connect() as conn:
        records = {(r["table_key"], r["stable_code"]): r["record_id"] for r in conn.execute(
            "SELECT table_key,stable_code,record_id FROM v_records")}
        old_analysis = {}
        for row in conn.execute("SELECT source_id,target_id FROM v_links WHERE source_table='findings' AND role='Analisi'"):
            old_analysis.setdefault(row["source_id"], []).append(row["target_id"])

    archive = ROOT / "source_archives" / "giada_teacher_double_oracle.zip"
    archive_bytes = archive.read_bytes()
    archive_hash = hashlib.sha256(archive_bytes).hexdigest()
    expected_hash = "4bd705f37c61985c2685c84a290cd06364778d104962c3177b4df583d16bbae1"
    if archive_hash != expected_hash:
        raise RuntimeError("Task 0.4 ZIP hash mismatch")
    with zipfile.ZipFile(archive) as handle:
        if handle.testzip() is not None:
            raise RuntimeError("Task 0.4 ZIP CRC mismatch")
    result = json.loads((ROOT.parent / "experiments" / "teacher_double_oracle_result_v1.json").read_text(encoding="utf-8"))
    if not result.get("valid"):
        raise RuntimeError("Task 0.4 result is not valid")
    zip_artifact = mirror.local_upsert("artifacts", {
        "Nome": archive.name,
        "Codice stabile": "artifact-teacher-source-zip-giada-teacher-double-oracle-v1",
        "Descrizione": "ZIP originale Task 0.4, verificato SHA-256 e CRC; hash dichiarato nel risultato versionato.",
        "Tipo": "Report",
        "Percorso": "research_memory/source_archives/" + archive.name,
        "SHA-256": archive_hash,
        "Dimensione byte": len(archive_bytes),
        "Versione": "backfill-2026-09-27",
    })
    run = mirror.local_upsert("runs", {
        "Nome": "Esecuzione aggregata GIADA Task 0.4 double oracle",
        "Codice stabile": "run-teacher-source-task-0-4-cahva-double-oracle-v1",
        "Descrizione": "Risultato dell'oracolo atomico sul Ca_HVA compilato; non prova validità sul neurone accoppiato. Il passaggio dalla chiamata HOC diretta alla sezione elettricamente silente è documentato nella fonte.",
        "Stato": "Completata",
        "Validità tecnica": "ZIP e report validi; 7360 casi, zero failure, massimo errore assoluto 2.220446049250313e-16.",
        "Artefatti prodotti": [zip_artifact["record_id"]],
    })
    mirror.local_upsert("quality", {
        "Nome": "Integrità fonte Task 0.4",
        "Codice stabile": "quality-teacher-source-task-0-4-cahva-double-oracle-v1",
        "Descrizione": "Controllo retrospettivo di provenienza, non nuova validazione scientifica.",
        "Tipo controllo": "SHA-256, CRC ZIP e JSON",
        "Regola": "Hash uguale a quello dichiarato nel risultato; tutti i membri ZIP leggibili.",
        "Esito": "Superato",
        "Dettagli": f"SHA-256 {archive_hash}; report valid={result['valid']}.",
        "Run": [run["record_id"]],
        "Artefatti diagnostici": [zip_artifact["record_id"]],
    })
    analyses = []
    for suffix, note_name, json_name, artifact_code in PREPARATORY:
        note = ROOT.parent / "experiments" / note_name
        payload = ROOT.parent / "experiments" / json_name
        if not note.is_file() or not payload.is_file():
            raise FileNotFoundError(f"Missing preparatory source: {note_name} or {json_name}")
        json.loads(payload.read_text(encoding="utf-8"))
        finding_code = "finding-task-" + suffix + "-v1"
        finding_id = records[("findings", finding_code)]
        artifact_id = records[("artifacts", artifact_code)]
        analysis = mirror.local_upsert("analyses", {
            "Nome": "Analisi documentata GIADA Task " + suffix[:3].replace("-", "."),
            "Codice stabile": "analysis-teacher-preparatory-task-" + suffix + "-v1",
            "Descrizione": f"Indice retrospettivo delle fonti versionate experiments/{note_name} e experiments/{json_name}. "
                           + ("Task 0.4: risultato sperimentale dell'oracolo atomico; ZIP SHA-256 " + archive_hash + "." if suffix.startswith("0-4") else "Contratto preparatorio, non presentato come misura di performance."),
            "Versione": "backfill-v1",
            "Procedura": "Consultare i documenti sorgente; nessun nuovo fit, test o riclassificazione causale nel backfill.",
            **({"Run analizzati": [run["record_id"]]} if suffix.startswith("0-4") else {}),
            "Output artefatti": [artifact_id] + ([zip_artifact["record_id"]] if suffix.startswith("0-4") else []),
        })
        links = list(dict.fromkeys(old_analysis.get(finding_id, []) + [analysis["record_id"]]))
        mirror.local_upsert("findings", {"Analisi": links}, finding_id)
        analyses.append(analysis["record_id"])
    write_json(ROOT / "data" / "airtable_snapshot.json", mirror.export_snapshot())
    status = mirror.verify()
    if not status["valid"]:
        raise RuntimeError("Mirror verification failed")
    return {"linked_preparatory_analyses": len(analyses), "task_0_4_archive_sha256": archive_hash,
            "valid": True, "snapshot_hash": status["snapshot_hash"]}


if __name__ == "__main__":
    print(json.dumps(main(), ensure_ascii=False, indent=2))
