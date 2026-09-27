"""Index exact scalar summary metrics from verified GIADA teacher ZIP reports.

The ZIP remains authoritative for full-resolution arrays and nested diagnostics.
This index never interprets configuration numbers as observations, never folds
distinct sites/seeds, and records the JSON path and exact source hash.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import zipfile
from pathlib import Path

from .audit_teacher_sources import MANIFEST, audit
from .mirror import Mirror, ROOT, write_json


METRIC_NAME = re.compile(r"rmse|error|score|loss|gain|drift|latency|fraction|violation|difference|delta|retention", re.I)
EXCLUDED_ROOTS = {"config", "configuration", "support", "roles", "environment", "selection_freeze",
                  "prerequisite", "source", "sealed_contract", "fresh_contract", "data_contract",
                  "corpus_contract", "task14_prerequisite", "task15b_prerequisite"}


def walk_summary(value: dict):
    for first, obj in value.items():
        if first in EXCLUDED_ROOTS:
            continue
        if isinstance(obj, (int, float)) and not isinstance(obj, bool):
            if METRIC_NAME.search(first):
                yield (first,), float(obj)
        elif isinstance(obj, dict):
            for second, scalar in obj.items():
                if isinstance(scalar, (int, float)) and not isinstance(scalar, bool):
                    if METRIC_NAME.search(first) or METRIC_NAME.search(second):
                        yield (first, second), float(scalar)


def main() -> dict:
    source = audit()
    if source["missing_archives"] or source["crc_invalid"]:
        raise RuntimeError("Source audit failed")
    mapping = json.loads(MANIFEST.read_text(encoding="utf-8"))
    mirror = Mirror()
    with mirror.connect() as conn:
        experiment_ids = {r["stable_code"]: r["record_id"] for r in conn.execute(
            "SELECT stable_code,record_id FROM v_records WHERE table_key='experiments'")}
        protocols = {r["target_id"]: r["source_id"] for r in conn.execute(
            "SELECT source_id,target_id FROM v_links WHERE source_table='protocols' AND role='Esperimento'")}
        run_ids = {r["stable_code"]: r["record_id"] for r in conn.execute(
            "SELECT stable_code,record_id FROM v_records WHERE table_key='runs'")}
        artifact_ids = {r["stable_code"]: r["record_id"] for r in conn.execute(
            "SELECT stable_code,record_id FROM v_records WHERE table_key='artifacts'")}
    source_by_experiment = {row["experiment"]: row for row in source["entries"]}
    count = 0
    for experiment_code, (archive_name, _) in mapping.items():
        row = source_by_experiment[experiment_code]
        if not row["json_reports"]:
            continue
        protocol_id = protocols[experiment_ids[experiment_code]]
        run_id = run_ids["run-teacher-source-" + experiment_code.removeprefix("experiment-") + "-v1"]
        artifact_id = artifact_ids["artifact-teacher-source-zip-" + archive_name[:-4].replace("_", "-") + "-v1"]
        with zipfile.ZipFile(Path(row["archive"])) as handle:
            report_path = row["json_reports"][0]
            report = json.loads(handle.read(report_path))
        for path, value in walk_summary(report):
            if not math.isfinite(value):
                continue
            dotted = ".".join(path)
            identity = hashlib.sha256((experiment_code + "\0" + report_path + "\0" + dotted).encode()).hexdigest()[:16]
            metric = mirror.local_upsert("metrics", {
                "Nome": dotted,
                "Codice stabile": "metric-json-summary-" + identity + "-v1",
                "Descrizione": f"Scalare del report JSON originale, percorso {dotted}. Formula non ricostruita dal nome; usare il codice originale per il calcolo.",
                "Famiglia": "Regressione" if "rmse" in dotted.lower() else "Altro",
                "Direzione": "Minimizzare" if any(word in dotted.lower() for word in ("rmse", "error", "loss", "violation")) else "Descrittiva",
                **({"Unità": "mV"} if dotted.lower().endswith("_mv") else {}),
            })
            evaluation = mirror.local_upsert("evaluations", {
                "Nome": f"{experiment_code}: {dotted}",
                "Codice stabile": "evaluation-json-summary-" + identity + "-v1",
                "Descrizione": f"Indice a precisione nativa del valore presente in {archive_name}!{report_path}, percorso JSON {dotted}. Nessuna riaggregazione o nuova selezione. SHA-256 ZIP {row['archive_sha256']}.",
                "Versione": "backfill-v1",
                "Target": dotted,
                "Popolazione e regioni": "Come specificato dal report JSON originale; questo indice non inferisce indipendenza statistica.",
                "Aggregazione e pesi": "Valore scalare già aggregato nel report originale; nessuna nuova aggregazione nel backfill.",
                "Ruolo": "Diagnostica",
                "Protocollo": [protocol_id],
                "Metrica": [metric["record_id"]],
            })
            mirror.local_upsert("observations", {
                "Nome": f"{experiment_code}: {dotted}",
                "Codice stabile": "observation-json-summary-" + identity + "-v1",
                "Descrizione": f"Valore numerico esatto del campo {dotted} in {archive_name}!{report_path}. ZIP SHA-256 {row['archive_sha256']}. Non è un risultato indipendente dalle altre metriche dello stesso report.",
                "Valore": value,
                "Strato o sottogruppo": dotted,
                "Specifica di valutazione": [evaluation["record_id"]],
                "Run": [run_id],
                "Artefatti dettagliati": [artifact_id],
            })
            count += 1
    write_json(ROOT / "data" / "airtable_snapshot.json", mirror.export_snapshot())
    status = mirror.verify()
    if not status["valid"]:
        raise RuntimeError("Mirror failed verification")
    return {"exact_json_summary_observations": count, "verified": status["valid"], "snapshot_hash": status["snapshot_hash"]}


if __name__ == "__main__":
    print(json.dumps(main(), ensure_ascii=False, indent=2))
