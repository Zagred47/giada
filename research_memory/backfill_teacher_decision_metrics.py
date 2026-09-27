"""Index selected nested, decision-relevant metrics at native JSON precision.

Selection below is explicit by experiment and JSON report member. Entire
per-example arrays remain in their immutable source ZIP, not in SQLite.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import sys
import zipfile
from pathlib import Path

from .audit_teacher_sources import MANIFEST, audit
from .mirror import Mirror, ROOT, write_json


SELECTORS = {
    "experiment-task5-primitive-scaling-v1": [("final_report.json", "physical", r"(?:mean_score|max_seed_score)$"),
                                              ("final_report.json", "numerical", r"score$")],
    "experiment-task9c-dense-physiological-teacher-replay-v1": [("final_report.json", "metrics", r"(?:rmse|error)$")],
    "experiment-roadmap-task12-cahva-current-v1": [("task12_current_report.json", "arms", r"(?:rmse_ma_cm2|mae_ma_cm2|voltage_mv|^m$|^h$)$")],
    "experiment-roadmap-task13-cahva-timing-v1": [("task13_timing_report.json", "current_all_samples", r"(?:rmse|maximum_absolute_error)$"),
                                                   ("task13_timing_report.json", "current_active_samples", r"(?:rmse|maximum_absolute_error)$"),
                                                   ("task13_timing_report.json", "inferred_conductance_metrics_us_cm2", r"(?:rmse|maximum_absolute_error)$")],
    "experiment-roadmap-task14-cahva-parameter-variation-v1": [("task14_parameter_matrix_report.json", "results", r"(?:rmse|error|difference|delta)(?:_[a-z0-9]+)*$")],
    "experiment-roadmap-task15-cahva-current-architecture-v1": [("final_report.json", "sealed_one_step_and_available_rollout", r"(?:rmse|mae)(?:_[a-z0-9]+)*$"),
                                                                    ("final_report.json", "registered_decision", r"(?:rmse|gain|improvement)(?:_[a-z0-9]+)*$")],
    "experiment-roadmap-task15b-cahva-gate-bottleneck-v1": [("final_report.json", "sealed", r"(?:rmse|mae|score)(?:_[a-z0-9]+)*$")],
    "experiment-task6-voltage-path-stress-v1": [("final_report.json", "sealed", r"(?:normalized_score|m_rmse|h_rmse|open_rmse)$")],
    "experiment-task8-cahva-continuous-path-stress-v1": [("final_report.json", "sealed", r"(?:normalized_score|m_rmse|h_rmse|open_rmse)$")],
    "experiment-task9b-physiological-path-floor-forensic-v1": [("final_report.json", "global_methods", r"(?:rmse|error)$")],
}


def walk_dict(value, path=()):
    if isinstance(value, dict):
        for key, child in value.items():
            yield from walk_dict(child, path + (str(key),))
    elif isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
        yield path, float(value)


def main() -> dict:
    source = audit()
    if source["missing_archives"] or source["crc_invalid"]:
        raise RuntimeError("Source audit failed")
    mapping = json.loads(MANIFEST.read_text(encoding="utf-8"))
    source_by_experiment = {row["experiment"]: row for row in source["entries"]}
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
        existing_observations = {r["stable_code"] for r in conn.execute(
            "SELECT stable_code FROM v_records WHERE table_key='observations'")}
    counts = {}
    for experiment_code, selectors in SELECTORS.items():
        if len(sys.argv) > 1 and experiment_code not in sys.argv[1:]:
            continue
        archive_name = mapping[experiment_code][0]
        entry = source_by_experiment[experiment_code]
        protocol_id = protocols[experiment_ids[experiment_code]]
        run_id = run_ids["run-teacher-source-" + experiment_code.removeprefix("experiment-") + "-v1"]
        artifact_id = artifact_ids["artifact-teacher-source-zip-" + archive_name[:-4].replace("_", "-") + "-v1"]
        with zipfile.ZipFile(Path(entry["archive"])) as handle:
            members = handle.namelist()
            records = {}
            for suffix, root, pattern in selectors:
                matches = [name for name in members if name.endswith(suffix)]
                if len(matches) != 1:
                    raise RuntimeError(f"Expected one {suffix} in {archive_name}, found {matches}")
                member = matches[0]
                report = records.setdefault(member, json.loads(handle.read(member)))
                if root not in report or not isinstance(report[root], dict):
                    raise RuntimeError(f"Missing metric root {root} in {member}")
                regex = re.compile(pattern)
                for path, value in walk_dict(report[root], (root,)):
                    if not regex.search(path[-1]):
                        continue
                    dotted = ".".join(path)
                    identity = hashlib.sha256((experiment_code + "\0" + member + "\0" + dotted).encode()).hexdigest()[:16]
                    if "observation-decision-" + identity + "-v1" in existing_observations:
                        continue
                    metric = mirror.local_upsert("metrics", {
                        "Nome": dotted,
                        "Codice stabile": "metric-decision-" + identity + "-v1",
                        "Descrizione": f"Metrica riportata in {member}, percorso JSON {dotted}. Il nome non sostituisce la formula originale del codice.",
                        "Famiglia": "Regressione" if "rmse" in dotted.lower() else "Altro",
                        "Direzione": "Minimizzare" if any(part in dotted.lower() for part in ("rmse", "mae", "error")) else "Descrittiva",
                        **({"Unità": "mV"} if dotted.lower().endswith("_mv") else {}),
                    })
                    evaluation = mirror.local_upsert("evaluations", {
                        "Nome": f"{experiment_code}: {dotted}",
                        "Codice stabile": "evaluation-decision-" + identity + "-v1",
                        "Descrizione": f"Campo numerico decisionale esatto di {archive_name}!{member}, percorso {dotted}. Nessun ricalcolo; SHA-256 ZIP {entry['archive_sha256']}.",
                        "Versione": "backfill-v1",
                        "Target": dotted,
                        "Popolazione e regioni": "Definite nel report originale; il percorso JSON preserva braccio e strato.",
                        "Aggregazione e pesi": "Aggregazione preesistente nel report originale; nessuna nuova aggregazione.",
                        "Ruolo": "Diagnostica",
                        "Protocollo": [protocol_id],
                        "Metrica": [metric["record_id"]],
                    })
                    mirror.local_upsert("observations", {
                        "Nome": f"{experiment_code}: {dotted}",
                        "Codice stabile": "observation-decision-" + identity + "-v1",
                        "Descrizione": f"Scalare originale {dotted} in {archive_name}!{member}. ZIP SHA-256 {entry['archive_sha256']}. Non indipendente dalle altre celle dello stesso esperimento.",
                        "Valore": value,
                        "Strato o sottogruppo": dotted,
                        "Specifica di valutazione": [evaluation["record_id"]],
                        "Run": [run_id],
                        "Artefatti dettagliati": [artifact_id],
                    })
                    counts[experiment_code] = counts.get(experiment_code, 0) + 1
        print(f"[GIADA DB] {experiment_code}: {counts.get(experiment_code, 0)} metriche", flush=True)
    write_json(ROOT / "data" / "airtable_snapshot.json", mirror.export_snapshot())
    verification = mirror.verify()
    if not verification["valid"]:
        raise RuntimeError("Mirror verification failed")
    return {"counts": counts, "verified": verification["valid"], "snapshot_hash": verification["snapshot_hash"]}


if __name__ == "__main__":
    print(json.dumps(main(), ensure_ascii=False, indent=2))
