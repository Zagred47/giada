"""Read-only relational coverage check for the historical teacher experiments."""

from __future__ import annotations

import json

from .audit_teacher_sources import MANIFEST, audit
from .mirror import Mirror


def coverage() -> dict:
    sources = audit()
    mapping = json.loads(MANIFEST.read_text(encoding="utf-8"))
    with Mirror().connect() as conn:
        record = {row["stable_code"]: (row["table_key"], row["record_id"])
                  for row in conn.execute("SELECT table_key,stable_code,record_id FROM v_records")}
        links = {}
        for row in conn.execute("SELECT source_table,source_id,role,target_id FROM v_links"):
            links.setdefault((row["source_table"], row["source_id"], row["role"]), set()).add(row["target_id"])
    rows = []
    for code in mapping:
        exp_id = record[code][1]
        proto = {source_id for (table, source_id, role), targets in links.items()
                 if table == "protocols" and role == "Esperimento" and exp_id in targets}
        evaluations = {source_id for (table, source_id, role), targets in links.items()
                       if table == "evaluations" and role == "Protocollo" and targets & proto}
        observations = {source_id for (table, source_id, role), targets in links.items()
                        if table == "observations" and role == "Specifica di valutazione" and targets & evaluations}
        findings = {source_id for (table, source_id, role), targets in links.items()
                    if table == "findings" and role == "Esperimenti" and exp_id in targets}
        runs = {record["run-teacher-source-" + code.removeprefix("experiment-") + "-v1"][1]}
        source_artifacts = {target for run_id in runs for target in links.get(("runs", run_id, "Artefatti prodotti"), ())}
        analyses = {target for finding_id in findings for target in links.get(("findings", finding_id, "Analisi"), ())}
        qualities = {source_id for (table, source_id, role), targets in links.items()
                     if table == "quality" and role == "Run" and targets & runs}
        predictions = {target for proto_id in proto for target in links.get(("protocols", proto_id, "Predizioni"), ())}
        rows.append({"experiment": code, "protocols": len(proto), "predictions": len(predictions),
                     "runs": len(runs), "source_artifacts": len(source_artifacts),
                     "evaluations": len(evaluations), "observations": len(observations),
                     "findings": len(findings), "analyses_linked_to_findings": len(analyses),
                     "quality_checks": len(qualities)})
    required = ("protocols", "predictions", "runs", "source_artifacts", "evaluations", "observations", "findings", "analyses_linked_to_findings", "quality_checks")
    return {"schema_version": "giada-teacher-relational-coverage-v1",
            "experiment_count": len(rows), "source_archive_count": sources["unique_archives"],
            "hash_matched_experiment_count": sources["hash_matched"],
            "all_core_links_present": all(row[key] > 0 for row in rows for key in required),
            "missing_by_field": {key: [row["experiment"] for row in rows if row[key] == 0] for key in required},
            "rows": rows}


if __name__ == "__main__":
    print(json.dumps(coverage(), ensure_ascii=False, indent=2))
