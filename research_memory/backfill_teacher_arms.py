"""Backfill experimentally named conditions without guessing hidden factors.

Only explicit mapping keys in original JSON reports become arms. A role is
assigned only when the original condition name unambiguously states it.
"""

from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

from .audit_teacher_sources import MANIFEST, audit
from .mirror import Mirror, ROOT, write_json


ROOTS = {
    "experiment-task1-cahva-m-atomic-learnability-v1": ["aggregate_sealed_rmse"],
    "experiment-task1b-m-gate-diagnosis-v1": ["fresh_confirmation_macro_rmse"],
    "experiment-task2-cahva-h-atomic-learnability-v1": ["aggregate_sealed_rmse"],
    "experiment-task2b-cahva-h-rate-identifiability-v1": ["metrics"],
    "experiment-task3-joint-mh-cell-v1": ["metrics"],
    "experiment-task3b-shared-mh-optimization-diagnosis-v1": ["scores_at_50000"],
    "experiment-task3d-joint-gate-generalization-matrix-v1": ["summaries"],
    "experiment-task4-paired-primitive-matrix-v1": ["learned", "numerical"],
    "experiment-task5-primitive-scaling-v1": ["physical", "numerical"],
    "experiment-task6-voltage-path-stress-v1": ["sealed.metrics"],
    "experiment-task8-cahva-continuous-path-stress-v1": ["sealed.metrics"],
    "experiment-task9b-physiological-path-floor-forensic-v1": ["global_methods"],
    "experiment-task9c-dense-physiological-teacher-replay-v1": ["metrics"],
    "experiment-roadmap-task11-cahva-causal-operator-v1": ["H0_input_contract", "H1_coupled_numerical_stages"],
    "experiment-roadmap-task14-cahva-parameter-variation-v1": ["results"],
    "experiment-roadmap-task15-cahva-current-architecture-v1": ["selected"],
    "experiment-roadmap-task15b-cahva-gate-bottleneck-v1": ["selected"],
    "experiment-roadmap-task15c-cahva-interface-bridge-v1": ["views"],
}


def role(name: str) -> str | None:
    low = name.lower()
    if "oracle" in low or low in {"formula", "formula_start_only", "formula_coarse_path"}:
        return "Oracle diagnostico"
    if "persistence" in low or low in {"baseline", "baseline_current", "baseline_extended", "pre_all"}:
        return "Baseline"
    if low in {"mechanism_off_hidden"}:
        return "Controllo negativo"
    return None


def main() -> dict:
    source = audit()
    if source["missing_archives"] or source["crc_invalid"]:
        raise RuntimeError("Source audit failed")
    mapping = json.loads(MANIFEST.read_text(encoding="utf-8"))
    source_by_experiment = {row["experiment"]: row for row in source["entries"]}
    mirror = Mirror()
    with mirror.connect() as conn:
        experiments = {r["stable_code"]: r["record_id"] for r in conn.execute(
            "SELECT stable_code,record_id FROM v_records WHERE table_key='experiments'")}
        protocols = {r["target_id"]: r["source_id"] for r in conn.execute(
            "SELECT source_id,target_id FROM v_links WHERE source_table='protocols' AND role='Esperimento'")}
    counts = {}
    for experiment_code, roots in ROOTS.items():
        row = source_by_experiment[experiment_code]
        archive_name = mapping[experiment_code][0]
        protocol_id = protocols[experiments[experiment_code]]
        with zipfile.ZipFile(Path(row["archive"])) as handle:
            member = row["json_reports"][0]
            report = json.loads(handle.read(member))
        for path in roots:
            values = report
            for key in path.split("."):
                values = values[key]
            if not isinstance(values, dict):
                raise RuntimeError(f"Expected condition mapping at {archive_name}!{path}")
            for name in values:
                identity = hashlib.sha256((experiment_code + "\0" + path + "\0" + str(name)).encode()).hexdigest()[:16]
                mirror.local_upsert("arms", {
                    "Nome": str(name),
                    "Codice stabile": "arm-teacher-source-" + identity + "-v1",
                    "Descrizione": f"Condizione esplicita del report originale {archive_name}!{member}, chiave {path}.{name}. Indicizzata retrospettivamente; non implica una nuova prova o un nuovo run indipendente. ZIP SHA-256 {row['archive_sha256']}.",
                    "Configurazione residua": f"Percorso JSON: {path}.{name}; configurazione completa nello ZIP e nel protocollo originale.",
                    **({"Ruolo": role(str(name))} if role(str(name)) else {}),
                    "Protocollo": [protocol_id],
                })
                counts[experiment_code] = counts.get(experiment_code, 0) + 1
    write_json(ROOT / "data" / "airtable_snapshot.json", mirror.export_snapshot())
    status = mirror.verify()
    if not status["valid"]:
        raise RuntimeError("Mirror verification failed")
    return {"counts": counts, "valid": status["valid"], "snapshot_hash": status["snapshot_hash"]}


if __name__ == "__main__":
    print(json.dumps(main(), ensure_ascii=False, indent=2))
