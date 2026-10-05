"""Register the independently audited Task28 result in the SQLite-only mirror."""
import hashlib
import json

from .mirror import ROOT, write_json
from .validated_batch import ValidatedBatchMirror


def main():
    repo = ROOT.parent
    folder = repo / "experiments/results/task28_kaggle_4c3cba7"
    report = json.loads((folder / "final_report.json").read_text(encoding="utf-8"))
    audit = json.loads((folder / "result_audit.json").read_text(encoding="utf-8"))
    contract = json.loads((folder / "run_contract.json").read_text(encoding="utf-8"))
    assert audit["valid"] and report["valid"] and report["teacher_forced_ionic_block_passed"]
    assert audit["fresh_rows"] == 144 and audit["path_rows"] == 48 and audit["all_arms_passed"]
    assert not report["gate_d_completed"] and not report["task29_authorized"]
    assert not report["fresh_used_for_selection"] and not report["training_performed"]
    mirror = ValidatedBatchMirror()
    cache = {}

    def put(table, key, **fields):
        rid = mirror.local_upsert(table, {"Codice stabile": f"{table}-giada-roadmap-task28-{key}-v1", **fields})["record_id"]
        cache[table, key] = rid
        return rid

    def ref(table, key):
        if (table, key) not in cache:
            code = f"{table}-giada-roadmap-task28-{key}-v1"
            rows = mirror.query("SELECT record_id FROM v_records WHERE stable_code='" + code + "'")["rows"]
            assert len(rows) == 1, code
            cache[table, key] = rows[0]["record_id"]
        return cache[table, key]

    artifacts = {}
    for path in sorted(folder.iterdir()):
        if path.suffix not in (".json", ".zip", ".log"):
            continue
        artifacts[path.name] = put("artifacts", "result-" + path.stem, **{
            "Nome": "Task28 " + path.name,
            "Tipo": "Checkpoint" if path.suffix == ".zip" else "Report",
            "Percorso": path.relative_to(repo).as_posix(),
            "SHA-256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "Dimensione byte": path.stat().st_size, "Versione": "v1",
        })

    specs = {
        "worst_gate_rmse": "gate_rmse",
        "worst_individual_current_normalized_rmse": "current_individual_normalized_rmse",
        "worst_total_current_normalized_rmse": "current_total_normalized_rmse",
        "path_gate_rmse": "path_gate_rmse",
        "path_individual_current_normalized_rmse": "path_current_normalized_rmse",
    }
    observations = []
    counts = {"fresh": 0, "path": 0}
    path_block = put("blocks", "imposed-path", **{
        "Nome": "Task28 imposed paths", "Protocollo": [ref("protocols", "frozen")],
        "Seed": "280505..280508", "Descrizione": "Four externally prescribed voltage/calcium trajectories, never autonomous feedback.",
        "Regola di appaiamento": "Identical imposed paths and initial gate state across all frozen arms.",
    })
    for kind, rows in (("fresh", report["fresh_rows"]), ("path", report["path_rows"])):
        for row in rows:
            key = (f"{kind}-{row['domain']}-{row['support_seed']}-{row['family']}-{row['arm']}-{row['model_seed']}"
                   if kind == "fresh" else f"{kind}-{row['path']}-{row['family']}-{row['arm']}-{row['model_seed']}")
            run = put("runs", key, **{
                "Nome": "Task28 " + key, "Stato": "Completata",
                "Braccio": [ref("arms", row["family"] + "-" + row["arm"])],
                "Blocco": [ref("blocks", "fresh")] if kind == "fresh" else [path_block],
                "Seed": str(row["support_seed"] if kind == "fresh" else row["model_seed"]),
                "Artefatti prodotti": [artifacts["final_report.json"], artifacts["artifact_bundle.zip"]],
                "Validità tecnica": "Kaggle COMPLETE/exit0; immutable Task27 parent and code provenance verified; frozen evaluation only.",
            })
            metrics = (("worst_gate_rmse", row["worst_gate_rmse"]),
                       ("worst_individual_current_normalized_rmse", row["worst_individual_current_normalized_rmse"]))
            if kind == "fresh":
                metrics += (("worst_total_current_normalized_rmse", row["worst_total_current_normalized_rmse"]),)
            for metric, value in metrics:
                eval_key = specs[metric] if kind == "fresh" else specs["path_" + metric.replace("worst_", "")]
                observations.append(put("observations", key + "-" + metric, **{
                    "Nome": "Task28 " + key + " " + metric, "Valore": float(value),
                    "Run": [run], "Specifica di valutazione": [ref("evaluations", eval_key)],
                    "Artefatti dettagliati": [artifacts["final_report.json"]],
                    "Strato o sottogruppo": key,
                    "Descrizione": "Frozen teacher-forced; per-channel gates and signed-current panels retained losslessly in linked final report.",
                }))
            counts[kind] += 1

    native_run = put("runs", "native-six", **{
        "Nome": "Task28 six canonical formula native audits", "Stato": "Completata",
        "Artefatti prodotti": [artifacts["native_audit.json"]],
        "Validità tecnica": "Six canonical teacher mechanisms checked against compiled NEURON; no frozen candidate reselection.",
    })
    native = json.loads((folder / "native_audit.json").read_text(encoding="utf-8"))
    for row in native["rows"]:
        for metric in ("maximum_step_error", "maximum_current_error_ma_cm2"):
            observations.append(put("observations", "native-" + row["channel"] + "-" + metric, **{
                "Nome": "Task28 " + row["channel"] + " " + metric,
                "Valore": float(row[metric]), "Run": [native_run],
                "Artefatti dettagliati": [artifacts["native_audit.json"]],
                "Strato o sottogruppo": row["channel"],
                "Descrizione": "Canonical formula versus native NEURON mechanism under imposed local state.",
            }))

    finding = put("findings", "composition", **{
        "Nome": "Task28 eleven-channel ionic block passes teacher-forced composition",
        "Esito": "Positivo", "Esperimenti": [ref("experiments", "hybrid")],
        "Osservazioni": observations,
        "Risultato": ("All four frozen Task27 arms pass all 144 independent fresh panels and 48 imposed paths. "
                      "Both-arm composition GO in independent and shared-head families. "
                      f"Worst fresh gate RMSE={audit['worst_fresh_gate_rmse']:.9g}; "
                      f"worst individual current normalized RMSE={audit['worst_fresh_individual_current_normalized_rmse']:.9g}; "
                      f"worst total current normalized RMSE={audit['worst_fresh_total_current_normalized_rmse']:.9g}. "
                      "Six formula channels pass native audits; all source and parent hashes and row-level thresholds independently recomputed."),
        "Limitazioni": contract["limits"],
    })
    put("evidence", "composition", **{
        "Nome": "Task28 teacher-forced composition evidence", "Esito": "Sostiene",
        "Affermazione valutata": [ref("claims", "composition")],
        "Risultati a sostegno": [finding],
        "Argomentazione": "Independent audit recomputes complete factorial, per-row conjunction, provenance and native formula correctness.",
    })
    put("claims", "composition", **{"Stato": "Supportata nel dominio"})
    compute_finding = put("findings", "compute-unmeasured", **{
        "Nome": "Task28 Gate D compute reduction not measured", "Esito": "Misto",
        "Esperimenti": [ref("experiments", "hybrid")],
        "Risultato": "No same-device material compute benchmark was run. Gate D is incomplete and Task29 is not authorized.",
        "Limitazioni": "Do not infer speedup from parameter count or teacher-forced accuracy.",
    })
    put("evidence", "compute-unmeasured", **{
        "Nome": "Task28 missing Gate D compute evidence", "Esito": "Indeterminata",
        "Affermazione valutata": [ref("claims", "compute")],
        "Risultati a sostegno": [compute_finding],
        "Argomentazione": "Registered materiality threshold is 10%; no paired measurement exists, so the claim remains open.",
    })
    put("decisions", "post28", **{
        "Nome": "Task28 teacher-forced GO; Gate D remains open", "Esito": "Continuare",
        "Risultati": [finding, compute_finding],
        "Motivazione": "Complete teacher-forced ionic composition passes; prepare a paired, same-device material-compute audit before any Task29 promotion.",
        "Condizioni di revisione": "Only a separately preregistered Gate D benchmark can authorize Task29. CaDynamics, passive/axial/synaptic/closed-loop dynamics remain outside Task28.",
    })
    put("experiments", "hybrid", **{"Stato": "Concluso"})
    mirror.commit_batch()
    write_json(ROOT / "data/airtable_snapshot.json", mirror.export_snapshot())
    assert mirror.verify()["valid"]
    print(json.dumps({"valid": True, "runs": counts, "observations": len(observations),
                      "task28_passed": True, "gate_d_completed": False, "airtable_accessed": False}))


if __name__ == "__main__":
    main()
