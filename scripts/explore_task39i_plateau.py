"""Read-only SQLite retrieval and frozen-artifact diagnostics, not a new run.

No training, scientific selection, DB writes, threshold changes, or remote calls.
The output preserves SQL, bindings and returned evidence for review.
"""
import collections
import json
import sqlite3
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/task39i_plateau_analysis"
CURRENT = ROOT / "artifacts/task39i_recovered_cpu"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def coefficient_probe():
    """Inspect frozen I3 only; equivalent tau is a diagnostic, not inference."""
    import torch
    from src.giada_teacher import task39i_coupled_relaxation as i
    weights = torch.load(CURRENT / "frozen_checkpoint.pt",
                         map_location="cpu", weights_only=True)["plain"]
    model = i.model_class(torch)("plain", [17, 29, 43])
    model.load_state_dict({
        k: v if k == "gbar" else v[9:12] for k, v in weights.items()
    })
    path = ROOT / "experiments/results/task37_playground_v2/development__slow_recovery.npz"
    with np.load(path) as data:
        v, ca = data["voltage"][0], data["bridge_ca"][0]
        initial = data["initial_gates"].copy()
    true_u, true_z = i.p.coefficients(v.astype(float), ca.astype(float), .1)
    with torch.no_grad():
        u, z, _ = model.coefficients(torch.tensor(v)[None].expand(3, -1),
                                     torch.tensor(ca)[None].expand(3, -1))
    u, z = u.numpy().astype(float), z.numpy().astype(float)
    rows = []
    for case, channel in ((0, "K_Pst"), (1, "Nap_Et2")):
        k = list(i.p.ionic.CHANNELS).index(channel)
        for gate in range(*i.p.OFFSETS[k:k+2]):
            rows.append(dict(
                model="I3", case=case, channel=channel, gate=gate,
                voltage_mv=float(v[case]), initial=float(initial[case, gate]),
                u_reference=float(true_u[case, gate]),
                u_prediction=u[:, case, gate].tolist(),
                tau_reference_ms=float(-.1 / np.log1p(-true_z[case, gate])),
                tau_prediction_ms=(-.1 / np.log1p(-z[:, case, gate])).tolist(),
                relative_z_error=(z[:, case, gate] / true_z[case, gate] - 1).tolist(),
            ))
    return rows


def main():
    OUT.mkdir(exist_ok=True)
    db = sqlite3.connect(
        (ROOT / "research_memory/data/research_memory.sqlite").as_uri() + "?mode=ro",
        uri=True,
    )
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA query_only=ON")
    queries = []

    def query(name, sql, args=()):
        rows = [dict(r) for r in db.execute(sql, args)]
        queries.append(dict(name=name, sql=sql, parameters=args, rows=rows))
        return rows

    history = query("localized_and_dynamic_precedents", """
        SELECT * FROM findings
        WHERE "Codice stabile" LIKE '%task18b%'
           OR "Codice stabile" LIKE '%task18c%'
           OR "Codice stabile" LIKE '%task18d%'
           OR "Codice stabile" LIKE '%task24b%'
           OR "Codice stabile" IN (
              'finding-task2b-h-physical-tau-identifiability-repaired-v1',
              'finding-task3b-symmetric-rates-stabilize-shared-mh-v1',
              'finding-task3d-voltage-support-conditional-shape-repair-v1',
              'finding-task3e-full-repair-sealed-generalization-v1',
              'findings-giada-roadmap-task21-duration-v1',
              'findings-giada-roadmap-task26-gradient_interference-v1')
        ORDER BY "_created_time", "Codice stabile"
    """)
    ids = [r["_record_id"] for r in history]
    query("precedents_evidence_chains", """
        SELECT c."Codice stabile" claim_code,c.Stato claim_status,
               c.Enunciato claim,e."Codice stabile" evidence_code,
               e.Esito assessment,e.Argomentazione argument,
               e."Limiti e spiegazioni alternative" alternatives,
               f."Codice stabile" finding_code
        FROM v_evidence_chains ec
        JOIN claims c ON c._record_id=ec.claim_id
        JOIN evidence e ON e._record_id=ec.assessment_id
        JOIN findings f ON f._record_id=ec.finding_id
        WHERE ec.finding_id IN (""" + ",".join("?" for _ in ids) + ")", ids)
    query("loss_and_optimization_lessons", """
        SELECT * FROM lessons WHERE "Codice stabile" IN (
          'lesson-gradient-conflict-requires-interventional-benefit-v1',
          'lesson-causal-rate-supervision-identifies-physical-cell-v1',
          'lesson-parameter-matched-seed-variance-localizes-optimization-v1',
          'lessons-giada-task39f-result-temporal-v1',
          'lessons-giada-task39g-temporal-numerical-preflight-v1')
    """)
    query("recent_claims_and_evidence", """
        SELECT c."Codice stabile" claim_code,c.Stato claim_status,
               c.Enunciato claim,e.Esito assessment,
               e.Argomentazione argument,f."Codice stabile" finding_code
        FROM claims c LEFT JOIN v_evidence_chains ec ON ec.claim_id=c._record_id
        LEFT JOIN evidence e ON e._record_id=ec.assessment_id
        LEFT JOIN findings f ON f._record_id=ec.finding_id
        WHERE c."Codice stabile" LIKE '%task39g%residual%'
           OR c."Codice stabile" LIKE '%task39h%residual%'
        ORDER BY c."Codice stabile",e."Codice stabile"
    """)
    query("39g_39h_observed_case_counts", """
        SELECT CASE WHEN o."Codice stabile" LIKE '%task39g%' THEN '39g'
                    ELSE '39h' END task,
               json_extract(j.value,'$.model') model,
               json_extract(j.value,'$.role') role,
               COUNT(*) cases,SUM(json_extract(j.value,'$.metrics.passed')) passed
        FROM observations o,json_each(o.Descrizione,'$.cases') j
        WHERE (o."Codice stabile" LIKE 'observations-giada-task39g-result-primary-%'
            OR o."Codice stabile" LIKE 'observations-giada-task39h-result-primary-%')
        GROUP BY task,model,role ORDER BY task,role,passed DESC
    """)
    query("39h_H8_failed_channels", """
        SELECT json_extract(j.value,'$.family') family,
               json_extract(j.value,'$.case') case_index,
               json_extract(j.value,'$.seed') seed,
               json_extract(c.value,'$.channel') channel,
               json_extract(c.value,'$.nrmse') nrmse,
               json_extract(c.value,'$.event.offset_error_ms') offset_error_ms
        FROM observations o,json_each(o.Descrizione,'$.cases') j,
             json_each(j.value,'$.metrics.channels') c
        WHERE o."Codice stabile" LIKE
             'observations-giada-task39h-result-primary-joint_wH8-development-%'
          AND json_extract(c.value,'$.passed')=0
        ORDER BY family,case_index,seed,channel
    """)
    query("39i_registration_boundary", """
        SELECT 'findings' kind,"Codice stabile" code FROM findings
        WHERE "Codice stabile" LIKE '%task39i%'
        UNION ALL SELECT 'experiments',"Codice stabile" FROM experiments
        WHERE "Codice stabile" LIKE '%task39i%'
    """)
    db.close()
    records = read(CURRENT / "recurrence_evaluation.json")
    residuals = []
    for r in records:
        if r["model"] == "joint_wI3" and r["role"] == "development":
            for channel in r["metrics"]["channels"]:
                if not channel["passed"]:
                    residuals.append(dict(
                        family=r["family"], case=r["case"], seed=r["seed"],
                        channel=channel["channel"], nrmse=channel["nrmse"],
                        event=channel["event"]))
    freezes = read(CURRENT / "selection_freeze.json")
    curves = read(CURRENT / "training_curves.json")
    first, last = curves[0], curves[-1]
    selection = dict(
        chosen_step_counts=dict(collections.Counter(freezes["chosen_steps"])),
        common_score_improved=sum(b < a for a, b in zip(
            first["common_development_score"], last["common_development_score"])),
        members=len(first["panel"]),
        worse_worst_panel_ratio=sum(
            b["rank"][1] > a["rank"][1] for a, b in zip(first["panel"], last["panel"])),
        example_initial_I0_seed17=first["panel"][0],
        example_final_I0_seed17=last["panel"][0],
        clipping_at_90000=next(x["clipping_counts"] for x in curves if x["step"] == 90000),
    )
    attribution = collections.defaultdict(lambda: dict(cases=0, passed=0))
    for r in read(CURRENT / "frozen_oracle_attribution.json"):
        k = f'{r["parent_arm"]}/{r["precision"]}/{r["intervention"]}'
        attribution[k]["cases"] += 1
        attribution[k]["passed"] += int(r["metrics"]["passed"])
    path = ROOT / "experiments/results/task37_playground_v2/development__slow_recovery.npz"
    with np.load(path) as z:
        inputs = {k: z[k].copy() for k in ("voltage", "bridge_ca", "initial_gates")}
    timing = []
    for r in read(CURRENT / "timing_audit.json"):
        if r["family"] != "slow_recovery" or r["precision"] != "float32":
            continue
        for channel in r["channels"]:
            if channel["event"].get("offset_error_ms", 0) <= .3:
                continue
            crossing = next(c for c in channel["crossings"] if c["kind"] == "offset")
            v = inputs["voltage"][:, r["case"]]
            changes = np.flatnonzero(np.diff(v))
            first_change = (int(changes[0])+1)*.1
            timing.append(dict(
                parent=r["parent"], seed=r["seed"], case=r["case"],
                channel=channel["channel"], crossing=crossing,
                offset_reference_ms=crossing["reference_index"]*.1,
                first_voltage_change_ms=first_change,
                imposed_voltage_at_crossing_mv=float(v[crossing["reference_index"]]),
                later_voltage_mv=float(v[-1]),
                reference_excursions=channel["reference_excursions"],
                prediction_excursions=channel["prediction_excursions"],
                allowable_local_current_error_linear_estimate=.3*abs(crossing["reference_slope"]),
            ))
    probes = []
    for r in read(CURRENT / "fixed_fit_gradient_probes.json"):
        if r["cohort"] != "plain" or r["step"] not in (0, 90000):
            continue
        norms = np.array(r["term_norms"])*np.array(r["weights"])
        cosines = np.array(r["cosines"])
        for name, idx in (("I0", 0), ("I3", 9), ("I4", 12), ("I7", 15)):
            probes.append(dict(
                step=r["step"], arm=name, terms=r["terms"],
                weighted_norms=norms[idx:idx+3].tolist(),
                variable_held_cosine=cosines[idx:idx+3, 0, 1].tolist(),
                u_logz_cosine=cosines[idx:idx+3, 2, 3].tolist(),
            ))
    result = dict(
        task="39i exploratory plateau diagnosis", db_access="read-only",
        training_performed=False, thresholds_changed=False,
        queries=queries, current_I3_residuals=residuals,
        frozen_I3_coefficient_probe=coefficient_probe(),
        selection_and_optimization=selection, frozen_H8_attribution=dict(attribution),
        frozen_H8_timing_localization=timing, fixed_fit_gradient_probes=probes,
        limits=[
            "Historical analogies do not establish the present cause.",
            "39i results are read from recovered artifacts, not yet registered in SQLite.",
            "H8 timing/oracles concern the parent, not newly trained I3 seed17.",
            "Development-informed exploratory analysis; no independent confirmation.",
            "Gradient probes are fixed sampled batches, not causal interventions.",
            "Local crossing linearization is diagnostic, not an altered event metric.",
        ],
    )
    (OUT / "memory_and_residual_queries.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+"\n",
        encoding="utf-8")
    print(json.dumps(dict(
        query_counts={q["name"]: len(q["rows"]) for q in queries},
        selection=selection, timing=timing, attribution=dict(attribution)),
        indent=2))


if __name__ == "__main__":
    main()
