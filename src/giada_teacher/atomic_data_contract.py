"""Atomic data contracts derived from the GIADA teacher taxonomy."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


COMMON_PRECISION = {
    "reference_dtype": "float64",
    "candidate_training_dtype": "float32",
    "float32_requires_equivalence_audit": True,
    "event_time_dtype": "float64",
}


FAMILY_CONTRACTS: dict[str, dict[str, Any]] = {
    "voltage_dependent_gate": {
        "state": ["gate occupancy x_t for every declared STATE gate"],
        "inputs": ["membrane-voltage control path V(tau), tau in [t,t+dt]"],
        "parameters": ["mechanism kinetic parameters", "temperature-dependent constants if active"],
        "targets": ["gate STATE at t+dt", "optional x_inf(V) and tau_x(V) privileged targets"],
        "step": {"macro_dt_ms": 1.0, "validation_micro_dt_ms": 0.025},
        "solver": "exponential/Rush-Larsen update when voltage is held; path-aware or jointly coupled update otherwise",
        "privileged_observables": ["x_inf", "tau_x", "gate derivative", "resulting conductance/current"],
        "sufficiency": "(x_t,V_t) is sufficient only for the voltage-clamp constant-V playground; coupled use requires V(tau) or a joint voltage-state solver.",
        "forbidden_inputs": ["teacher gate at t+dt", "teacher endpoint voltage V_t+dt in causal deployment"],
    },
    "calcium_dependent_gate": {
        "state": ["calcium-dependent gate occupancy x_t"],
        "inputs": ["intracellular-calcium control path cai(tau)", "membrane voltage for analytic current readout"],
        "parameters": ["calcium activation kinetics", "maximal conductance", "reversal potential"],
        "targets": ["gate STATE at t+dt", "analytic current/conductance as diagnostic target"],
        "step": {"macro_dt_ms": 1.0, "validation_micro_dt_ms": 0.025},
        "solver": "exponential gate update under held calcium; path-aware or jointly coupled update otherwise",
        "privileged_observables": ["x_inf(cai)", "tau_x(cai)", "potassium current"],
        "sufficiency": "Endpoint cai alone is not assumed sufficient when calcium changes materially inside the step.",
        "forbidden_inputs": ["teacher gate at t+dt", "future calcium samples unavailable to the deployed joint solver"],
    },
    "ion_concentration_dynamics": {
        "state": ["intracellular ion concentration at t"],
        "inputs": ["causal ionic-current path over the step"],
        "parameters": ["decay time constant", "conversion factor/depth", "resting or minimum concentration"],
        "targets": ["ion concentration at t+dt", "integrated ionic influx"],
        "step": {"macro_dt_ms": 1.0, "validation_micro_dt_ms": 0.025},
        "solver": "exponential decay with current forcing; integrate forcing on the native or audited micro-grid",
        "privileged_observables": ["current integral", "decay-only counterfactual", "source-only counterfactual"],
        "sufficiency": "Initial concentration plus the within-step current forcing and fixed parameters is the atomic Markov contract.",
        "forbidden_inputs": ["teacher concentration at t+dt"],
    },
    "event_driven_synapse": {
        "state": ["all synaptic decay/rise STATE", "short-term-plasticity NET_RECEIVE weights/state", "synapse RNG state"],
        "inputs": ["ordered presynaptic events with intra-step timestamps and weights", "postsynaptic voltage path for voltage-dependent current"],
        "parameters": ["rise/decay constants", "reversal potentials", "release/STP parameters", "NMDA block parameters when present"],
        "targets": ["synaptic state at t+dt", "release outcomes", "conductance/current microtrace", "RNG state at t+dt"],
        "step": {"macro_dt_ms": 1.0, "event_time_resolution": "float64 continuous intra-step", "validation_micro_dt_ms": 0.025},
        "solver": "exact ordered event jumps plus analytic/exponential inter-event decay",
        "privileged_observables": ["realized release", "AMPA/NMDA or GABAA/GABAB component currents", "pre/post-event state"],
        "sufficiency": "Event order, exact timestamps, complete synaptic/STP state and RNG state are required for replay.",
        "forbidden_inputs": ["future release outcome when training the release predictor", "unordered event counts", "teacher endpoint state"],
        "causal_interface_note": "U_realized may enter the membrane core only after an authentic causal synaptic front-end has generated it; it is not a legal input to that release decision itself.",
    },
    "prescribed_time_dependent_point_process": {
        "state": ["phase/time relative to onset when the source remains active; otherwise empty"],
        "inputs": ["absolute or relative time over the step", "onset/delay control"],
        "parameters": ["amplitude", "rise/decay or duration parameters", "reversal/current parameters"],
        "targets": ["current microtrace", "current at t+dt", "phase at t+dt"],
        "step": {"macro_dt_ms": 1.0, "validation_micro_dt_ms": 0.025},
        "solver": "analytic waveform evaluation",
        "privileged_observables": ["waveform phase", "analytic current"],
        "sufficiency": "The contract is Markov only if phase/time-since-onset is represented explicitly or derivable from the supplied clock and onset.",
        "forbidden_inputs": ["teacher current at t+dt"],
    },
    "passive_membrane": {
        "state": ["segment voltage V_t"],
        "inputs": ["total causal transmembrane-current path", "axial-current contribution or joint axial solve", "externally applied current path"],
        "parameters": ["segment capacitance", "area", "passive conductance", "passive reversal potential"],
        "targets": ["segment voltage V_t+dt", "capacitive and passive current balance"],
        "step": {"macro_dt_ms": 1.0, "validation_micro_dt_ms": 0.025},
        "solver": "implicit or structure-preserving membrane update; coupled with axial tree solve in deployment",
        "privileged_observables": ["total membrane source", "capacitive current", "current-balance residual"],
        "sufficiency": "V_t is sufficient only together with all causal current forcing and the axial boundary/coupling contract.",
        "forbidden_inputs": ["teacher V_t+dt", "teacher total current computed from future state"],
    },
    "axial_coupling": {
        "state": ["voltage vector on the complete instantiated segment tree"],
        "inputs": ["local membrane source for every segment over the step"],
        "parameters": ["parent topology", "segment geometry", "axial resistance", "capacitance/area discretization"],
        "targets": ["coupled voltage vector at t+dt", "axial currents", "tree-solve residual"],
        "step": {"macro_dt_ms": 1.0, "validation_micro_dt_ms": 0.025},
        "solver": "Hines/tree-structured implicit solve or numerically equivalent differentiable solver",
        "privileged_observables": ["axial current per edge", "linear-system residual", "condition diagnostics"],
        "sufficiency": "The complete voltage vector, full tree coefficients and causal local membrane sources are required; isolated segment transitions are insufficient.",
        "forbidden_inputs": ["teacher endpoint voltage", "future teacher axial currents"],
    },
}


def build_atomic_data_contract(inventory: dict[str, Any], classification: dict[str, Any]) -> dict[str, Any]:
    inventory_by_name = {row["mechanism"]["name"]: row for row in inventory["mechanisms"]}
    bindings = []
    for classified in classification["mechanisms"]:
        name = classified["name"]
        source = inventory_by_name.get(name)
        bindings.append({
            "mechanism_id": classified["id"],
            "mechanism": name,
            "family": classified["primary_family"],
            "declared_state": [x["name"] for x in source["declarations"]["state"]] if source else [],
            "declared_parameters": [x["name"] for x in source["declarations"]["parameter"]] if source else [],
            "currents": source["mechanism"]["currents"] if source else [],
            "contract_ref": f"family:{classified['primary_family']}",
        })
    result = {
        "schema_version": "giada-atomic-data-contract-v1",
        "task": "0.3",
        "teacher_commit": inventory["teacher_commit"],
        "inventory_source_sha256": inventory["combined_source_sha256"],
        "classification_artifact_sha256": hashlib.sha256(
            json.dumps(classification, sort_keys=True, ensure_ascii=False).encode("utf-8")
        ).hexdigest(),
        "global_contract": {
            "canonical_macro_step_ms": 1.0,
            "event_ordering": "stable total order by absolute timestamp then source order",
            "boundary_state_policy": "store causal S_t and S_t+1; privileged observables are targets/diagnostics, never silent deployment inputs",
            "control_path_policy": "constant-control playgrounds and coupled path-aware deployment are distinct contracts",
            "precision": COMMON_PRECISION,
        },
        "family_contracts": {f"family:{key}": value for key, value in FAMILY_CONTRACTS.items()},
        "mechanism_bindings": bindings,
    }
    result["validation"] = validate_atomic_data_contract(result, classification)
    if not result["validation"]["valid"]:
        raise ValueError(result["validation"])
    return result


def validate_atomic_data_contract(contract: dict[str, Any], classification: dict[str, Any]) -> dict[str, Any]:
    failures: list[str] = []
    families = contract["family_contracts"]
    for family_id, row in families.items():
        for required in ("state", "inputs", "parameters", "targets", "step", "solver", "privileged_observables", "sufficiency", "forbidden_inputs"):
            if required not in row or row[required] in (None, "", []):
                failures.append(f"{family_id}: missing {required}")
    classified_ids = {row["id"] for row in classification["mechanisms"]}
    bound_ids = {row["mechanism_id"] for row in contract["mechanism_bindings"]}
    if classified_ids != bound_ids:
        failures.append("mechanism binding coverage differs from Task 0.2")
    used_families = {f"family:{row['primary_family']}" for row in classification["mechanisms"]}
    if used_families != set(families):
        failures.append("family contract coverage differs from Task 0.2")
    voltage = families["family:voltage_dependent_gate"]
    if "only" not in voltage["sufficiency"] or "path" not in voltage["solver"]:
        failures.append("voltage-gate contract fails to distinguish clamp and coupled modes")
    synapse = families["family:event_driven_synapse"]
    if "RNG" not in " ".join(synapse["state"]) or "unordered event counts" not in synapse["forbidden_inputs"]:
        failures.append("event-driven contract omits RNG or ordered-event requirement")
    if contract["global_contract"]["canonical_macro_step_ms"] != 1.0:
        failures.append("canonical macro-step is not 1 ms")
    return {
        "valid": not failures,
        "failures": failures,
        "family_count": len(families),
        "mechanism_binding_count": len(bound_ids),
        "classification_mechanism_count": len(classified_ids),
    }


def write_atomic_data_contract(contract: dict[str, Any], json_path: str | Path, markdown_path: str | Path) -> None:
    json_path, markdown_path = Path(json_path), Path(markdown_path)
    json_path.write_text(json.dumps(contract, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = [
        "# GIADA atomic data contract", "",
        f"- Teacher commit: `{contract['teacher_commit']}`",
        f"- Canonical macro-step: **{contract['global_contract']['canonical_macro_step_ms']} ms**",
        f"- Families: **{contract['validation']['family_count']}**",
        f"- Mechanism bindings: **{contract['validation']['mechanism_binding_count']}**", "",
        "| Family | Causal state | Inputs | Solver | Sufficiency boundary |",
        "|---|---|---|---|---|",
    ]
    for family_id, row in contract["family_contracts"].items():
        lines.append(
            f"| `{family_id.removeprefix('family:')}` | {'; '.join(row['state'])} | "
            f"{'; '.join(row['inputs'])} | {row['solver']} | {row['sufficiency']} |"
        )
    lines += [
        "", "## Global anti-leakage rule", "",
        "Teacher endpoint values and privileged observables are labels or diagnostics. They are not deployment inputs unless an upstream causal component has generated them before the downstream update.", "",
        "## Voltage-path distinction", "",
        "A constant-voltage atomic playground may use the exponential gate solution from `(x_t, V_t)`. The coupled neuron may not silently reuse that assumption: it must receive an intra-step control path or solve voltage and internal state jointly.", "",
    ]
    markdown_path.write_text("\n".join(lines), encoding="utf-8")
