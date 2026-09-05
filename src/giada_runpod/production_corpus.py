"""Prospective S1e hybrid-production composition and support gates."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Mapping

from .corpus_audit import audit_soma_corpus
from .hybrid_inputs import (
    PRODUCTION_BACKGROUND_PROTOCOLS,
    PRODUCTION_TARGET_PROTOCOLS,
)


SCHEMA_VERSION = "giada-runpod-composite-corpus-v1"
EXPECTED_COMPONENTS = {
    "background": {
        "stage": "s1e_hybrid_background",
        "purpose": "giada_hybrid_production_background",
        "transition_count": 360_000,
    },
    "targeted": {
        "stage": "s1e_hybrid_targeted",
        "purpose": "giada_hybrid_production_targeted",
        "transition_count": 240_000,
    },
}
EXPECTED_SPLIT_TRANSITIONS = {"train": 480_000, "validation": 120_000}


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(dict(payload), indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(destination)


def _component_check(component_id: str, root: Path) -> tuple[list[str], Dict[str, Any]]:
    expected = EXPECTED_COMPONENTS[component_id]
    blockers = []
    plan_path = root / "plan.json"
    validation_path = root / "validation_report.json"
    if not plan_path.is_file():
        return [f"{component_id}: plan.json missing"], {}
    if not validation_path.is_file():
        return [f"{component_id}: validation_report.json missing"], {}
    plan = _read_json(plan_path)
    validation = _read_json(validation_path)
    config = plan.get("config", {})
    if config.get("stage") != expected["stage"]:
        blockers.append(f"{component_id}: wrong stage")
    if config.get("purpose") != expected["purpose"]:
        blockers.append(f"{component_id}: wrong generation purpose")
    if int(config.get("target_transitions", -1)) != expected["transition_count"]:
        blockers.append(f"{component_id}: wrong planned transition count")
    if not validation.get("valid"):
        blockers.append(f"{component_id}: structural validation failed")
    if int(validation.get("validated_transition_count", -1)) != expected["transition_count"]:
        blockers.append(f"{component_id}: validated transition count mismatch")
    return blockers, {
        "stage": config.get("stage"),
        "purpose": config.get("purpose"),
        "transition_count": expected["transition_count"],
        "validated_shard_count": validation.get("validated_shard_count"),
    }


def build_and_audit_s1e_composite(
    background_root: Path,
    targeted_root: Path,
    output_root: Path,
    *,
    progress=None,
) -> Dict[str, Any]:
    """Seal the two validated S1e components as one logical training corpus."""

    roots = {
        "background": Path(background_root).resolve(),
        "targeted": Path(targeted_root).resolve(),
    }
    output = Path(output_root).resolve()
    output.mkdir(parents=True, exist_ok=True)
    blockers = []
    component_reports = {}
    for component_id, root in roots.items():
        failures, report = _component_check(component_id, root)
        blockers.extend(failures)
        component_reports[component_id] = report
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "project": "GIADA",
        "stage": "s1e_hybrid_production",
        "valid": False,
        "total_transition_count": 600_000,
        "split_transition_counts": EXPECTED_SPLIT_TRANSITIONS,
        "components": [
            {
                "component_id": component_id,
                "root": os.path.relpath(root, output),
                "plan": "plan.json",
                **component_reports.get(component_id, {}),
            }
            for component_id, root in roots.items()
        ],
        "physical_merge_performed": False,
        "selection_role": "development_validation",
        "paper_test_claimed": False,
    }
    _atomic_json(output / "composite_manifest.json", manifest)
    audit = audit_soma_corpus(output, progress=progress)
    blockers.extend(audit.get("blockers", []))

    checks = []
    for split, expected_count in EXPECTED_SPLIT_TRANSITIONS.items():
        observed = int(audit.get("splits", {}).get(split, {}).get("transition_count", -1))
        checks.append({
            "gate": f"{split}_transition_count",
            "observed": observed,
            "required": expected_count,
            "passed": observed == expected_count,
        })
    expected_protocol_rows = {
        **{
            protocol: {"train": 144_000, "validation": 36_000}
            for protocol in PRODUCTION_BACKGROUND_PROTOCOLS
        },
        **{
            protocol: {"train": 16_000, "validation": 4_000}
            for protocol in PRODUCTION_TARGET_PROTOCOLS
        },
    }
    protocol_splits = audit.get("protocol_splits", {})
    for protocol, split_counts in expected_protocol_rows.items():
        for split, required in split_counts.items():
            observed = int(
                protocol_splits.get(protocol, {}).get(split, {}).get(
                    "transition_count", -1
                )
            )
            checks.append({
                "gate": "protocol_split_transition_count",
                "protocol": protocol,
                "split": split,
                "observed": observed,
                "required": required,
                "passed": observed == required,
            })
    support_gates = (
        ("train", "absolute_delta_ge_5mv_count", 1024),
        ("validation", "absolute_delta_ge_5mv_count", 256),
        ("train", "somatic_upcrossings_minus55mv", 512),
        ("validation", "somatic_upcrossings_minus55mv", 128),
    )
    for split, metric, required in support_gates:
        observed = int(audit.get("splits", {}).get(split, {}).get(metric, -1))
        checks.append({
            "gate": metric,
            "split": split,
            "observed": observed,
            "required": required,
            "passed": observed >= required,
        })
    failed_checks = [row for row in checks if not row["passed"]]
    blockers.extend(f"failed gate: {row}" for row in failed_checks)
    report = {
        "schema_version": "giada-runpod-s1e-production-audit-v1",
        "valid": not blockers,
        "blockers": blockers,
        "component_reports": component_reports,
        "composition": {
            "total_transition_count": 600_000,
            "long_stochastic_background_fraction": 0.6,
            "confirmed_targeted_fraction": 0.4,
            "train_fraction": 0.8,
            "validation_fraction": 0.2,
        },
        "support_checks": checks,
        "corpus_audit": audit,
        "architecture_experiments_modified": False,
        "paper_test_claimed": False,
    }
    _atomic_json(output / "production_audit.json", report)
    manifest["valid"] = report["valid"]
    manifest["production_audit"] = "production_audit.json"
    _atomic_json(output / "composite_manifest.json", manifest)
    return report
