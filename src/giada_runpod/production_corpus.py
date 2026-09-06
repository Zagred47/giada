"""Prospective hybrid-production composition and distribution gates."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, Mapping

from .corpus_audit import audit_soma_corpus
from .hybrid_inputs import PRODUCTION_BACKGROUND_PROTOCOLS, PRODUCTION_TARGET_PROTOCOLS


SCHEMA_VERSION = "giada-runpod-composite-corpus-v1"

# Each scale is a sealed expansion of the same scientific mixture. Component
# roots stay separate because background and targeted trajectories have
# different natural durations.
PRODUCTION_PROFILES: Dict[str, Dict[str, Any]] = {
    "s1e": {
        "composite_stage": "s1e_hybrid_production",
        "audit_schema": "giada-runpod-s1e-production-audit-v1",
        "total": 600_000,
        "splits": {"train": 480_000, "validation": 120_000},
        "components": {
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
        },
        "protocol_splits": {
            **{
                protocol: {"train": 144_000, "validation": 36_000}
                for protocol in PRODUCTION_BACKGROUND_PROTOCOLS
            },
            **{
                protocol: {"train": 16_000, "validation": 4_000}
                for protocol in PRODUCTION_TARGET_PROTOCOLS
            },
        },
        "support_ranges": {
            ("train", "absolute_delta_ge_5mv_count"): (1_024, None),
            ("validation", "absolute_delta_ge_5mv_count"): (256, None),
            ("train", "somatic_upcrossings_minus55mv"): (512, None),
            ("validation", "somatic_upcrossings_minus55mv"): (128, None),
        },
    },
    "s2": {
        "composite_stage": "s2_hybrid_production",
        "audit_schema": "giada-runpod-s2-production-audit-v1",
        "total": 3_600_000,
        "splits": {"train": 2_880_000, "validation": 720_000},
        "components": {
            "background": {
                "stage": "s2_hybrid_background",
                "purpose": "giada_hybrid_production_background",
                "transition_count": 2_160_000,
            },
            "targeted": {
                "stage": "s2_hybrid_targeted",
                "purpose": "giada_hybrid_production_targeted",
                "transition_count": 1_440_000,
            },
        },
        "protocol_splits": {
            **{
                protocol: {"train": 864_000, "validation": 216_000}
                for protocol in PRODUCTION_BACKGROUND_PROTOCOLS
            },
            **{
                protocol: {"train": 96_000, "validation": 24_000}
                for protocol in PRODUCTION_TARGET_PROTOCOLS
            },
        },
        # Prospective 75--125% fidelity bands around the sixfold-scaled S1e
        # observations. Both a collapse and a major shift block training.
        "support_ranges": {
            ("train", "absolute_delta_ge_5mv_count"): (49_671, 82_785),
            ("validation", "absolute_delta_ge_5mv_count"): (12_303, 20_505),
            ("train", "somatic_upcrossings_minus55mv"): (18_279, 30_465),
            ("validation", "somatic_upcrossings_minus55mv"): (4_518, 7_530),
        },
    },
}


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(dict(payload), indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(destination)


def _sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def fingerprint_validated_shards(composite_root: Path) -> Dict[str, Any]:
    """Fingerprint every completion marker and verify its physical HDF5."""

    root = Path(composite_root).resolve()
    manifest = _read_json(root / "composite_manifest.json")
    components = {
        str(row["component_id"]): (root / str(row["root"])).resolve()
        for row in manifest["components"]
    }
    aggregate = hashlib.sha256()
    mismatches = []
    shard_count = 0
    total_size_bytes = 0
    for component_id in sorted(components):
        component_root = components[component_id]
        for marker_path in sorted((component_root / "status").glob("*.done.json")):
            relative = f"{component_id}/status/{marker_path.name}"
            marker_sha256 = _sha256_file(marker_path)
            aggregate.update(f"{marker_sha256}  {relative}\n".encode("utf-8"))
            marker = _read_json(marker_path)
            shard_id = str(marker.get("shard_id", marker_path.name.removesuffix(".done.json")))
            shard_path = component_root / "shards" / f"{shard_id}.h5"
            if not shard_path.is_file():
                mismatches.append(f"missing physical shard {component_id}/{shard_id}")
                continue
            physical_sha256 = _sha256_file(shard_path)
            if physical_sha256 != marker.get("sha256"):
                mismatches.append(f"physical SHA-256 mismatch {component_id}/{shard_id}")
            shard_count += 1
            total_size_bytes += shard_path.stat().st_size
    return {
        "schema_version": "giada-runpod-corpus-fingerprint-v1",
        "valid": not mismatches,
        "marker_fingerprint_sha256": aggregate.hexdigest(),
        "shard_count": shard_count,
        "total_size_bytes": total_size_bytes,
        "physical_mismatch_count": len(mismatches),
        "physical_mismatch_examples": mismatches[:10],
    }


def _component_check(
    component_id: str, root: Path, expected: Mapping[str, Any]
) -> tuple[list[str], Dict[str, Any]]:
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


def build_and_audit_hybrid_composite(
    background_root: Path,
    targeted_root: Path,
    output_root: Path,
    *,
    scale: str,
    progress=None,
) -> Dict[str, Any]:
    """Seal two validated components and enforce a scale-specific contract."""

    if scale not in PRODUCTION_PROFILES:
        raise ValueError(f"unknown hybrid production scale {scale!r}")
    profile = PRODUCTION_PROFILES[scale]
    roots = {
        "background": Path(background_root).resolve(),
        "targeted": Path(targeted_root).resolve(),
    }
    output = Path(output_root).resolve()
    output.mkdir(parents=True, exist_ok=True)
    blockers: list[str] = []
    component_reports = {}
    for component_id, root in roots.items():
        failures, report = _component_check(
            component_id, root, profile["components"][component_id]
        )
        blockers.extend(failures)
        component_reports[component_id] = report
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "project": "GIADA",
        "stage": profile["composite_stage"],
        "valid": False,
        "total_transition_count": profile["total"],
        "split_transition_counts": profile["splits"],
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
    for split, required in profile["splits"].items():
        observed = int(audit.get("splits", {}).get(split, {}).get("transition_count", -1))
        checks.append({
            "gate": f"{split}_transition_count",
            "observed": observed,
            "required": required,
            "passed": observed == required,
        })
    protocol_splits = audit.get("protocol_splits", {})
    for protocol, split_counts in profile["protocol_splits"].items():
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
    for (split, metric), (minimum, maximum) in profile["support_ranges"].items():
        observed = int(audit.get("splits", {}).get(split, {}).get(metric, -1))
        passed = observed >= minimum and (maximum is None or observed <= maximum)
        checks.append({
            "gate": metric,
            "split": split,
            "observed": observed,
            "minimum": minimum,
            "maximum": maximum,
            "passed": passed,
        })
    failed_checks = [row for row in checks if not row["passed"]]
    blockers.extend(f"failed gate: {row}" for row in failed_checks)
    report = {
        "schema_version": profile["audit_schema"],
        "valid": not blockers,
        "blockers": blockers,
        "scale": scale,
        "component_reports": component_reports,
        "composition": {
            "total_transition_count": profile["total"],
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


def build_and_audit_s1e_composite(
    background_root: Path,
    targeted_root: Path,
    output_root: Path,
    *,
    progress=None,
) -> Dict[str, Any]:
    """Backward-compatible S1e entry point."""

    return build_and_audit_hybrid_composite(
        background_root, targeted_root, output_root, scale="s1e", progress=progress
    )


def build_and_audit_s2_composite(
    background_root: Path,
    targeted_root: Path,
    output_root: Path,
    *,
    progress=None,
) -> Dict[str, Any]:
    """Seal the preregistered 3.6-million-transition S2 corpus."""

    return build_and_audit_hybrid_composite(
        background_root, targeted_root, output_root, scale="s2", progress=progress
    )
