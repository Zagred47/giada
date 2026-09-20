"""Leakage-resistant domain split contract for atomic Ca_HVA gate studies."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from itertools import product
from typing import Any, Iterable


def _inclusive(start: float, stop: float, step: float) -> tuple[float, ...]:
    count = round((stop - start) / step)
    return tuple(round(start + index * step, 10) for index in range(count + 1))


def _case_key(gate: str, voltage_mv: float, state: float, dt_ms: float) -> str:
    return f"{gate}|{voltage_mv:.10g}|{state:.10g}|{dt_ms:.10g}"


def _fingerprint(cases: Iterable[tuple[str, float, float, float]]) -> str:
    digest = hashlib.sha256()
    for case in sorted(_case_key(*row) for row in cases):
        digest.update(case.encode("ascii") + b"\n")
    return digest.hexdigest()


@dataclass(frozen=True)
class AtomicDomainSplitConfig:
    gates: tuple[str, ...] = ("m", "h")
    train_voltage_range_mv: tuple[float, float] = (-100.0, 40.0)
    train_voltage_step_mv: float = 1.0
    train_states: tuple[float, ...] = (
        0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95
    )
    train_dt_ms: tuple[float, ...] = (0.025, 0.1, 0.5, 1.0)
    interpolation_voltage_windows_development: tuple[tuple[float, float], ...] = (
        (-85.0, -79.0), (-15.0, -9.0)
    )
    interpolation_voltage_windows_test: tuple[tuple[float, float], ...] = (
        (-60.0, -54.0), (15.0, 21.0)
    )
    voltage_window_guard_mv: float = 2.0
    interpolation_states_development: tuple[float, ...] = (0.15, 0.65)
    interpolation_states_test: tuple[float, ...] = (0.35, 0.85)
    interpolation_dt_development_ms: tuple[float, ...] = (0.05, 0.25)
    interpolation_dt_test_ms: tuple[float, ...] = (0.75,)
    anchor_voltages_mv: tuple[float, ...] = (-90.0, -70.0, -40.0, 0.0, 30.0)
    singular_voltages_mv: tuple[float, ...] = (-27.0001, -27.0, -26.9999)
    singular_guard_range_mv: tuple[float, float] = (-28.0, -26.0)
    boundary_states: tuple[float, ...] = (0.0, 0.01, 0.99, 1.0)
    support_edge_voltages_mv: tuple[float, ...] = (-100.0, 40.0)
    ood_low_voltages_mv: tuple[float, ...] = (-120.0, -115.0, -110.0, -105.0)
    ood_high_voltages_mv: tuple[float, ...] = (45.0, 50.0, 55.0, 60.0)
    ood_dt_ms: tuple[float, ...] = (1.5, 2.0)
    stress_states: tuple[float, ...] = (0.0, 1.0)
    stress_voltages_mv: tuple[float, ...] = (-120.0, 60.0)
    stress_dt_ms: tuple[float, ...] = (2.0,)


def _cartesian(config, voltages, states, dt_values):
    return {
        (gate, float(voltage), float(state), float(dt))
        for gate, voltage, state, dt in product(config.gates, voltages, states, dt_values)
    }


def build_atomic_domain_splits(config: AtomicDomainSplitConfig | None = None) -> dict[str, Any]:
    config = config or AtomicDomainSplitConfig()
    full_train_voltage_grid = _inclusive(
        config.train_voltage_range_mv[0], config.train_voltage_range_mv[1], config.train_voltage_step_mv
    )
    reserved_windows = (
        config.interpolation_voltage_windows_development + config.interpolation_voltage_windows_test
    )

    def reserved_for_training(voltage: float) -> bool:
        return (
            voltage in config.support_edge_voltages_mv
            or config.singular_guard_range_mv[0] <= voltage <= config.singular_guard_range_mv[1]
            or any(
                low - config.voltage_window_guard_mv <= voltage <= high + config.voltage_window_guard_mv
                for low, high in reserved_windows
            )
        )

    train_voltages = tuple(v for v in full_train_voltage_grid if not reserved_for_training(v))

    def window_points(windows):
        return tuple(
            voltage for low, high in windows
            for voltage in _inclusive(low, high, config.train_voltage_step_mv)
        )

    strata: dict[str, dict[str, Any]] = {}

    def add(name, role, axis, voltages, states, dt_values, purpose):
        cases = _cartesian(config, voltages, states, dt_values)
        strata[name] = {
            "role": role,
            "varied_axis": axis,
            "purpose": purpose,
            "voltages_mv": list(voltages),
            "states": list(states),
            "dt_ms": list(dt_values),
            "case_count": len(cases),
            "case_fingerprint_sha256": _fingerprint(cases),
            "_cases": cases,
        }

    add("train", "fit", "support", train_voltages, config.train_states, config.train_dt_ms,
        "Fit only on non-reserved support; no adjacent points from held voltage windows.")
    add("interpolation_voltage_development", "development", "voltage",
        window_points(config.interpolation_voltage_windows_development), config.train_states,
        config.train_dt_ms, "Model selection on complete contiguous voltage windows inside the fit envelope.")
    add("interpolation_voltage_test", "sealed_test", "voltage",
        window_points(config.interpolation_voltage_windows_test), config.train_states,
        config.train_dt_ms, "Sealed interpolation confirmation on different contiguous voltage windows.")
    add("interpolation_state_development", "development", "state", config.anchor_voltages_mv,
        config.interpolation_states_development, config.train_dt_ms,
        "State interpolation with voltage and dt held on fit support.")
    add("interpolation_state_test", "sealed_test", "state", config.anchor_voltages_mv,
        config.interpolation_states_test, config.train_dt_ms,
        "Sealed state interpolation at disjoint occupancy values.")
    add("interpolation_dt_development", "development", "dt", config.anchor_voltages_mv,
        config.train_states, config.interpolation_dt_development_ms,
        "Variable-step development with voltage and state on fit support.")
    add("interpolation_dt_test", "sealed_test", "dt", config.anchor_voltages_mv,
        config.train_states, config.interpolation_dt_test_ms,
        "Sealed variable-step interpolation at a disjoint time step.")
    add("boundary_singularity_test", "sealed_test", "voltage_boundary",
        config.singular_voltages_mv, config.train_states, config.train_dt_ms,
        "Exact and one-sided probes of the teacher's -27 mV singularity handling.")
    add("boundary_state_test", "sealed_test", "state_boundary", config.anchor_voltages_mv,
        config.boundary_states, config.train_dt_ms,
        "Physical occupancy boundaries and their near-boundary controls.")
    add("boundary_support_edge_test", "sealed_test", "support_edge",
        config.support_edge_voltages_mv, config.train_states, config.train_dt_ms,
        "Edges of the declared in-support voltage envelope.")
    add("ood_voltage_test", "sealed_test", "voltage_ood",
        config.ood_low_voltages_mv + config.ood_high_voltages_mv, config.train_states,
        config.train_dt_ms, "Voltage extrapolation with state and dt held in support.")
    add("ood_dt_test", "sealed_test", "dt_ood", config.anchor_voltages_mv,
        config.train_states, config.ood_dt_ms,
        "Time-step extrapolation with voltage and state held in support.")
    add("factorial_stress_test", "sealed_test", "multi_axis_ood", config.stress_voltages_mv,
        config.stress_states, config.stress_dt_ms,
        "Deliberately confounded stress test, reported separately from single-axis OOD.")

    owners: dict[tuple[str, float, float, float], list[str]] = {}
    for name, row in strata.items():
        for case in row["_cases"]:
            owners.setdefault(case, []).append(name)
    overlaps = [
        {"case": _case_key(*case), "strata": names}
        for case, names in owners.items() if len(names) > 1
    ]
    fit_cases = strata["train"]["_cases"]
    evaluation_cases = set().union(*(row["_cases"] for name, row in strata.items() if name != "train"))
    fit_evaluation_overlap = fit_cases & evaluation_cases
    for row in strata.values():
        row.pop("_cases")

    report = {
        "schema_version": "giada-atomic-domain-splits-v1",
        "task": "0.5",
        "mechanism": "Ca_HVA",
        "teacher_commit": "074c4666300a8ad246601dab179a97a6942f0f29",
        "unit_of_split": "complete domain stratum, never random adjacent points",
        "roles": {
            "fit": "optimization only",
            "development": "model selection and debugging allowed",
            "sealed_test": "single final evaluation; no selection or threshold tuning",
            "embedded_confirmation": "future independent coupled-teacher protocols; contains no atomic rows",
        },
        "strata": strata,
        "excluded_guards": {
            "voltage_window_guard_mv": config.voltage_window_guard_mv,
            "singularity_guard_range_mv": list(config.singular_guard_range_mv),
            "train_to_ood_voltage_gap_mv": [[-104.0, -101.0], [41.0, 44.0]],
            "policy": "Guard points are not fit data and are not scored; they prevent near-duplicate support leakage.",
        },
        "embedded_confirmation": {
            "atomic_rows": 0,
            "status": "reserved_not_generated",
            "required_later": [
                "held-out voltage-path protocols",
                "independent initial states or snapshots",
                "teacher-coupled gate and current readout",
                "no model selection on outcomes",
            ],
        },
        "validation": {
            "valid": not overlaps and not fit_evaluation_overlap,
            "stratum_count": len(strata),
            "total_case_count": sum(row["case_count"] for row in strata.values()),
            "overlap_count": len(overlaps),
            "fit_evaluation_overlap_count": len(fit_evaluation_overlap),
            "overlaps": overlaps[:20],
            "all_strata_nonempty": all(row["case_count"] > 0 for row in strata.values()),
            "development_and_test_are_disjoint": not overlaps,
        },
    }
    if not report["validation"]["valid"] or not report["validation"]["all_strata_nonempty"]:
        raise ValueError(f"invalid atomic domain split contract: {report['validation']}")
    return report


def render_atomic_domain_splits_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# GIADA Task 0.5 — atomic domain splits", "",
        "No point-wise random split is used. Complete voltage windows and complete state/`dt` values are held out.", "",
        "| Stratum | Role | Isolated axis | Cases |", "|---|---|---|---:|",
    ]
    for name, row in report["strata"].items():
        lines.append(f"| `{name}` | `{row['role']}` | `{row['varied_axis']}` | {row['case_count']:,} |")
    lines += [
        "", "## Leakage audit", "",
        f"- Total cases: **{report['validation']['total_case_count']:,}**",
        f"- Cross-stratum duplicate cases: **{report['validation']['overlap_count']}**",
        f"- Fit/evaluation overlap: **{report['validation']['fit_evaluation_overlap_count']}**",
        "- Guard bands adjacent to held-out voltage windows are excluded from both fitting and scoring.",
        "- Single-axis OOD tests keep the other axes in support; the multi-axis stress test is reported separately.",
        "", "## Embedded confirmation", "",
        "No atomic case is relabelled as embedded evidence. Coupled-teacher confirmation remains a future, independently generated and sealed protocol.", "",
    ]
    return "\n".join(lines)
