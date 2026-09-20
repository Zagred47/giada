"""Independent formula-vs-NEURON oracle for atomic NMODL gate dynamics."""

from __future__ import annotations

import ast
import hashlib
import json
import math
import platform
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


DEFAULT_VOLTAGES_MV = tuple(float(v) for v in range(-120, 61)) + (-27.0001, -27.0, -26.9999)
DEFAULT_STATES = (0.0, 0.1, 0.5, 0.9, 1.0)
DEFAULT_DT_MS = (0.025, 0.1, 1.0, 2.0)
DEFAULT_ATOL = 1e-10
DEFAULT_RTOL = 1e-10


class OracleUnavailable(RuntimeError):
    """Raised when the authentic NEURON oracle cannot be executed."""


_ALLOWED_FUNCTIONS = {"exp": math.exp}
_ALLOWED_BINOPS = {
    ast.Add: lambda a, b: a + b,
    ast.Sub: lambda a, b: a - b,
    ast.Mult: lambda a, b: a * b,
    ast.Div: lambda a, b: a / b,
    ast.Pow: lambda a, b: a**b,
}
_ALLOWED_UNARY = {ast.UAdd: lambda a: a, ast.USub: lambda a: -a}


def _safe_eval(expression: str, variables: dict[str, float]) -> float:
    """Evaluate the arithmetic subset used by the teacher rate procedures."""

    node = ast.parse(expression.replace("^", "**"), mode="eval")

    def visit(item: ast.AST) -> float:
        if isinstance(item, ast.Expression):
            return visit(item.body)
        if isinstance(item, ast.Constant) and isinstance(item.value, (int, float)):
            return float(item.value)
        if isinstance(item, ast.Name) and item.id in variables:
            return float(variables[item.id])
        if isinstance(item, ast.BinOp) and type(item.op) in _ALLOWED_BINOPS:
            return float(_ALLOWED_BINOPS[type(item.op)](visit(item.left), visit(item.right)))
        if isinstance(item, ast.UnaryOp) and type(item.op) in _ALLOWED_UNARY:
            return float(_ALLOWED_UNARY[type(item.op)](visit(item.operand)))
        if isinstance(item, ast.Call) and isinstance(item.func, ast.Name):
            function = _ALLOWED_FUNCTIONS.get(item.func.id)
            if function is not None and len(item.args) == 1 and not item.keywords:
                return float(function(visit(item.args[0])))
        raise ValueError(f"unsupported NMODL arithmetic: {ast.dump(item)}")

    return visit(node)


def _procedure_body(text: str, name: str) -> str:
    match = re.search(rf"\bPROCEDURE\s+{re.escape(name)}\s*\([^)]*\)\s*\{{", text, re.I)
    if not match:
        raise ValueError(f"PROCEDURE {name} not found")
    depth = 1
    index = match.end()
    start = index
    while index < len(text) and depth:
        depth += (text[index] == "{") - (text[index] == "}")
        index += 1
    if depth:
        raise ValueError(f"unterminated PROCEDURE {name}")
    return text[start : index - 1]


@dataclass(frozen=True)
class ExtractedGateFormula:
    """Sequential assignments extracted from an NMODL rates procedure."""

    source_path: Path
    source_sha256: str
    procedure: str
    assignments: tuple[tuple[str, str], ...]
    singular_voltage_mv: float | None = None
    singular_perturbation_mv: float = 0.0

    @classmethod
    def from_mod(cls, path: str | Path, procedure: str = "rates") -> "ExtractedGateFormula":
        path = Path(path)
        raw = path.read_bytes()
        text = raw.decode("utf-8")
        body = _procedure_body(text, procedure)
        assignments = []
        for line in body.splitlines():
            line = line.split(":", 1)[0].strip()
            match = re.fullmatch(r"([A-Za-z_]\w*)\s*=\s*(.+?)\s*", line)
            if match and match.group(1) != "v":
                assignments.append((match.group(1), match.group(2)))
        required = {"mInf", "mTau", "hInf", "hTau"}
        if not required.issubset({name for name, _ in assignments}):
            raise ValueError(f"rates procedure lacks {sorted(required)}")
        singular = re.search(
            r"if\s*\(\s*\(\s*v\s*==\s*([-+0-9.eE]+)\s*\)\s*\)\s*\{\s*v\s*=\s*v\s*\+\s*([-+0-9.eE]+)",
            body,
            re.I | re.S,
        )
        return cls(
            source_path=path.resolve(),
            source_sha256=hashlib.sha256(raw).hexdigest(),
            procedure=procedure,
            assignments=tuple(assignments),
            singular_voltage_mv=float(singular.group(1)) if singular else None,
            singular_perturbation_mv=float(singular.group(2)) if singular else 0.0,
        )

    def rates(self, voltage_mv: float) -> dict[str, float]:
        effective_voltage = float(voltage_mv)
        if self.singular_voltage_mv is not None and effective_voltage == self.singular_voltage_mv:
            effective_voltage += self.singular_perturbation_mv
        values = {"v": effective_voltage}
        for name, expression in self.assignments:
            values[name] = _safe_eval(expression, values)
        return {
            "effective_voltage_mv": effective_voltage,
            "m_inf": values["mInf"],
            "m_tau_ms": values["mTau"],
            "h_inf": values["hInf"],
            "h_tau_ms": values["hTau"],
        }

    def step(self, gate: str, state: float, voltage_mv: float, dt_ms: float) -> float:
        if gate not in {"m", "h"}:
            raise ValueError("gate must be m or h")
        if dt_ms <= 0:
            raise ValueError("dt_ms must be positive")
        rates = self.rates(voltage_mv)
        inf = rates[f"{gate}_inf"]
        tau = rates[f"{gate}_tau_ms"]
        return float(inf + (float(state) - inf) * math.exp(-float(dt_ms) / tau))


class NeuronIsolatedGateOracle:
    """Authentic compiled NMODL state update, without membrane integration."""

    def __init__(self, mod_directory: str | Path, suffix: str = "Ca_HVA") -> None:
        try:
            from neuron import h, load_mechanisms  # type: ignore
        except ImportError as exc:
            raise OracleUnavailable(
                "NEURON is not installed. Run this oracle in the pinned Linux/Kaggle environment."
            ) from exc
        self.h = h
        self.mod_directory = Path(mod_directory).resolve()
        self.suffix = suffix
        if not self.mod_directory.is_dir():
            raise FileNotFoundError(self.mod_directory)
        if not load_mechanisms(str(self.mod_directory)):
            raise OracleUnavailable(
                f"compiled mechanisms were not found under {self.mod_directory}; run nrnivmodl first"
            )
        self.section = h.Section(name="giada_double_oracle")
        self.section.L = self.section.diam = 10.0
        self.section.nseg = 1
        self.section.insert(suffix)
        self.segment = self.section(0.5)
        self._state_procedure = getattr(h, f"states_{suffix}", None)
        if self._state_procedure is None:
            raise OracleUnavailable(f"compiled mechanism does not expose states_{suffix}")

    def step(self, gate: str, state: float, voltage_mv: float, dt_ms: float) -> float:
        if gate not in {"m", "h"}:
            raise ValueError("gate must be m or h")
        self.h.dt = float(dt_ms)
        self.h.finitialize(float(voltage_mv))
        self.segment.v = float(voltage_mv)
        setattr(self.segment, f"{gate}_{self.suffix}", float(state))
        self._state_procedure(sec=self.section)
        return float(getattr(self.segment, f"{gate}_{self.suffix}"))


def compile_nmodl(mod_file: str | Path, build_directory: str | Path) -> Path:
    """Compile exactly one copied MOD file and return the mechanism root."""

    mod_file, build_directory = Path(mod_file).resolve(), Path(build_directory).resolve()
    build_directory.mkdir(parents=True, exist_ok=True)
    copied = build_directory / mod_file.name
    if not copied.exists() or copied.read_bytes() != mod_file.read_bytes():
        shutil.copy2(mod_file, copied)
    command = shutil.which("nrnivmodl")
    if command is None:
        raise OracleUnavailable("nrnivmodl is unavailable; use the pinned Linux/Kaggle environment")
    completed = subprocess.run(
        [command, str(build_directory)], cwd=build_directory, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False,
    )
    if completed.returncode:
        raise OracleUnavailable(f"nrnivmodl failed ({completed.returncode}):\n{completed.stdout[-4000:]}")
    return build_directory


def run_double_oracle(
    formula: ExtractedGateFormula,
    neuron_oracle: NeuronIsolatedGateOracle,
    *,
    voltages_mv: Iterable[float] = DEFAULT_VOLTAGES_MV,
    states: Iterable[float] = DEFAULT_STATES,
    dt_ms: Iterable[float] = DEFAULT_DT_MS,
    atol: float = DEFAULT_ATOL,
    rtol: float = DEFAULT_RTOL,
) -> dict[str, Any]:
    voltages_mv = tuple(float(value) for value in voltages_mv)
    states = tuple(float(value) for value in states)
    dt_ms = tuple(float(value) for value in dt_ms)
    cases = []
    maximum_absolute_error = 0.0
    maximum_relative_error = 0.0
    worst_case: dict[str, Any] | None = None
    failure_count = 0
    for gate in ("m", "h"):
        for voltage in voltages_mv:
            for state in states:
                for step in dt_ms:
                    expected = formula.step(gate, state, voltage, step)
                    observed = neuron_oracle.step(gate, state, voltage, step)
                    absolute = abs(observed - expected)
                    relative = absolute / max(abs(expected), abs(observed), 1e-300)
                    passed = absolute <= atol + rtol * abs(expected)
                    failure_count += int(not passed)
                    case = {
                        "gate": gate, "voltage_mv": float(voltage), "state_t": float(state),
                        "dt_ms": float(step), "formula_state_t_plus_dt": expected,
                        "neuron_state_t_plus_dt": observed, "absolute_error": absolute,
                        "relative_error": relative, "passed": passed,
                    }
                    cases.append(case)
                    if absolute >= maximum_absolute_error:
                        maximum_absolute_error = absolute
                        maximum_relative_error = relative
                        worst_case = case
    return {
        "schema_version": "giada-double-oracle-v1",
        "task": "0.4",
        "mechanism": "Ca_HVA",
        "gates": ["m", "h"],
        "formula_oracle": {
            "kind": "NMODL rates procedure interpreted independently in Python",
            "source_path": str(formula.source_path),
            "source_sha256": formula.source_sha256,
            "procedure": formula.procedure,
            "assignment_count": len(formula.assignments),
        },
        "neuron_oracle": {
            "kind": "compiled NMODL DERIVATIVE states invoked by NEURON",
            "membrane_integrated": False,
            "runtime_version": str(getattr(neuron_oracle.h, "nrnversion")()),
        },
        "grid": {
            "voltage_count": len(voltages_mv), "state_count": len(states),
            "dt_count": len(dt_ms), "case_count": len(cases),
        },
        "tolerance": {"atol": atol, "rtol": rtol, "dtype": "float64"},
        "maximum_absolute_error": maximum_absolute_error,
        "maximum_relative_error_at_worst_absolute_case": maximum_relative_error,
        "failure_count": failure_count,
        "worst_case": worst_case,
        "valid": failure_count == 0,
        "environment": {"python": platform.python_version(), "platform": platform.platform()},
        "cases": cases,
    }


def write_double_oracle_report(report: dict[str, Any], json_path: str | Path, markdown_path: str | Path) -> None:
    json_path, markdown_path = Path(json_path), Path(markdown_path)
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = [
        "# GIADA Task 0.4 — Ca_HVA double oracle", "",
        f"- Valid: **{report['valid']}**", f"- Cases: **{report['grid']['case_count']}**",
        f"- Failures: **{report['failure_count']}**",
        f"- Maximum absolute error: **{report['maximum_absolute_error']:.3e}**", "",
        "## Independence contract", "",
        "The formula oracle interprets assignments extracted from `Ca_HVA.mod`. The NEURON oracle invokes the compiled NMODL `DERIVATIVE states` procedure directly. It does not reuse the Python formula and it does not integrate membrane voltage.", "",
        "## Acceptance", "",
        f"Every case must satisfy `abs(error) <= {report['tolerance']['atol']:.1e} + {report['tolerance']['rtol']:.1e} * abs(reference)`.", "",
    ]
    markdown_path.write_text("\n".join(lines), encoding="utf-8")
