"""Deterministic structural inventory of NMODL mechanisms.

This module deliberately records syntax and executable contracts only.  It does
not decide which variables are causal state or which neural component should
replace a mechanism; those are separate scientific decisions.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any, Iterable


_DECLARATION_BLOCKS = ("STATE", "PARAMETER", "ASSIGNED")
_EXECUTABLE_BLOCKS = ("INITIAL", "BREAKPOINT", "DERIVATIVE", "KINETIC", "NET_RECEIVE")
_RESERVED = {
    "and", "else", "exp", "fabs", "from", "if", "log", "net_event", "net_send",
    "nrn_random_pick", "nrn_random_play", "pow", "printf", "rates", "return",
    "setRNG", "sqrt", "state", "states", "tanh", "to", "urand", "while",
}


def _strip_comments(text: str) -> str:
    text = re.sub(r"(?is)\bCOMMENT\b.*?\bENDCOMMENT\b", "", text)
    text = re.sub(r"(?m):.*$", "", text)
    return text


def _balanced_block(text: str, keyword: str) -> tuple[str, int] | None:
    match = re.search(rf"(?m)^\s*{re.escape(keyword)}\b[^{{]*{{", text)
    if not match:
        return None
    start = text.find("{", match.start())
    depth = 0
    for index in range(start, len(text)):
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
            if depth == 0:
                return text[start + 1 : index], text.count("\n", 0, match.start()) + 1
    raise ValueError(f"unclosed {keyword} block")


def _declarations(body: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for raw in body.splitlines():
        line = raw.strip()
        if not line or line.startswith(("LOCAL", "UNITSOFF", "UNITSON")):
            continue
        line = re.sub(r"\([^)]*\)", "", line)
        for item in line.split(","):
            if "=" not in item:
                for name in re.findall(r"\b[A-Za-z_]\w*\b", item):
                    if name not in {"FROM", "TO", "WITH"}:
                        rows.append({"name": name})
                continue
            match = re.match(r"\s*([A-Za-z_]\w*)(?:\s*=\s*([^<]+?))?(?:\s*<([^>]*)>)?\s*$", item)
            if match:
                row: dict[str, Any] = {"name": match.group(1)}
                if match.group(2):
                    row["default"] = match.group(2).strip()
                if match.group(3):
                    row["bounds"] = match.group(3).strip()
                rows.append(row)
    seen: set[str] = set()
    return [row for row in rows if not (row["name"] in seen or seen.add(row["name"]))]


def _identifiers(expression: str) -> list[str]:
    names = re.findall(r"\b[A-Za-z_]\w*\b", expression)
    return sorted({name for name in names if name not in _RESERVED and not name.isupper()})


def _equations(block_name: str, body: str) -> list[dict[str, Any]]:
    rows = []
    for number, raw in enumerate(body.splitlines(), 1):
        line = raw.strip()
        match = re.match(r"([A-Za-z_]\w*'?)(?:\s*)=\s*(.+)$", line)
        if not match or "==" in line:
            continue
        target, expression = match.group(1), match.group(2).strip()
        rows.append({
            "block": block_name,
            "line_in_block": number,
            "target": target,
            "expression": expression,
            "dependencies": [name for name in _identifiers(expression) if name != target.rstrip("'")],
        })
    return rows


def _neuron_contract(body: str) -> dict[str, Any]:
    kind_match = re.search(r"\b(SUFFIX|POINT_PROCESS|ARTIFICIAL_CELL)\s+([A-Za-z_]\w*)", body)
    ions = []
    for match in re.finditer(r"(?m)^\s*USEION\s+(\w+)\s+([^\n]+)", body):
        ion, tail = match.groups()
        read_match = re.search(r"\bREAD\s+(.+?)(?=\s+WRITE\b|\s+VALENCE\b|$)", tail)
        write_match = re.search(r"\bWRITE\s+(.+?)(?=\s+READ\b|\s+VALENCE\b|$)", tail)
        ions.append({
            "ion": ion,
            "read": re.findall(r"[A-Za-z_]\w*", read_match.group(1)) if read_match else [],
            "write": re.findall(r"[A-Za-z_]\w*", write_match.group(1)) if write_match else [],
        })
    currents = []
    for match in re.finditer(r"\bNONSPECIFIC_CURRENT\s+([^\n]+)", body):
        currents.extend(re.findall(r"[A-Za-z_]\w*", match.group(1)))
    currents.extend(variable for ion in ions for variable in ion["write"] if variable.startswith("i"))
    return {
        "kind": kind_match.group(1) if kind_match else "UNKNOWN",
        "name": kind_match.group(2) if kind_match else None,
        "ions": ions,
        "currents": sorted(set(currents)),
        "range": sorted(set(sum((re.findall(r"[A-Za-z_]\w*", m.group(1)) for m in re.finditer(r"\bRANGE\s+([^\n]+)", body)), []))),
        "global": sorted(set(sum((re.findall(r"[A-Za-z_]\w*", m.group(1)) for m in re.finditer(r"\bGLOBAL\s+([^\n]+)", body)), []))),
        "pointers": sorted(set(sum((re.findall(r"[A-Za-z_]\w*", m.group(1)) for m in re.finditer(r"\b(?:POINTER|BBCOREPOINTER)\s+([^\n]+)", body)), []))),
    }


def parse_mod_file(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    raw = path.read_text(encoding="utf-8", errors="replace")
    text = _strip_comments(raw)
    neuron = _balanced_block(text, "NEURON")
    if neuron is None:
        raise ValueError(f"{path}: NEURON block not found")
    declarations = {}
    for name in _DECLARATION_BLOCKS:
        block = _balanced_block(text, name)
        declarations[name.lower()] = _declarations(block[0]) if block else []
    blocks: dict[str, Any] = {}
    equations: list[dict[str, Any]] = []
    for name in _EXECUTABLE_BLOCKS:
        block = _balanced_block(text, name)
        if block:
            blocks[name.lower()] = {"source_line": block[1], "present": True}
            equations.extend(_equations(name, block[0]))
    solves = [
        {"target": m.group(1), "method": m.group(2) or None}
        for m in re.finditer(r"\bSOLVE\s+(\w+)(?:\s+METHOD\s+(\w+))?", text)
    ]
    procedures = sorted(set(re.findall(r"(?m)^\s*(?:PROCEDURE|FUNCTION)\s+([A-Za-z_]\w*)", text)))
    contract = _neuron_contract(neuron[0])
    return {
        "file": path.name,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "mechanism": contract,
        "declarations": declarations,
        "integration": {"solve_statements": solves},
        "executable_blocks": blocks,
        "procedures_and_functions": procedures,
        "equations": equations,
    }


def _git_commit(root: Path) -> str | None:
    try:
        return subprocess.run(
            ["git", "-c", f"safe.directory={root}", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def build_inventory(mod_dir: str | Path, *, teacher_root: str | Path | None = None) -> dict[str, Any]:
    mod_dir = Path(mod_dir).resolve()
    paths = sorted(mod_dir.glob("*.mod"), key=lambda p: p.name.lower())
    if not paths:
        raise ValueError(f"no .mod files found in {mod_dir}")
    mechanisms = [parse_mod_file(path) for path in paths]
    combined = hashlib.sha256("".join(row["sha256"] for row in mechanisms).encode()).hexdigest()
    summary = {
        "mod_file_count": len(mechanisms),
        "state_variable_count": sum(len(row["declarations"]["state"]) for row in mechanisms),
        "parameter_count": sum(len(row["declarations"]["parameter"]) for row in mechanisms),
        "assigned_variable_count": sum(len(row["declarations"]["assigned"]) for row in mechanisms),
        "ion_contract_count": sum(len(row["mechanism"]["ions"]) for row in mechanisms),
        "current_count": sum(len(row["mechanism"]["currents"]) for row in mechanisms),
        "net_receive_mechanism_count": sum("net_receive" in row["executable_blocks"] for row in mechanisms),
    }
    root = Path(teacher_root).resolve() if teacher_root else mod_dir
    return {
        "schema_version": "giada-teacher-mechanism-inventory-v1",
        "scope": "syntactic-and-executable-contract-only",
        "teacher_root": str(root),
        "teacher_commit": _git_commit(root),
        "mod_directory": str(mod_dir),
        "combined_source_sha256": combined,
        "summary": summary,
        "mechanisms": mechanisms,
    }


def _markdown(inventory: dict[str, Any]) -> str:
    lines = [
        "# GIADA teacher mechanism inventory", "",
        f"- Teacher commit: `{inventory['teacher_commit']}`",
        f"- Combined source SHA-256: `{inventory['combined_source_sha256']}`",
        f"- NMODL files: **{inventory['summary']['mod_file_count']}**", "",
        "| Mechanism | Kind | STATE | Ions read/write | Currents | Solver | NET_RECEIVE |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in inventory["mechanisms"]:
        mech = row["mechanism"]
        ions = "; ".join(f"{x['ion']}: R={','.join(x['read']) or '-'} W={','.join(x['write']) or '-'}" for x in mech["ions"]) or "—"
        solver = "; ".join(f"{x['target']}:{x['method'] or 'unspecified'}" for x in row["integration"]["solve_statements"]) or "—"
        lines.append(
            f"| `{mech['name'] or row['file']}` | {mech['kind']} | "
            f"{', '.join(x['name'] for x in row['declarations']['state']) or '—'} | {ions} | "
            f"{', '.join(mech['currents']) or '—'} | {solver} | "
            f"{'yes' if 'net_receive' in row['executable_blocks'] else 'no'} |"
        )
    lines += ["", "> This inventory is descriptive. Causal classification is intentionally deferred to Task 0.2.", ""]
    return "\n".join(lines)


def write_inventory(inventory: dict[str, Any], json_path: str | Path, markdown_path: str | Path | None = None) -> None:
    json_path = Path(json_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(inventory, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if markdown_path:
        Path(markdown_path).write_text(_markdown(inventory), encoding="utf-8")
