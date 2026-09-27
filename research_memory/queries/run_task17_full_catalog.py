"""Run every registered SQL lens read-only on real GIADA mirror examples.

This is a coverage/diagnostic audit. Zero rows can mean absent relational links;
it is never interpreted as a scientific negative result.
"""

from __future__ import annotations

import json
import sqlite3

from .run_task17_preflight import CATALOG, DB


PARAMS = {
    "model_id": "rec45ab4e6a5bd32f",
    "model_a": "rec45ab4e6a5bd32f",
    "model_b": "rec45ab4e6a5bd32f",
    "component_id": "rec27072c598592e1",
    "from_component": "rec27072c598592e1",
    "to_component": "reca32b3d68ccd764",
    "max_depth": 12,
    "teacher_model": "rec45ab4e6a5bd32f",
    "claim_id": "rec3b43da20f52430",
    "evaluation_id": "rec28ffe5f0a4362d",
    "factor_id": "recca4a56c35069c4",
    "contrast_id": "rec0bYJMHw9hkgh8U",
    "factor_a": "recca4a56c35069c4",
    "factor_b": "recd40e949ccde1c9",
    "level_a0": "recfcfae08b88e2cb",
    "level_a1": "rec119b7d34565b86",
    "level_b0": "rec10ada432a6f339",
    "level_b1": "rec813590daf6da88",
    "substitution_id": "rec00000000000000",  # no substitution record exists
    "experiment_id": "rec3655cd16bc32f0",  # Task 16
    "axis": "instances",
    "needle": "Ca_HVA",
    "expected_strata_json": '["soma:quiet","soma:spike"]',
}


def main() -> dict:
    summary = []
    with sqlite3.connect(DB.as_uri() + "?mode=ro", uri=True) as conn:
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA query_only=ON")
        for key in sorted(CATALOG, key=lambda value: int(value[1:])):
            query = CATALOG[key]
            params = {name: PARAMS[name] for name in query["params"]}
            rows = [dict(row) for row in conn.execute(query["full_sql"], params).fetchmany(1001)]
            summary.append({"query": key, "rows": len(rows), "truncated": len(rows) > 1000,
                            "parameter_keys": list(params),
                            "example": rows[0] if rows else None})
    return {"query_count": len(summary), "read_only": True, "results": summary}


if __name__ == "__main__":
    print(json.dumps(main(), ensure_ascii=False, indent=2, default=str))
