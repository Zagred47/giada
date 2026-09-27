"""Read-only, parameter-bound catalog audit before GIADA Task 17."""

import json
import sqlite3
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CATALOG = {row["id"]: row for row in json.loads(
    (ROOT / "queries/query_catalog_review.json").read_text(encoding="utf-8"))}
DB = (ROOT / "data/research_memory.sqlite").resolve()


def main():
    cases = [
        ("Q09", {}), ("Q11", {}), ("Q14", {}), ("Q27", {}),
        ("Q28", {}), ("Q29", {}), ("Q31", {}), ("Q32", {}), ("Q34", {}),
        ("Q10", {"model_id": "rec45ab4e6a5bd32f"}),
        ("Q24", {"model_id": "rec45ab4e6a5bd32f"}),
        ("Q08", {"claim_id": "rec90931b623c5191"}),
        ("Q12", {"claim_id": "rec90931b623c5191"}),
        ("Q20", {"experiment_id": "rec3655cd16bc32f0"}),
        ("Q20", {"experiment_id": "recba5e744edb0b2d"}),
        ("Q25", {"experiment_id": "rec3655cd16bc32f0"}),
        ("Q22", {"needle": "causal"}),
        ("Q22", {"needle": "voltaggio"}),
        ("Q22", {"needle": "Ca_HVA"}),
    ]
    with sqlite3.connect(DB.as_uri() + "?mode=ro", uri=True) as conn:
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA query_only=ON")
        for query_id, params in cases:
            query = CATALOG[query_id]
            cursor = conn.execute(query["full_sql"], params)
            rows = [dict(row) for row in cursor.fetchmany(501)]
            print(json.dumps({"query": query_id, "params": params,
                              "row_count": len(rows), "truncated": len(rows) > 500,
                              "rows": rows[:12]}, ensure_ascii=True, default=str))


if __name__ == "__main__":
    main()
