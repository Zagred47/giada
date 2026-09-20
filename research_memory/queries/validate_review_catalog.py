"""Validate the proposed queries without changing the on-disk mirror or Airtable."""
import json
from pathlib import Path
import sqlite3

HERE = Path(__file__).resolve().parent
DB = HERE.parent / "data" / "research_memory.sqlite"
CATALOG = json.loads((HERE / "query_catalog_review.json").read_text(encoding="utf-8"))
BY_ID = {q["id"]: q for q in CATALOG}


def run(conn, qid, **params):
    return [dict(row) for row in conn.execute(BY_ID[qid]["full_sql"], params)]


def main():
    source = sqlite3.connect(DB.resolve().as_uri() + "?mode=ro", uri=True)
    source.row_factory = sqlite3.Row
    source.execute("PRAGMA query_only=ON")
    syntax = []
    for query in CATALOG:
        params = {p: "rec00000000000000" for p in query["params"]}
        if "max_depth" in params:
            params["max_depth"] = 12
        if "expected_strata_json" in params:
            params["expected_strata_json"] = '["riposo","spike"]'
        if "axis" in params:
            params["axis"] = "instances"
        if "needle" in params:
            params["needle"] = "calcio"
        source.execute(query["full_sql"], params).fetchall()
        syntax.append(query["id"])

    # Fixtures are inserted ONLY in an in-memory SQLite database.
    mem = sqlite3.connect(":memory:")
    mem.row_factory = sqlite3.Row
    source.backup(mem)
    source.close()
    relations = list(mem.execute("SELECT * FROM _mirror_relations"))
    keys = [r[0] for r in mem.execute("SELECT table_key FROM _mirror_tables")]
    for rel in relations:
        mem.execute('DELETE FROM "' + rel["sql_table"] + '"')
    for key in keys:
        mem.execute('DELETE FROM "' + key + '"')
    mem.execute("DELETE FROM _remote_records")
    mem.execute("DELETE FROM _outbox")

    def record(key, ident, **fields):
        columns = ["_record_id", "Nome"] + list(fields)
        mem.execute('INSERT INTO "' + key + '" (' + ','.join('"' + c + '"' for c in columns) + ') VALUES (' + ','.join('?' for _ in columns) + ')', [ident, ident, *fields.values()])
        return ident

    def link(key, role, source_id, target_id):
        rel = next(r for r in relations if r["source_table"] == key and r["field_name"] == role)
        mem.execute('INSERT INTO "' + rel["sql_table"] + '" VALUES (?,?,0,0)', (source_id, target_id))

    record("models", "model")
    for ident in ("root", "child", "leaf"):
        record("instances", ident, **{"Identificatore locale": ident})
        link("instances", "Modello", ident, "model")
    for edge, parent, child in (("part1", "root", "child"), ("part2", "child", "leaf"), ("cycle", "leaf", "root")):
        record("composition", edge)
        link("composition", "Modello", edge, "model")
        link("composition", "Contenitore", edge, parent)
        link("composition", "Parte", edge, child)
    tree = run(mem, "Q02", model_id="model", component_id="root", max_depth=12)
    assert [r["_record_id"] for r in tree] == ["root", "child", "leaf"], tree

    record("claims", "claim", Stato="Aperta")
    record("claims", "without_evidence", Stato="Aperta")
    for ident, outcome in (("yes", "Sostiene"), ("no", "Contraddice")):
        record("evidence", ident, Esito=outcome)
        link("evidence", "Affermazione valutata", ident, "claim")
    contradictory = run(mem, "Q09")
    assert len(contradictory) == 1 and contradictory[0]["sostiene"] == 1 and contradictory[0]["contraddice"] == 1
    assert {r["_record_id"] for r in run(mem, "Q11")} == {"without_evidence"}

    record("evaluations", "eval")
    record("observations", "o1", Valore=0.4, **{"Strato o sottogruppo": "riposo"})
    record("observations", "o2", **{"Strato o sottogruppo": "spike"})
    for ident in ("o1", "o2"):
        link("observations", "Specifica di valutazione", ident, "eval")
    strata = run(mem, "Q26", evaluation_id="eval", expected_strata_json='["riposo","spike","plateau"]')
    strata = {r["sottogruppo_atteso"]: r for r in strata}
    assert strata["riposo"]["misure_numeriche"] == 1
    assert strata["spike"]["misure_registrate"] == 1 and strata["spike"]["misure_numeriche"] == 0
    assert strata["plateau"]["misure_registrate"] == 0

    record("protocols", "protocol")
    record("blocks", "block")
    record("partitions", "split")
    link("evaluations", "Protocollo", "eval", "protocol")
    for factor, levels in (("fa", ("a0", "a1")), ("fb", ("b0", "b1"))):
        record("factors", factor)
        for level in levels:
            record("factorlevels", level, Valore=level)
            link("factorlevels", "Fattore", level, factor)
    # y00=1, y10=2, y01=3, y11=7 -> interaction = 3.
    for i, (a, b, value) in enumerate((("a0", "b0", 1), ("a1", "b0", 2), ("a0", "b1", 3), ("a1", "b1", 7))):
        arm, run_id, cp, obs = (f"arm{i}", f"run{i}", f"cp{i}", f"obs{i}")
        record("arms", arm)
        link("arms", "Protocollo", arm, "protocol")
        for factor, level in (("fa", a), ("fb", b)):
            ass = f"assignment{i}{factor}"
            record("assignments", ass)
            link("assignments", "Braccio", ass, arm)
            link("assignments", "Livello", ass, level)
        record("runs", run_id, Stato="Completata", Seed="17")
        link("runs", "Braccio", run_id, arm)
        link("runs", "Blocco", run_id, "block")
        link("runs", "Partizioni effettive", run_id, "split")
        record("checkpoints", cp, **{"Budget consumato": 100, "Unità budget": "step"})
        link("checkpoints", "Run", cp, run_id)
        record("observations", obs, Valore=value, Numerosità=10, **{"Strato o sottogruppo": "riposo"})
        link("observations", "Run", obs, run_id)
        link("observations", "Checkpoint", obs, cp)
        link("observations", "Specifica di valutazione", obs, "eval")
    params = dict(evaluation_id="eval", factor_a="fa", factor_b="fb", level_a0="a0", level_a1="a1", level_b0="b0", level_b1="b1")
    interaction = run(mem, "Q17", **params)
    assert len(interaction) == 1 and interaction[0]["interazione"] == 3, interaction
    assert len(run(mem, "Q15", evaluation_id="eval")) == 6
    record("contrasts", "contrast")
    for i in range(4):
        link("contrasts", "Bracci", "contrast", f"arm{i}")
    effects = run(mem, "Q16", evaluation_id="eval", factor_id="fa", contrast_id="contrast")
    assert sorted(r["differenza_b_meno_a"] for r in effects) == [1, 4], effects
    record("partitions", "other_split")
    link("runs", "Partizioni effettive", "run3", "other_split")
    assert len(run(mem, "Q15", evaluation_id="eval")) == 3
    assert run(mem, "Q17", **params) == [], "Different split sets must not form a 2x2 block"
    record("quality", "failed_check", Esito="Fallito")
    link("quality", "Run", "failed_check", "run0")
    assert len(run(mem, "Q15", evaluation_id="eval")) == 1
    mem.close()
    print(json.dumps({"syntax_checked": len(syntax), "query_ids": syntax,
                      "synthetic_checks": ["cycle-safe decomposition", "conflicting evidence", "missing evidence", "missing vs null vs measured subgroup", "2x2 interaction", "paired comparisons", "single-factor contrasts", "mismatched partitions rejected", "failed quality check excluded"],
                      "persistent_database_modified": False, "airtable_writes": 0}, indent=2))


if __name__ == "__main__":
    main()
