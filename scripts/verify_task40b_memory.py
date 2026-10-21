"""Read-only provenance check after Task40b mirror transaction."""
import hashlib,json,sqlite3
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
    c=sqlite3.connect((ROOT/'research_memory/data/research_memory.sqlite').resolve().as_uri()+'?mode=ro',uri=True)
    c.row_factory=sqlite3.Row
    rows=[dict(r) for r in c.execute('SELECT Percorso,"SHA-256" AS sha FROM artifacts WHERE "Codice stabile" LIKE ?',('artifacts-giada-task40b-result-%',))]
    mismatches=[]
    for row in rows:
        path=ROOT/row['Percorso'];actual=hashlib.sha256(path.read_bytes()).hexdigest()
        if actual!=row['sha']:mismatches.append(dict(path=row['Percorso'],actual=actual,registered=row['sha']))
    result=dict(valid=not mismatches,artifact_hashes_verified=len(rows),mismatches=mismatches,read_only=True)
    print(json.dumps(result),flush=True)
    assert not mismatches,result
if __name__=='__main__':main()
