"""Read-only Task41 artifact provenance, no external service access."""
import hashlib,json,sqlite3
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
with sqlite3.connect((ROOT/'research_memory/data/research_memory.sqlite').resolve().as_uri()+'?mode=ro',uri=True) as c:
    rows=c.execute('SELECT Percorso,"SHA-256" FROM artifacts WHERE "Codice stabile" LIKE ?',('artifacts-giada-task41-%',)).fetchall()
    mismatches=[dict(path=p,registered=h,actual=hashlib.sha256((ROOT/p).read_bytes()).hexdigest()) for p,h in rows if hashlib.sha256((ROOT/p).read_bytes()).hexdigest()!=h]
    result=dict(valid=bool(rows) and not mismatches,artifact_hashes_verified=len(rows),mismatches=mismatches,read_only=True)
    print(json.dumps(result),flush=True);assert result['valid']
