"""Verify every original41f report survived the41g metadata registration."""
import hashlib
import json
from pathlib import Path
import sqlite3
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from research_memory.mirror import write_json
receipt=json.loads((ROOT/'experiments/results/task41f_audited_v1/memory_receipt.json').read_text())
db=ROOT/'research_memory/data/research_memory.sqlite'
rows=[]
with sqlite3.connect(db.resolve().as_uri()+'?mode=ro',uri=True) as con:
    for report in receipt['raw_reports']:
        found=con.execute('SELECT rowid FROM observations WHERE _record_id=?',(report['observation'],)).fetchone();assert found
        sha=hashlib.sha256()
        with con.blobopen('observations','Descrizione',found[0],readonly=True) as blob:
            while block:=blob.read(1024*1024):sha.update(block)
        assert sha.hexdigest()==report['canonical_json_sha256'],report['member']
        rows.append(dict(member=report['member'],sha256=sha.hexdigest(),valid=True))
value=dict(valid=True,reports=len(rows),all_original_scientific_JSON_byte_identical=True,read_only=True,rows=rows)
write_json(ROOT/'experiments/results/task41g_protocol_v1/preserved_reports.json',value)
print(json.dumps(dict(valid=True,reports=len(rows),all_original_scientific_JSON_byte_identical=True)),flush=True)
