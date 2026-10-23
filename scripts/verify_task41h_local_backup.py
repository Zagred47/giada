"""Bind the local-only consistent SQLite backup to the verified launch state."""
import json,sqlite3,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from research_memory.mirror import write_json

receipt=json.loads((ROOT/'research_memory/data/local_backup_receipt.json').read_text())
launch=json.loads((ROOT/'experiments/results/task41h_protocol_v1/launch_memory_receipt.json').read_text())
assert receipt['valid'] and receipt['integrity_check']=='ok' and not receipt['foreign_key_errors']
assert 'Task41h' in receipt['scope'] and not receipt['drive_updated'] and not receipt['airtable_accessed']
path=Path(receipt['local_sqlite']);assert path.stat().st_size==receipt['sqlite_bytes']
with sqlite3.connect(path.resolve().as_uri()+'?mode=ro',uri=True) as con:
    snapshot=con.execute("SELECT value FROM _mirror_meta WHERE key='snapshot_hash'").fetchone()[0]
assert snapshot==launch['verification']['snapshot_hash']
result=dict(valid=True,snapshot_hash=snapshot,sqlite_sha256=receipt['sqlite_sha256'],bytes=receipt['sqlite_bytes'],
    records=receipt['semantic_records'],local_sqlite=str(path),integrity_check='ok',foreign_key_errors=[],
    DB_backup_local_only=True,drive_updated=False,airtable_accessed=False,scope=receipt['scope'])
write_json(ROOT/'experiments/results/task41h_protocol_v1/local_backup_verification.json',result)
print(json.dumps(result),flush=True)
