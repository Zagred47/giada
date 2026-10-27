"""Attach a verified local SQLite backup receipt, never upload the database."""
import json,sqlite3,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from research_memory.mirror import write_json

backup=json.loads((ROOT/'research_memory/data/local_backup_receipt.json').read_text())
assert backup['valid'] and backup['single_file'] and not backup['drive_updated']
assert backup['integrity_check']=='ok' and not backup['foreign_key_errors']
path=Path(backup['local_sqlite']);assert path.stat().st_size==backup['sqlite_bytes']
latest=json.loads((ROOT/'experiments/results/task41k_protocol_v1/launch_memory_receipt.json').read_text())
with sqlite3.connect(path.resolve().as_uri()+'?mode=ro',uri=True) as con:
    logical=con.execute("SELECT value FROM _mirror_meta WHERE key='snapshot_hash'").fetchone()[0]
assert logical==latest['verification']['snapshot_hash']
assert backup['semantic_records']==latest['verification']['records']
for name in ('task41j_audited_v1/memory_receipt.json','task41k_protocol_v1/memory_receipt.json','task41k_protocol_v1/launch_memory_receipt.json'):
    target=ROOT/'experiments/results'/name;receipt=json.loads(target.read_text())
    assert receipt['valid'];receipt['DB_backup']=dict(backup,logical_snapshot_hash=logical)
    write_json(target,receipt)
print(json.dumps(dict(valid=True,records=backup['semantic_records'],sha256=backup['sqlite_sha256'],drive_updated=False)),flush=True)
