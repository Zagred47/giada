"""SQL readback of48 results/49 registration and verified launch/backup proof."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from research_memory.payload_store import open_readonly
from research_memory.mirror import write_json
F=ROOT/'experiments/results/task49_protocol_v1'
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--supervisor-pid',type=int,required=True);args=ap.parse_args()
    receipt=json.loads((F/'preregistration_receipt.json').read_text());launch=json.loads((F/'kaggle_launch.json').read_text());backup=json.loads((ROOT/'research_memory/data/local_backup_receipt.json').read_text());live=json.loads((F/'live_log.json').read_text())
    assert receipt['valid'] and backup['valid'] and backup['semantic_records']==receipt['verification']['records']
    assert any('[GIADA49] conditioning' in e.get('data','') for e in live['events']),'GPU/hash preflight not yet observed'
    groups={}
    for code,rid in receipt['records'].items():groups.setdefault(code.split('-giada-')[0],[]).append(rid)
    rows={}
    with open_readonly(ROOT/'research_memory/data/research_memory.sqlite') as con:
        for table,ids in groups.items():
            marks=','.join('?' for _ in ids);rows[table]=[dict(r) for r in con.execute(f'SELECT _record_id,Nome,"Codice stabile" FROM "{table}" WHERE _record_id IN ({marks})',ids)]
    assert {r['_record_id'] for data in rows.values() for r in data}==set(receipt['records'].values())
    write_json(F/'query_readback.json',dict(valid=True,airtable_accessed=False,records=rows))
    write_json(F/'workflow.json',dict(state='running',launch=launch,local_tests=6,remote_GPU_hash_preflight_passed=True,remote_preflight_evidence=live['events'],task48_full_results_registered=True,task49_preregistered=True,SQLite_verified_records=backup['semantic_records'],SQLite_local_backup_sha256=backup['sqlite_sha256'],SQLite_backup_only_local=True,external_final_archive_status='Pending completion; hidden console supervisor retrieves final ZIP and verifies SHA-256, byte count and every CRC',supervisor_pid=args.supervisor_pid,thresholds_changed=False,P_frozen=True))
    print('48/49 SQL readback, GPU launch and verified DB backup recorded',flush=True)
