"""Local SQL readback and launch/backup receipts, no memory mutation."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from research_memory.payload_store import open_readonly
from research_memory.mirror import write_json
F=ROOT/'experiments/results/task51_protocol_v1'
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--supervisor-pid',type=int,required=True);args=ap.parse_args()
    receipt=json.loads((F/'preregistration_receipt.json').read_text());backup=json.loads((ROOT/'research_memory/data/local_backup_receipt.json').read_text());launch=json.loads((F/'kaggle_launch.json').read_text());live=json.loads((F/'live_log.json').read_text());assert receipt['valid'] and backup['valid'] and backup['semantic_records']==receipt['verification']['records']
    groups={}
    for code,rid in receipt['records'].items():groups.setdefault(code.split('-giada-')[0],[]).append(rid)
    rows={}
    with open_readonly(ROOT/'research_memory/data/research_memory.sqlite') as con:
        for table,ids in groups.items():rows[table]=[dict(r) for r in con.execute(f'SELECT _record_id,Nome,"Codice stabile" FROM "{table}" WHERE _record_id IN ({",".join("?" for _ in ids)})',ids)]
    assert {r['_record_id'] for group in rows.values() for r in group}==set(receipt['records'].values());write_json(F/'query_readback.json',dict(valid=True,airtable_accessed=False,records=rows))
    started=any('dataset ready' in e.get('data','') or '[GIADA51] training' in e.get('data','') for e in live['events'])
    write_json(F/'workflow.json',dict(launch=launch,local_preflight_tests=4,remote_dataset_ready=started,remote_evidence=live['events'],SQLite_registration_verified=True,SQLite_backup_sha256=backup['sqlite_sha256'],DB_backup_only_local=True,supervisor_pid=args.supervisor_pid,external_final_backup='Pending completion; supervisor downloads ZIP and verifies SHA/size/CRC',P_frozen=True,thresholds_changed=False))
    print('50/51 SQL readback and launch/backup proof recorded',flush=True)
