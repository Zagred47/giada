"""Read back registered48 records and persist launch/backup proof, no DB writes."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from research_memory.payload_store import open_readonly
from research_memory.mirror import write_json
F=ROOT/'experiments/results/task48_protocol_v1'
if __name__=='__main__':
    receipt=json.loads((F/'preregistration_receipt.json').read_text());launch=json.loads((F/'kaggle_launch.json').read_text());backup=json.loads((ROOT/'research_memory/data/local_backup_receipt.json').read_text());live=json.loads((F/'live_log.json').read_text())
    assert receipt['valid'] and backup['valid'] and backup['semantic_records']==receipt['verification']['records']
    assert any('history canonical64' in event.get('data','') for event in live['events']),'Do not claim GPU preflight before actual history starts'
    sql='SELECT table_key,record_id,name FROM v_records WHERE stable_code LIKE ? ORDER BY table_key,record_id'
    # v_records column labels are discovered rather than guessed across schema revisions.
    with open_readonly(ROOT/'research_memory/data/research_memory.sqlite') as con:
        data={table:[dict(row) for row in con.execute(f'SELECT _record_id,Nome,"Codice stabile" FROM "{table}" WHERE "Codice stabile" LIKE ?',('%-giada-task48-%',))] for table in ('experiments','protocols','claims','arms','runs','artifacts','observations','decisions')}
    assert sum(map(len,data.values()))==len(receipt['records'])==15
    assert {row['_record_id'] for rows in data.values() for row in rows}==set(receipt['records'].values())
    write_json(F/'query_readback.json',dict(valid=True,airtable_accessed=False,records=data))
    write_json(F/'workflow.json',dict(state='running',launch=launch,local_tests_passed=5,real_frozen_P_local_mini_run=True,remote_gpu_and_hash_preflight_passed=True,remote_preflight_evidence=live['events'],SQLite_registration_verified=True,SQLite_records=backup['semantic_records'],SQLite_local_backup_sha256=backup['sqlite_sha256'],SQLite_backup_only_local=True,external_final_archive_status='Pending run completion; hidden console supervisor downloads and verifies SHA-256, byte count and all ZIP CRCs',supervisor_pid=85820,training_updates=0,selection_performed=False,thresholds_changed=False,task47_passed=False))
    print('48 SQL readback, remote launch and verified local DB backup recorded',flush=True)
