"""Recover export/verification after the already committed39i launch batch."""
import gc,json
from .mirror import Mirror,ROOT,write_json
def main():
    m=Mirror();rows=m.query("SELECT \"Stato\",count(*) AS n FROM runs WHERE \"Codice stabile\" LIKE 'runs-giada-task39i-temporal-%' GROUP BY \"Stato\"")['rows']
    assert rows==[{'Stato':'In corso','n':36}],rows
    print(json.dumps(dict(stage='export_committed_launch',runs=rows)),flush=True)
    write_json(ROOT/'data/airtable_snapshot.json',m.export_snapshot());gc.collect()
    verification=m.verify();assert verification['valid']
    rows=m.query("SELECT stable_code,record_id FROM v_records WHERE stable_code LIKE '%-giada-task39i-temporal-%'",limit=1000)['rows']
    receipt=dict(valid=True,records={r['stable_code']:r['record_id'] for r in rows},verification=verification,
        airtable_accessed=False,drive_backup_updated=False,scientific_result_claimed=False,
        recovered_after_export_memory_error=True,semantic_commit_completed_before_export=True,
        recovery='Stream identical JSON bytes atomically; no repeated semantic write or training restart')
    write_json(ROOT.parent/'experiments/results/task39i_protocol_v1/launch_memory_receipt.json',receipt)
    print(json.dumps(dict(valid=True,records=verification['records'],task39i_records=len(rows))),flush=True)
if __name__=='__main__':main()
