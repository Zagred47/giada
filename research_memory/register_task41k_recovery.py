"""Record a technical evaluator recovery, never a scientific model failure."""
import hashlib,json
from .mirror import ROOT,write_json
from .incremental_batch import IncrementalBatchMirror


def main():
    repo=ROOT.parent;folder=repo/'experiments/results/task41k_protocol_v1'
    recovery=json.loads((folder/'evaluation_recovery_launch.json').read_text())
    baseline=json.loads((folder/'launch_memory_receipt.json').read_text())['verification']
    m=IncrementalBatchMirror()
    exp=m.query("SELECT record_id FROM v_records WHERE stable_code='experiments-giada-task41k-comparison-v1'")['rows'][0]['record_id']
    artifact=m.local_upsert('artifacts',{'Codice stabile':'artifacts-giada-task41k-evaluation-recovery-v1',
        'Nome':'Task41k frozen evaluation recovery','Tipo':'Report','Versione':'v1',
        'Percorso':'experiments/results/task41k_protocol_v1/evaluation_recovery_launch.json',
        'SHA-256':hashlib.sha256((folder/'evaluation_recovery_launch.json').read_bytes()).hexdigest(),
        'Dimensione byte':(folder/'evaluation_recovery_launch.json').stat().st_size})['record_id']
    observation=m.local_upsert('observations',{'Codice stabile':'observations-giada-task41k-evaluation-recovery-v1',
        'Nome':'Task41k technical CUDA capture interruption and frozen recovery',
        'Descrizione':json.dumps(dict(recovery=recovery,technical_failure=True,scientific_no_go=False,
            checkpoint_finite=True,diagnosis='Scanner.step closes over self, retaining graph pools; lifetime fix tested on CPU. Exact-batch GPU parity and release preflight mandatory before missing quality panels.'),ensure_ascii=False),
        'Artefatti dettagliati':[artifact]})['record_id']
    m.local_upsert('findings',{'Codice stabile':'findings-giada-task41k-evaluation-recovery-v1',
        'Nome':'Task41k training complete but full scientific evaluation pending',
        'Esperimenti':[exp],'Osservazioni':[observation],
        'Risultato':'1024 updates completed; recovered checkpoint tensors finite. Evaluation v4 interrupted by CUDA graph capture failure. Partial panels, freeze and frozen weights preserved with verified SHA256.',
        'Limitazioni':'Do not treat inherited training final_report.json as full scientific outcome. No retraining, no reselection, no threshold or abstraction change. CUDA lifecycle fix requires actual GPU confirmation.'})
    change=m.commit_batch();verification=m.verify_incremental(baseline);assert verification['valid']
    write_json(folder/'recovery_memory_receipt.json',dict(valid=True,change=change,verification=verification,
        airtable_accessed=False,scientific_result_pending=True,DB_backup='Local operational DB updated; prior full backup retained.'))
    m.export_snapshot_to(ROOT/'data/airtable_snapshot.json')
    print('Task41k recovery recorded and verified in local SQLite',flush=True)


if __name__=='__main__':main()
