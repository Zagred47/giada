"""Freeze the verified transport amendment without changing scientific protocol."""
import json,hashlib,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from research_memory.mirror import write_json

def main():
    folder=ROOT/'experiments/results/task42_transport_recovery_v1'
    transfer=json.loads((folder/'input_verified.json').read_text());cpu=json.loads((folder/'cpu_input_verified.json').read_text())
    old=json.loads((ROOT/'experiments/task42_gpu_recovery_v1.json').read_text())
    assert transfer['valid'] and transfer['api_readback_verified'] and cpu['valid'] and cpu['GPU_used'] is False
    assert transfer['sha256']==cpu['sha256']==old['source_sha256']
    plan=dict(task='42',version='verified-transport-recovery-v1',original_gpu_recovery=old,
        transfer_readback=transfer,actual_cpu_mount=cpu,
        cause='Prior worker required exactly one archive filename via kernel output mounting; that contract failed. No inventory was logged, so missing vs multiple files/provider extraction is not established.',
        correction='Explicit private Dataset carrying original ZIP bytes as native42.zip.bin. API download readback plus actual Kaggle CPU mount verify bytes/SHA256/CRC/eight members before GPU submission. Mounted path passed explicitly to unchanged GPU evaluator; duplicate byte-identical copies are safe.',
        training_updates=0,selection_performed=False,native_regenerated=False,thresholds_changed=False,abstraction_changed=False,
        worker='scripts/run_task42_gpu_recovery.py',
        original_protocol_canonical_sha256=old['original_protocol_canonical_sha256'],
        stop='No GPU submission without both readback and actual CPU mount receipts. Reverify the actual mounted archive before pip/GPU setup and in worker. Hash mismatch/missing input stops without native fallback.',
        backup='Original18MB native output, prior failed recovery logs and private input preserved; new final output recovered with SHA256/CRC. Compact DB and report bundle local only, last2 verified generations.')
    write_json(ROOT/'experiments/task42_verified_transport_v1.json',plan)
    print('Verified transport amendment frozen; scientific configuration unchanged.')
if __name__=='__main__':main()
