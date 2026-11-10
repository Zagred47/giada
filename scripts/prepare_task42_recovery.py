"""Pin the failed42 output as immutable GPU-only recovery input."""
import hashlib,json,sys,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from research_memory.mirror import write_json

def main():
    receipt=json.loads((ROOT/'artifacts/task42_kaggle_external/external_backup_receipt.json').read_text())
    assert receipt['valid'] and receipt['technical_failure']
    path=Path(receipt['local'])
    with path.open('rb') as stream:assert hashlib.file_digest(stream,'sha256').hexdigest()==receipt['remote']['sha256']
    config=json.loads((ROOT/'experiments/task42_preregistration_v1.json').read_text())
    members=['native_'+g['id']+'.npz' for g in config['geometries']]+['native_floor.json','frozen41r.pt','preregistration.json']
    with zipfile.ZipFile(path) as archive:
        assert archive.testzip() is None
        assert json.loads(archive.read('preregistration.json'))==config
        hashes={name:hashlib.sha256(archive.read(name)).hexdigest() for name in members}
    plan=dict(task='42',version='gpu-recovery-v1',source_sha256=receipt['remote']['sha256'],source_bytes=path.stat().st_size,
        source_kernel='alessandrobelli/giada-task42-exploratory-axial-port',source_archive='task42_final.zip',members=hashes,
        original_protocol_canonical_sha256=hashlib.sha256(json.dumps(config,sort_keys=True,separators=(',',':')).encode()).hexdigest(),
        adapter_fix='Preserve [seed,case,node] and reshape only case/node to [seed,case*node]; never flatten seed into sites.',
        native_regenerated=False,training_updates=0,selection_performed=False,thresholds_changed=False,abstraction_changed=False,
        scope='Only rerun formula and frozen41r GPU evaluations. All60 native active traces,15 passive checks and frozen checkpoint restored byte-for-byte from saved output. Original failure retained as technical, not scientific.',
        stop='Abort on original/archive/member/protocol/weight hash mismatch, missing native traces, nonfinite input or GPU/parity error. No native compilation, installation or teacher checkout in recovery.',
        backup='Private kernel-source input plus locally verified original ZIP; recovered output persisted and downloaded with SHA/CRC checks. Complete compact DB backup local only; last2 verified.')
    write_json(ROOT/'experiments/task42_gpu_recovery_v1.json',plan)
    print('Pinned60 saved native trajectories and frozen41r; no native regeneration.')
if __name__=='__main__':main()
