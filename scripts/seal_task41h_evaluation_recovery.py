"""Seal byte-verified recovery inputs before any new selection evaluation."""
import hashlib,json,re,sys,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from research_memory.mirror import write_json

folder=ROOT/'experiments/results/task41h_recovery_v1';folder.mkdir(exist_ok=True)
receipt=json.loads((ROOT/'artifacts/task41h_reserve_external/external_backup_receipt.json').read_text())
archive=Path(receipt['local'])
with archive.open('rb') as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
assert digest==receipt['remote']['sha256']
with zipfile.ZipFile(archive) as z:
    assert z.testzip() is None
    prereg=z.read('preregistration.json');assert prereg==(ROOT/'experiments/task41h_preregistration_v1.json').read_bytes()
    spec=json.loads(prereg);curves=json.loads(z.read('training_curves.json'))
    assert [r['step'] for r in curves]==spec['checkpoints'][:-1]
    checkpoints={f'checkpoint_{step}.pt':hashlib.sha256(z.read(f'checkpoint_{step}.pt')).hexdigest() for step in spec['checkpoints']}
    original_revision=json.loads(z.read('revision.json'))['revision']
logs=(ROOT/'artifacts/task41h_reserve_external/console.log').read_text()
aggregates=sorted(set(re.findall(r'\baggregate_H[01]_D\d+_N\d+(?:_B128)?',logs)))
assert aggregates==['aggregate_H1_D16_N16384','aggregate_H1_D16_N4096']
seal=dict(valid=True,original_training_revision=original_revision,source_archive_sha256=digest,source_archive_bytes=archive.stat().st_size,
    protocol_sha256=hashlib.sha256(prereg).hexdigest(),checkpoint_sha256=checkpoints,saved_rank_steps=[r['step'] for r in curves],
    missing_rank_step=60000,observed_aggregate_models=aggregates,GPU='Tesla T4',torch_version='2.11.0+cu130',
    training_updates_performed=0,thresholds_changed=False,selection_rule='Original lexicographic minima, strict-less comparison, earliest tie; stratified primary and aggregate diagnostic.',
    source_backup=str(archive),runtime_loss='No active reserve assignment; CLI recorded session_terminated2026-10-23T14:46:59Z during evaluation; no final report recovered.')
write_json(folder/'input_seal.json',seal);print(json.dumps(seal),flush=True)
