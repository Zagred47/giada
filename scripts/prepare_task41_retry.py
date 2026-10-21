"""Preserve the technical failure and verify a measurement-only retry."""
import hashlib,json,subprocess,sys,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from research_memory.mirror import write_json
from src.giada_teacher import task41_independent_confirmation as c
folder=ROOT/'experiments/results/task41_protocol_v1';external=ROOT/'artifacts/task41_primary_external'
backup=json.loads((external/'external_backup_receipt.json').read_text());archive=Path(backup['local'])
assert backup['valid'] and backup['remote']['label']=='final' and c.p.digest(archive)==backup['remote']['sha256']
with zipfile.ZipFile(archive) as z:
    assert z.testzip() is None
    for name in ('failure_report.json','native_interface_report.json','native_calibration.json','candidate_contract.json','preregistration.json'):
        target=folder/('attempt1_'+name)
        if target.exists():assert target.read_bytes()==z.read(name)
        else:target.write_bytes(z.read(name))
    assert not any(k in z.namelist() for k in ('causal_results.json','fresh_stress_results.json','final_report.json'))
probe=json.loads((folder/'synaptic_semantics_probe.json').read_text());assert probe['valid'] and not probe['B6_accessed']
old=json.loads((folder/'attempt1_preregistration.json').read_text());new=json.loads((ROOT/c.CONFIG).read_text())
for key in old:
    if key!='sources':assert old[key]==new[key],key
assert {k for k in old['sources'] if old['sources'][k]!=new['sources'][k]}=={'src/giada_teacher/task41_independent_confirmation.py'}
assert set(new)-set(old)=={'receptor_time_semantics'}
write_json(folder/'execution_amendment.json',dict(valid=True,first_attempt_technical_failure=True,B6_judged=False,
    new_confirmation_labels_used_for_tuning=False,cause='Task41 omitted ems_fixed_step synaptic age flag; continuous event age differs on off-grid events.',
    evidence='Four already opened protocols: shadow current errors ~0.004nA -> <1.3e-16nA; receptor state ~0.0667 -> <3.2e-15; same native trajectory and RNG.',
    change='Explicit existing EMS fixed-step semantics passed through calibration_config. Add isolated GPU smoke before native phase. Same B6, all seeds, inputs, protocols, thresholds and evaluation gates.',
    original_revision='ac24e4a442267bed254521874c2dd16466e0c7d4',first_archive_sha256=backup['remote']['sha256'],
    model_checkpoint_sha256=new['sources'][new['checkpoint']],thresholds_changed=False,scientific_retry_tuning=False))
print('Technical retry amendment verified; no scientific result inferred',flush=True)
