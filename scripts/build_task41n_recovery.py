"""Seal recovery metadata from the already verified original41n archive."""
import hashlib,json,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
source=ROOT/'artifacts/task41n_primary_external/recovery_final.zip'
with source.open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()=='7e07984c6c8770fee223a83f4694922c2f8cb92529e3df899bc511cf8d5080be'
names=['preregistration.json','execution_freeze.json','canonical_floors.json','native_phase_floor.json','native_interface_report.json','native_shadow_traces.npz','reference_autonomous_primary.npz','reference_autonomous_voltage_solve64.npz','causal_results.json','fresh_execution_parity.json','frozen_source.zip.bin','cases.json','gpu_preflight.json','revision.json']
with zipfile.ZipFile(source) as z:members={name:hashlib.sha256(z.read(name)).hexdigest() for name in names}
plan=dict(task='41n',version='evaluation-recovery-v1',diagnostic_completion=True,new_independent_confirmation=False,source_local=source.relative_to(ROOT).as_posix(),source_sha256='7e07984c6c8770fee223a83f4694922c2f8cb92529e3df899bc511cf8d5080be',source_bytes=source.stat().st_size,members=members,
    eligible=['dual','donor_pruned','shared_features'],arms=['dual','donor_pruned','shared_features','H_control','dual_fp32'],
    remaining=['autonomous evaluation vs both saved canonical references','declared24x3 imposed stress histories','original ionic and integrated cost at1/642/4096'],
    reused_without_recomputation=['native teacher data','imposed evaluations','original implementation freeze','all four canonical floor reports','native phase/interface/calibration'],
    training_updates=0,selection_recomputed=False,thresholds_changed=False,native_regenerated=False,
    original_primary_confirmation_blocked=True,failed_original_fp32_floor_indices=[218,234],task42_authorized=False,
    rule='Complete missing measurements only. Preserve failed original FP32 canonical floors and original thresholds. Diagnostics may score against both original/matched saved references but cannot reinstate the original confirmation or claim qualified speedup.',
    provider_priority='Kaggle console SDK; current quota21600s available. Colab primary/reserve fallback only if needed.',
    backup='Private persisted Kaggle output plus SHA/CRC-verified local final ZIP; atomic per-arm reports. No repeated whole DB snapshots or duplicate transport chunks; DB backups local payload-v2 bundles.',
    max_seconds=5400)
(ROOT/'experiments/task41n_recovery_v1.json').write_text(json.dumps(plan,indent=2)+'\n',encoding='utf-8')
print('Sealed41n recovery input and unchanged evaluation scope')
