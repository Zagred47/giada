"""Preserve the technical Jacobian fix and prove the scientific plan is equal."""
import hashlib
import json
from pathlib import Path
import subprocess
ROOT=Path(__file__).resolve().parents[1]
folder=ROOT/'experiments/results/task41e_protocol_v1'
before_revision='20d5986736d12879091e631ecaa6b5b22d29b7f2'
name='experiments/task41e_preregistration_v1.json'
raw_before=subprocess.check_output(['git','show',before_revision+':'+name],cwd=ROOT)
raw_after=(ROOT/name).read_bytes()
before,after=json.loads(raw_before),json.loads(raw_after)
unchanged={k:v for k,v in before.items() if k!='sources'}=={k:v for k,v in after.items() if k!='sources'}
changed=[k for k in before['sources'] if before['sources'][k]!=after['sources'][k]]
assert unchanged and changed==['src/giada_teacher/task41e_transfer_evaluation.py']
log=ROOT/'artifacts/task41e_evaluation_smoke_attempt1.log'
assert log.exists() and 'Inference tensors cannot be saved for backward' in log.read_text()
receipt=dict(valid=True,kind='pretraining_technical_amendment',before_revision=before_revision,
             before_protocol_sha256=hashlib.sha256(raw_before).hexdigest(),after_protocol_sha256=hashlib.sha256(raw_after).hexdigest(),
             unchanged_scientific_contract=unchanged,changed_sources=changed,
             reason='Input Jacobian reused readout constants created under inference_mode; clone identical ordinary tensors before autograd.',
             thresholds_changed=False,scientific_training_performed_before_fix=False,
             synthetic_optimizer_preflight_updates_discarded=True,
             failure_log_sha256=hashlib.sha256(log.read_bytes()).hexdigest(),
             old_protocol=before,new_protocol=after)
(folder/'pretraining_amendment.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps({k:v for k,v in receipt.items() if k not in ('old_protocol','new_protocol')}))
