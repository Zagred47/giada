"""Second independent CLI call verifies persisted worker state and artifact."""
import hashlib
import json
from pathlib import Path
root=Path('/content/giada_console_contract_probe')
status=json.loads((root/'status.json').read_text())
assert status=={'state':'finished','exit_code':7,'expected_test_failure':True},status
log=(root/'worker.log').read_text()
assert 'WORKER_STARTED' in log and 'EXPECTED_FAILURE_EXIT_7' in log
print(json.dumps({'valid':True,'status':status,'log':log,
                  'archive_sha256':hashlib.sha256((root/'transfer_probe.zip').read_bytes()).hexdigest()}),flush=True)
