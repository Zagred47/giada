"""Recover only frozen selected calcium checkpoints from verified local ZIP."""
import hashlib
import json
import zipfile
from pathlib import Path
root=Path(__file__).resolve().parents[1]
archive=root/'artifacts/giada_task23_calcium_pair_0f79081_f87cac0a.zip'
dest=root/'experiments/results/task23_kaggle_0f79081'
with zipfile.ZipFile(archive) as z:
    assert z.testzip() is None
    freeze=json.loads(z.read('selection_freeze.json'))
    for family in ('independent','shared_heads'):
        name=freeze['selection'][family]['checkpoint'];raw=z.read(name)
        assert hashlib.sha256(raw).hexdigest()==freeze['checkpoint_hashes'][name]
        target=dest/name
        if target.exists() and target.read_bytes()!=raw:raise RuntimeError('Different checkpoint: '+name)
        target.write_bytes(raw)
print('Verified selected calcium checkpoints recovered; no retraining.')
