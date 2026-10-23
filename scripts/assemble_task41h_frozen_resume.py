"""Remote transport-only assembly; run before the scientific manager."""
import hashlib,json,shutil
from pathlib import Path
request=json.loads(Path('/content/giada_task41h_launch.json').read_text())
for item in request['inputs']:
    target=Path(item['remote'])
    with target.open('wb') as out:
        for part in item['parts']:
            path=Path(part['remote'])
            with path.open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==part['sha256']
            with path.open('rb') as f:shutil.copyfileobj(f,out)
    with target.open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==item['sha256']
    print('[GIADA frozen resume] verified input '+target.name,flush=True)
