"""Transport-only assembly of the unchanged original recovery ZIP."""
import hashlib,json
from pathlib import Path
request=json.loads(Path('/content/giada_task41h_launch.json').read_text())
assert request['evaluation_only'] and request['task']=='41h'
path=Path(request['recovery_archive']);parts=[path.with_name(path.name+f'.part{i:04d}') for i in range(4)]
assert all(p.is_file() for p in parts)
temp=path.with_suffix('.assembling');digest=hashlib.sha256()
with temp.open('wb') as target:
    for part in parts:
        with part.open('rb') as stream:
            for chunk in iter(lambda:stream.read(8*1024*1024),b''):digest.update(chunk);target.write(chunk)
expected=next(r['sha256'] for r in request['inputs'] if r['remote']==str(path))
assert digest.hexdigest()==expected
temp.replace(path)
print(json.dumps(dict(valid=True,transport_parts=4,assembled_sha256=digest.hexdigest(),bytes=path.stat().st_size)),flush=True)
