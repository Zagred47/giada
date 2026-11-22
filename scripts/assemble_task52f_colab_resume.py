import hashlib,json
from pathlib import Path
manifest=json.loads(Path('/content/task52f_transport_manifest.json').read_text())
for item in manifest:
    target=Path(item['remote']);tmp=target.with_suffix(target.suffix+'.assembling')
    with tmp.open('wb') as out:
        for part in item['parts']:
            block=Path(part['remote']).read_bytes();assert hashlib.sha256(block).hexdigest()==part['sha256'];out.write(block)
    with tmp.open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==item['sha256']
    tmp.replace(target)
    for part in item['parts']:Path(part['remote']).unlink()
receipt=dict(valid=True,inputs=[dict(remote=i['remote'],sha256=i['sha256']) for i in manifest])
Path('/content/task52f_input_verified.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt))
