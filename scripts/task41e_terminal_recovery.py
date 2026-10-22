"""Non-interrupting terminal transport for recovery when Contents returns 500.

Run with the installed WSL Colab CLI Python. No kernel execution or signals.
Only reads remote files; writes recovered copies and receipts locally.
"""
import argparse
import base64
import hashlib
import json
import os
import re
from pathlib import Path
import shlex
import time
import urllib.parse
import uuid
import zipfile

import requests
import websocket
from colab_profile_cli import configure_profile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--terminal', default='1')
    parser.add_argument('--remote')
    parser.add_argument('--local', type=Path)
    parser.add_argument('--inspect', action='store_true')
    parser.add_argument('--delta', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    offset = (args.local.with_suffix(args.local.suffix + '.partial').stat().st_size
              if args.resume and args.local and args.local.with_suffix(args.local.suffix + '.partial').exists() else 0)
    os.environ['GIADA_COLAB_PROFILE'] = 'reserve'
    configure_profile()
    from colab_cli.common import state
    s = state.get_session('giada-task41e-reserve')
    params = {'colab-runtime-proxy-token': s.token}
    if args.terminal == 'new':
        response = requests.post(s.url.rstrip('/') + '/api/terminals', params=params,
            headers={'X-Colab-Runtime-Proxy-Token': s.token}, json={}, timeout=15)
        response.raise_for_status()
        args.terminal = response.json()['name']
    url = s.url.rstrip('/').replace('https://', 'wss://')
    url += '/terminals/websocket/' + args.terminal + '?' + urllib.parse.urlencode(params)
    ws = websocket.create_connection(url, timeout=30,
        header=['X-Colab-Runtime-Proxy-Token: ' + s.token])
    root = '/content/giada_task41e_a1036644b416'
    if args.delta:
        folder = Path(__file__).resolve().parents[1]/'artifacts/task41e_reserve_external'
        receipt = json.loads((folder/'external_backup_receipt.json').read_text())
        with zipfile.ZipFile(folder/Path(receipt['local'].replace('\\','/')).name) as z:
            cached = {i.filename:[i.file_size,i.CRC] for i in z.infolist()}
        code = """import json,zipfile,pathlib,hashlib,shutil
r=pathlib.Path(ROOT);cached=CACHED
with zipfile.ZipFile(r/'recovery_final.zip') as z:
 manifest=[{'name':i.filename,'size':i.file_size,'crc':i.CRC} for i in z.infolist()]
 missing=[i for i in z.infolist() if cached.get(i.filename)!=[i.file_size,i.CRC]]
 with zipfile.ZipFile(r/'recovery_delta.zip','w',compression=zipfile.ZIP_DEFLATED,compresslevel=1) as out:
  for i in missing:
   with z.open(i) as src,out.open(i.filename,'w',force_zip64=True) as dst:shutil.copyfileobj(src,dst,1024*1024)
p=r/'recovery_delta.zip'
print(json.dumps({'status':json.loads((r/'job_status.json').read_text()),'manifest':manifest,'delta':{'bytes':p.stat().st_size,'sha256':hashlib.file_digest(p.open('rb'),'sha256').hexdigest(),'files':[i.filename for i in missing]}}),flush=True)
""".replace('ROOT',repr(root)).replace('CACHED',repr(cached))
    elif args.inspect:
        code = """import json,pathlib,os
r=pathlib.Path(ROOT)
print(json.dumps({'status':json.loads((r/'job_status.json').read_text()),'files':[{'path':str(p.relative_to(r)),'bytes':p.stat().st_size} for p in list((r/'artifacts').glob('*'))+list(r.glob('*.zip')) if p.is_file()]}))
""".replace('ROOT', repr(root))
    else:
        if not args.remote or not args.local:
            parser.error('--remote and --local required')
        if not args.remote.startswith(root + '/'):
            parser.error('remote must be inside the exact Task 41e root')
        code = """import base64,pathlib,hashlib,json
p=pathlib.Path(REMOTE)
with p.open('rb') as f:
 f.seek(OFFSET)
 while True:
  b=f.read(49152)
  if not b:break
  print(base64.b64encode(b).decode(),flush=True)
print('RECEIPT '+json.dumps({'bytes':p.stat().st_size,'sha256':hashlib.file_digest(p.open('rb'),'sha256').hexdigest()}),flush=True)
""".replace('REMOTE', repr(args.remote)).replace('OFFSET',str(offset))
    encoded = base64.b64encode(code.encode()).decode()
    marker = uuid.uuid4().hex
    begin, end_marker = 'GIADA_BEGIN_' + marker, 'GIADA_END_' + marker
    command = "stty -echo; printf '\\n" + begin + "\\n'; python -c "
    command += shlex.quote("import base64;exec(base64.b64decode(" + repr(encoded) + "))")
    command += "; printf '\\n" + end_marker + "\\n'\n"
    ws.send(json.dumps(['stdin', command]))
    buffer = ''; active = False; receipt = None; count = 0; digest = hashlib.sha256()
    output = None
    if args.local:
        args.local.parent.mkdir(parents=True, exist_ok=True)
        partial=args.local.with_suffix(args.local.suffix + '.partial')
        if offset:
            with partial.open('rb') as src:
                for block in iter(lambda:src.read(8*1024*1024),b''):
                    digest.update(block);count+=len(block)
        output = partial.open('ab' if offset else 'wb')
    started = time.monotonic()
    try:
        done = False
        while not done:
            msg = json.loads(ws.recv())
            if msg[0] != 'stdout':
                continue
            buffer += msg[1].replace('\r', '')
            while '\n' in buffer:
                line, buffer = buffer.split('\n', 1)
                line = re.sub(r'\x1b\[[0-?]*[ -/]*[@-~]', '', line)
                if line == begin:
                    active = True; continue
                if line == end_marker:
                    done = True; break
                if not active or not line:
                    continue
                if args.inspect or args.delta:
                    if args.delta:
                        json.loads(line)
                        (folder/'final_delta_manifest.json').write_text(line)
                    print(line, flush=True)
                elif line.startswith('RECEIPT '):
                    receipt = json.loads(line[8:])
                else:
                    try:
                        raw = base64.b64decode(line, validate=True)
                    except Exception:
                        raise RuntimeError('Unexpected terminal output: ' + repr(line[:200])) from None
                    output.write(raw); digest.update(raw); count += len(raw)
        if output:
            output.close()
            assert receipt and count == receipt['bytes'] and digest.hexdigest() == receipt['sha256']
            args.local.with_suffix(args.local.suffix + '.partial').replace(args.local)
            print(json.dumps({'local':str(args.local),'remote':args.remote,'verified':True,
                              **receipt,'seconds':round(time.monotonic()-started,2)}),flush=True)
    finally:
        if output and not output.closed:
            output.close()
        ws.close()


if __name__ == '__main__':
    main()
