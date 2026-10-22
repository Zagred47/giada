"""Resumable read-only file transport over a separate Jupyter terminal.

Fallback for Contents500; never executes in or interrupts the training kernel.
Authentication stays inside the existing official Colab CLI profile.
"""
import argparse,base64,hashlib,json,os,re,shlex,time,urllib.parse,uuid
from pathlib import Path
import requests
import websocket
from colab_profile_cli import configure_profile


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--session',required=True);parser.add_argument('--root',required=True)
    parser.add_argument('remote');parser.add_argument('local',type=Path);args=parser.parse_args()
    assert re.fullmatch(r'/content/(?:giada_task41[fg]_[0-9a-f]{12}|giada41[fg]_preflight_[0-9]{16,22})',args.root)
    assert args.remote.startswith(args.root+'/') and '..' not in Path(args.remote).parts
    configure_profile()
    from colab_cli.common import state
    session=state.get_session(args.session)
    headers={'X-Colab-Runtime-Proxy-Token':session.token};params={'colab-runtime-proxy-token':session.token}
    response=requests.post(session.url.rstrip('/')+'/api/terminals',headers=headers,params=params,json={},timeout=20)
    response.raise_for_status();terminal=response.json()['name']
    url=session.url.rstrip('/').replace('https://','wss://')+'/terminals/websocket/'+terminal+'?'+urllib.parse.urlencode(params)
    ws=websocket.create_connection(url,timeout=45,header=['X-Colab-Runtime-Proxy-Token: '+session.token])
    partial=args.local.with_suffix(args.local.suffix+'.terminal.partial')
    immutable=args.remote.endswith('.zip')
    offset=partial.stat().st_size if immutable and partial.exists() else 0
    code="""import pathlib,base64,hashlib,json
p=pathlib.Path(REMOTE);digest=hashlib.sha256();count=0
with p.open('rb') as stream:
 while True:
  block=stream.read(49152)
  if not block:break
  digest.update(block)
  end=count+len(block)
  if end>OFFSET:print(base64.b64encode(block[max(0,OFFSET-count):]).decode(),flush=True)
  count=end
print('RECEIPT '+json.dumps({'bytes':count,'sha256':digest.hexdigest()}),flush=True)
""".replace('REMOTE',repr(args.remote)).replace('OFFSET',str(offset))
    marker=uuid.uuid4().hex;begin='GIADA_BEGIN_'+marker;end='GIADA_END_'+marker
    command="stty -echo; printf '\\n"+begin+"\\n'; python -c "+shlex.quote('import base64;exec(base64.b64decode('+repr(base64.b64encode(code.encode()).decode())+'))')+"; printf '\\n"+end+"\\n'\n"
    args.local.parent.mkdir(parents=True,exist_ok=True)
    digest=hashlib.sha256();count=0
    if offset:
        with partial.open('rb') as stream:
            for chunk in iter(lambda:stream.read(8*1024*1024),b''):digest.update(chunk);count+=len(chunk)
    started=time.monotonic();buffer='';active=False;receipt=None;done=False
    try:
        ws.send(json.dumps(['stdin',command]))
        with partial.open('ab' if offset else 'wb') as dest:
            while not done:
                msg=json.loads(ws.recv())
                if msg[0]!='stdout':continue
                buffer+=msg[1].replace('\r','')
                while '\n' in buffer:
                    line,buffer=buffer.split('\n',1);line=re.sub(r'\x1b\[[0-?]*[ -/]*[@-~]','',line)
                    if line==begin:active=True;continue
                    if line==end:done=True;break
                    if not active or not line:continue
                    if line.startswith('RECEIPT '):receipt=json.loads(line[8:]);continue
                    block=base64.b64decode(line,validate=True);dest.write(block);digest.update(block);count+=len(block)
        assert receipt and count==receipt['bytes'] and digest.hexdigest()==receipt['sha256']
        partial.replace(args.local)
        print(json.dumps(dict(valid=True,bytes=count,sha256=digest.hexdigest(),resumed_from=offset,transport='independent_terminal',seconds=time.monotonic()-started)),flush=True)
    finally:
        ws.close()
        try:requests.delete(session.url.rstrip('/')+'/api/terminals/'+terminal,headers=headers,params=params,timeout=10)
        except requests.RequestException:pass


if __name__=='__main__':
    try:main()
    except Exception as exc:
        # Proxy tokens may appear in HTTP exception URLs; never print them.
        print(json.dumps(dict(valid=False,error_type=type(exc).__name__,transport='independent_terminal')),flush=True)
        raise SystemExit(1)
