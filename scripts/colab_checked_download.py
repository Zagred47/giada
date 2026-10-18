"""Official Contents client, explicit binary format and verified byte count.

The CLI0.7.4 default-format GET may return empty content for a large ZIP while
still printing success. Explicit base64 avoids that ambiguity. Never log tokens.
"""
import argparse
import base64
from pathlib import Path
import json
from colab_cli.common import state
from colab_cli.contents import ContentsClient

def main():
    p=argparse.ArgumentParser();p.add_argument('-s','--session',required=True);p.add_argument('remote');p.add_argument('local',type=Path);a=p.parse_args()
    c=ContentsClient(state.get_session(a.session))
    d=c._request('GET',a.remote,params={'content':'1','format':'base64'})
    if d.get('format')!='base64' or not isinstance(d.get('content'),str):raise RuntimeError('Missing explicit binary contents; refusing false download success')
    raw=base64.b64decode(d['content'])
    if len(raw)!=d.get('size'):raise RuntimeError(f'Download size mismatch: {len(raw)} != {d.get("size")}')
    tmp=a.local.with_name(a.local.name+'.download');tmp.write_bytes(raw);tmp.replace(a.local)
    state.history.log_event(a.session,'file_operation',{'op':'download','remote':a.remote,'local':str(a.local),'bytes_verified':len(raw)})
    print(json.dumps(dict(downloaded=str(a.local),bytes=len(raw),size_verified=True)),flush=True)
if __name__=='__main__':main()
