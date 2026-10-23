"""Read-only proxy health; never print credential-bearing exception URLs."""
import argparse,json,time
import requests
from colab_profile_cli import configure_profile
configure_profile()
from colab_cli.common import state
ap=argparse.ArgumentParser();ap.add_argument('--session',default='giada-task41h-eval-primary');args=ap.parse_args()
s=state.get_session(args.session)
for path in ('/api/status','/api/kernels'):
    start=time.monotonic()
    try:
        r=requests.get(s.url.rstrip('/')+path,headers={'X-Colab-Runtime-Proxy-Token':s.token},params={'colab-runtime-proxy-token':s.token},timeout=15)
        info={}
        if r.ok and path=='/api/kernels':info={'kernels':[{k:x.get(k) for k in ('id','execution_state','last_activity','connections')} for x in r.json()]}
        print(json.dumps(dict(path=path,status=r.status_code,seconds=time.monotonic()-start,**info)),flush=True)
    except Exception as exc:print(json.dumps(dict(path=path,error=type(exc).__name__,seconds=time.monotonic()-start)),flush=True)
