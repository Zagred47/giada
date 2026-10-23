"""Read-only transport diagnostics; never touches the scientific kernel."""
import json,time
import requests
from colab_profile_cli import configure_profile
configure_profile()
from colab_cli.common import state
s=state.get_session('giada-task41g-primary-r1')
headers={'X-Colab-Runtime-Proxy-Token':s.token}
params={'colab-runtime-proxy-token':s.token}
for path in ['/api/status','/api/kernels','/api/terminals','/api/contents/content/giada_task41g_296acd9956c2/job_status.json']:
    started=time.monotonic()
    try:
        r=requests.get(s.url.rstrip('/')+path,headers=headers,params=params,timeout=15)
        result={'path':path,'status':r.status_code,'seconds':round(time.monotonic()-started,2)}
        if r.ok:
            d=r.json()
            if path.endswith('job_status.json'):
                result['data']=d
            else: result['data']=d
        print(json.dumps(result),flush=True)
    except Exception as e: print(json.dumps({'path':path,'error':type(e).__name__,'seconds':round(time.monotonic()-started,2)}),flush=True)
