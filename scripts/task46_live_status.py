"""Bounded read-only live SSE log check through the installed Kaggle client."""
import json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import task46_kaggle as launch
from research_memory.mirror import write_json
def main():
    from kaggle.api.kaggle_api_extended import KaggleApi
    api=KaggleApi();api.authenticate();events=[];start=time.time();error=None
    try:
        with api.build_kaggle_client() as client:
            http=client._http_client;http._init_session();headers=dict(http._session.headers);headers['Accept']='text/event-stream, */*';headers.pop('Content-Type',None)
            # Same endpoint implemented by KaggleApi.kernels_logs_stream.
            url=http._endpoint+'/v1/kernels/logs/stream/'+launch.k.SLUG
            with http._session.get(url,stream=True,headers=headers,auth=http._session.auth,timeout=(10,10)) as response:
                response.raise_for_status()
                if 'text/event-stream' not in response.headers.get('Content-Type','').lower():
                    raw=response.json();events=raw if isinstance(raw,list) else [raw]
                else:
                    for line in response.iter_lines(chunk_size=1,decode_unicode=True):
                        if time.time()-start>20:break
                        if not line or not line.startswith('data:'):continue
                        payload=line[5:].strip()
                        if payload=='END_OF_LOG':break
                        try:events.append(json.loads(payload))
                        except ValueError:events.append(dict(data=payload))
                        if len(events)>2000:break
    except Exception as ex:error=type(ex).__name__+': '+str(ex).split('\n')[0]
    selected=[q for q in events if isinstance(q,dict) and ('[GIADA' in q.get('data','') or any(w in q.get('data','') for w in ('Traceback','FileExistsError','Error:')))]
    write_json(launch.F/'live_log.json',dict(observed_at=time.time(),events=selected,error=error,read_only=True));print(json.dumps(dict(events=selected[-12:],connection_error=error)))
if __name__=='__main__':main()
