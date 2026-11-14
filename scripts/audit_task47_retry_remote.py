"""Read individual ZIP members by HTTP range; archive SHA remains separate."""
import json,statistics,sys,zipfile
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from retrieve_task47_retry_stream import RemoteZip,launch
from research_memory.mirror import write_json
def main():
    import requests
    from kaggle.api.kaggle_api_extended import KaggleApi,ApiListKernelSessionOutputRequest
    api=KaggleApi();api.authenticate();req=ApiListKernelSessionOutputRequest();req.user_name='alessandrobelli';req.kernel_slug='giada-task47-protocol-retry'
    with api.build_kaggle_client() as client:response=client.kernels.kernels_api_client.list_kernel_session_output(req)
    receipt=json.loads((launch.k.DEST/'kaggle_receipt.json').read_text());item=next(q for q in response.files if q.file_name==receipt['archive'])
    with requests.Session() as s,zipfile.ZipFile(RemoteZip(item.url,receipt['bytes'],s)) as z:
        rows=json.loads(z.read('results.json'));failures=[]
        for row in rows:
            if row['arm']!='P' or row['current_passed'] and row['compartment_passed']:continue
            failed_nodes=[]
            for node in row['per_node']:
                channels=[v for v in node['channels'] if not v['passed']]
                if channels:failed_nodes.append(dict(node={k:v for k,v in node.items() if k!='channels'},channels=channels))
            failures.append(dict(case={k:v for k,v in row.items() if k!='per_node'},failed_nodes=failed_nodes))
        cost=[]
        for row in json.loads(z.read('cost.json')):cost.append(dict(batch=row['batch_per_seed'],scope=row['scope'],median_wall_ms={k:statistics.median(v['wall_ms']) for k,v in row['samples'].items()}))
        components=[dict(geometry=r['geometry'],median_wall_ms={n:{k:statistics.median(v) for k,v in scopes.items()} for n,scopes in r['samples_wall_ms'].items()}) for r in json.loads(z.read('coupled_component_cost.json'))]
        result=dict(provisional_until_archive_SHA=True,failures=failures,cost=cost,components=components)
        write_json(launch.F/'remote_audit.json',result);print(json.dumps(result),flush=True)
if __name__=='__main__':main()
