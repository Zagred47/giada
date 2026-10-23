"""Pin preflight and launch to source and a committed SQLite preregistration."""
import argparse,hashlib,io,json,subprocess,sys,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from research_memory.mirror import write_json

def sha(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()

def main():
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['preflight','dispatch','verify-launch']);parser.add_argument('--profile',default='reserve',choices=['primary','reserve']);a=parser.parse_args()
    folder=ROOT/'experiments/results/task41h_protocol_v1';spec=json.loads((ROOT/'experiments/task41h_preregistration_v1.json').read_text())
    if a.mode=='preflight':
        revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
        for name,expected in spec['sources'].items():
            raw=subprocess.check_output(['git','show',revision+':'+name],cwd=ROOT)
            assert raw==(ROOT/name).read_bytes() and hashlib.sha256(raw).hexdigest()==expected,name
        value=dict(revision=revision,profile=a.profile,protocol_sha256=sha(ROOT/'experiments/task41h_preregistration_v1.json'))
        write_json(folder/'preflight_request.json',value)
    elif a.mode=='dispatch':
        pf=json.loads((folder/'gpu_preflight.json').read_text());assert pf['valid'] and pf['auxiliary_update_executed'] and pf['precision_seams_executed']
        assert pf['protocol_sha256']==sha(ROOT/'experiments/task41h_preregistration_v1.json')
        receipt=json.loads((folder/'memory_receipt.json').read_text());assert receipt['valid'] and receipt['registration_before_scientific_run']
        from research_memory.mirror import Mirror
        rows=Mirror().query("SELECT [Hash preregistrazione] h FROM protocols WHERE [Codice stabile]='protocols-giada-task41h-factorial-v1'")['rows']
        assert rows[0]['h']==pf['protocol_sha256']
        value=dict(task='41h',revision=pf['revision'],profile=a.profile,inputs=[])
        for key,name in [('worker_sha256','scripts/run_task41h_path_repair.py'),('manager_sha256','scripts/task41b_console_manager.py')]:
            raw=subprocess.check_output(['git','show',pf['revision']+':'+name],cwd=ROOT);assert raw==(ROOT/name).read_bytes()
            value[key]=hashlib.sha256(raw).hexdigest()
        for name in ('artifacts/task41b_opened_inputs.zip','artifacts/task41d_opened41c.zip'):
            value['inputs'].append(dict(remote='/content/'+Path(name).name,sha256=sha(ROOT/name)))
        assert not (folder/'dispatch_request.json').exists();write_json(folder/'dispatch_request.json',value)
    else:
        import torch
        request=json.loads((folder/'dispatch_request.json').read_text());ext=ROOT/f'artifacts/task41h_{a.profile}_external'
        receipt=json.loads((ext/'external_backup_receipt.json').read_text());status=json.loads((ext/'job_status.json').read_text())
        assert status['revision']==request['revision'] and receipt['valid'] and receipt['runtime_independent']
        archive=Path(receipt['local']);assert sha(archive)==receipt['remote']['sha256']
        with zipfile.ZipFile(archive) as z:
            assert z.testzip() is None
            names=[n for n in z.namelist() if n.startswith('checkpoint_') and n.endswith('.pt')];assert names
            name=max(names,key=lambda n:int(n.split('_')[1].split('.')[0]))
            state=torch.load(io.BytesIO(z.read(name)),map_location='cpu',weights_only=False)
            def finite(v):
                if torch.is_tensor(v):return bool(torch.isfinite(v).all())
                if isinstance(v,dict):return all(finite(x) for x in v.values())
                if isinstance(v,(list,tuple)):return all(finite(x) for x in v)
                return True
            assert finite(state)
        value=dict(valid=True,request=request,status_observed=status,external_backup=receipt,last_recovered_checkpoint=name,
            all_recovered_tensors_finite=True,scientific_result_pending=True,thresholds_changed=False,DB_backup='localonly',
            numerical_scope='State64/solve32 casts the absolute solver output back to64 after FP32 arithmetic; it is an interface-dtype control, not an unquantized64-bit residual accumulator. Do not attribute a null result to the irrelevance of state accumulation. Solver64 and recentered-FP32 contrasts remain diagnostic, not selection-eligible.')
        write_json(folder/'launch.json',value)
    print(json.dumps(value),flush=True)
if __name__=='__main__':main()
