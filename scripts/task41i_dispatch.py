"""Pin preflight and launch to committed source and SQLite preregistration."""
import argparse,hashlib,io,json,subprocess,sys,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from research_memory.mirror import write_json

def sha(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['preflight','dispatch','verify-launch']);ap.add_argument('--profile',default='primary',choices=['primary','reserve']);args=ap.parse_args()
    folder=ROOT/'experiments/results/task41i_protocol_v1';spec=json.loads((ROOT/'experiments/task41i_preregistration_v1.json').read_text())
    if args.mode=='preflight':
        rev=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
        for path,expected in spec['sources'].items():
            data=subprocess.check_output(['git','show',rev+':'+path],cwd=ROOT);assert data==(ROOT/path).read_bytes() and hashlib.sha256(data).hexdigest()==expected,path
        value=dict(revision=rev,profile=args.profile,protocol_sha256=sha(ROOT/'experiments/task41i_preregistration_v1.json'))
        write_json(folder/'preflight_request.json',value)
    elif args.mode=='dispatch':
        pf=json.loads((folder/'gpu_preflight.json').read_text());receipt=json.loads((folder/'memory_receipt.json').read_text())
        assert pf['valid'] and pf['auxiliary_update_executed'] and pf['whole_chain_compile_tested'] and receipt['valid'] and receipt['registration_before_scientific_run']
        assert pf['protocol_sha256']==sha(ROOT/'experiments/task41i_preregistration_v1.json')
        from research_memory.mirror import Mirror
        row=Mirror().query("SELECT [Hash preregistrazione] h FROM protocols WHERE [Codice stabile]='protocols-giada-task41i-factorial-v1'")['rows'];assert row[0]['h']==pf['protocol_sha256']
        value=dict(task='41i',revision=pf['revision'],profile=args.profile,inputs=[])
        for key,path in [('worker_sha256','scripts/run_task41i_selective_repair.py'),('manager_sha256','scripts/task41b_console_manager.py')]:
            data=subprocess.check_output(['git','show',pf['revision']+':'+path],cwd=ROOT);assert data==(ROOT/path).read_bytes();value[key]=hashlib.sha256(data).hexdigest()
        for path in ('artifacts/task41b_opened_inputs.zip','artifacts/task41d_opened41c.zip'):
            value['inputs'].append(dict(remote='/content/'+Path(path).name,sha256=sha(ROOT/path)))
        assert not (folder/'dispatch_request.json').exists();write_json(folder/'dispatch_request.json',value)
    else:
        import torch
        request=json.loads((folder/'dispatch_request.json').read_text());ext=ROOT/f'artifacts/task41i_{args.profile}_external'
        receipt=json.loads((ext/'external_backup_receipt.json').read_text());status=json.loads((ext/'job_status.json').read_text())
        assert status['revision']==request['revision'] and receipt['valid'] and receipt['runtime_independent']
        archive=Path(receipt['local']);assert sha(archive)==receipt['remote']['sha256']
        with zipfile.ZipFile(archive) as z:
            assert z.testzip() is None
            names=[n for n in z.namelist() if n.startswith('checkpoint_') and n.endswith('.pt')];assert names
            name=max(names,key=lambda n:int(n.split('_')[1].split('.')[0]));state=torch.load(io.BytesIO(z.read(name)),map_location='cpu',weights_only=False)
            def finite(value):
                if torch.is_tensor(value):return bool(torch.isfinite(value).all())
                if isinstance(value,dict):return all(finite(x) for x in value.values())
                if isinstance(value,(list,tuple)):return all(finite(x) for x in value)
                return True
            assert finite(state)
        value=dict(valid=True,request=request,status_observed=status,external_backup=receipt,last_recovered_checkpoint=name,
                   all_recovered_tensors_finite=True,scientific_result_pending=True,thresholds_changed=False,DB_backup='local only')
        write_json(folder/'launch.json',value)
    print(json.dumps(value),flush=True)
if __name__=='__main__':main()
