"""Prepare exact revision requests and verify actual off-runtime launch evidence."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import zipfile
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from research_memory.mirror import write_json

def digest(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()

def main():
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['preflight','dispatch','verify-launch'])
    parser.add_argument('--profile',choices=['primary','reserve'],default='primary');args=parser.parse_args()
    folder=ROOT/'experiments/results/task41g_protocol_v1'
    spec=json.loads((ROOT/'experiments/task41g_preregistration_v1.json').read_text())
    if args.mode=='preflight':
        revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
        for name,sha in spec['sources'].items():
            raw=subprocess.check_output(['git','show',revision+':'+name],cwd=ROOT)
            assert raw==(ROOT/name).read_bytes() and hashlib.sha256(raw).hexdigest()==sha,name
        value=dict(revision=revision,profile=args.profile,protocol_sha256=digest(ROOT/'experiments/task41g_preregistration_v1.json'))
        write_json(folder/'preflight_request.json',value)
    elif args.mode=='dispatch':
        pf=json.loads((folder/'gpu_preflight.json').read_text())
        assert pf['valid'] and pf['all_parameters_finite'] and pf['whole_loop_4096_parity_passed'] and pf['auxiliary_update_executed']
        assert pf['protocol_sha256']==digest(ROOT/'experiments/task41g_preregistration_v1.json')
        receipt=json.loads((folder/'memory_receipt.json').read_text());assert receipt['valid'] and receipt['registration_before_scientific_run']
        revision=pf['revision'];value=dict(task='41g',revision=revision,profile=args.profile,inputs=[])
        for key,name in [('worker_sha256','scripts/run_task41g_preservation.py'),('manager_sha256','scripts/task41b_console_manager.py')]:
            raw=subprocess.check_output(['git','show',revision+':'+name],cwd=ROOT);assert raw==(ROOT/name).read_bytes()
            value[key]=hashlib.sha256(raw).hexdigest()
        for name in ('artifacts/task41b_opened_inputs.zip','artifacts/task41d_opened41c.zip'):
            value['inputs'].append(dict(remote='/content/'+Path(name).name,sha256=digest(ROOT/name)))
        assert not (folder/'dispatch_request.json').exists();write_json(folder/'dispatch_request.json',value)
    else:
        import torch
        request=json.loads((folder/'dispatch_request.json').read_text())
        ext=ROOT/f'artifacts/task41g_{args.profile}_external'
        receipt=json.loads((ext/'external_backup_receipt.json').read_text());status=json.loads((ext/'job_status.json').read_text())
        assert status['revision']==request['revision'] and receipt['valid'] and receipt['runtime_independent']
        archive=Path(receipt['local']);assert digest(archive)==receipt['remote']['sha256']
        with zipfile.ZipFile(archive) as z:
            assert z.testzip() is None
            names=[n for n in z.namelist() if n.startswith('checkpoint_') and n.endswith('.pt')];assert names
            checkpoint=max(names,key=lambda n:int(n.split('_')[1].split('.')[0]))
            value=torch.load(io.BytesIO(z.read(checkpoint)),map_location='cpu',weights_only=False)
            def finite(v):
                if torch.is_tensor(v):return bool(torch.isfinite(v).all())
                if isinstance(v,dict):return all(finite(x) for x in v.values())
                if isinstance(v,(list,tuple)):return all(finite(x) for x in v)
                return True
            assert finite(value)
        value=dict(valid=True,request=request,status_observed=status,external_backup=receipt,
            last_recovered_checkpoint=checkpoint,all_recovered_tensors_finite=True,thresholds_changed=False,scientific_result_pending=True,DB_backup='localonly')
        write_json(folder/'launch.json',value)
    print(json.dumps(value),flush=True)

if __name__=='__main__':main()
