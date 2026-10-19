"""Verify OFF-VM checkpoint evidence before recording a successful dispatch."""
import hashlib,io,json,zipfile
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
DEST=ROOT/'artifacts/task39f_reserve_external'
OUT=ROOT/'experiments/results/task39f_protocol_v1'

def main():
    backup=json.loads((DEST/'external_backup_receipt.json').read_text())
    status=json.loads((DEST/'job_status.json').read_text())
    request=json.loads((OUT/'dispatch_request.json').read_text())
    assert status['state']=='running' and status['revision']==request['revision']
    archive=Path(backup['local']);assert backup['valid'] and backup['runtime_independent']
    sha=hashlib.sha256(archive.read_bytes()).hexdigest();assert sha==backup['remote']['sha256']
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        execution=json.loads(z.read('execution.json'));assert execution['valid'] and execution['members']==24 and execution['optimizer_state_lifetime_retained']
        checkpoints=[n for n in z.namelist() if n.startswith('checkpoint_') and n.endswith('.pt')]
        assert checkpoints,'No weights in external archive yet'
        chosen=max(checkpoints,key=lambda n:int(n.split('_')[1].split('.')[0]));step=int(chosen.split('_')[1].split('.')[0])
        finite={}
        for name in (chosen,f'optimizer_{step}.pt'):
            obj=torch.load(io.BytesIO(z.read(name)),map_location='cpu',weights_only=False);tensors=[]
            def walk(x):
                if torch.is_tensor(x):tensors.append(x)
                elif isinstance(x,dict):
                    for v in x.values():walk(v)
                elif isinstance(x,(tuple,list)):
                    for v in x:walk(v)
            walk(obj);finite[name]=dict(tensors=len(tensors),finite=all(bool(torch.isfinite(t).all()) for t in tensors));assert tensors and finite[name]['finite']
        assert json.loads(z.read('preregistration.json'))==json.loads((ROOT/'experiments/task39f_preregistration_v1.json').read_text())
    def save(name,data):(OUT/name).write_text(json.dumps(data,indent=2)+'\n',encoding='utf-8')
    save('external_recovery_verified.json',dict(valid=True,archive=archive.relative_to(ROOT).as_posix(),sha256=sha,bytes=archive.stat().st_size,runtime_independent=True,zip_crc_valid=True,checkpoint_step=step,numerical_verification=finite,original_receipt=backup))
    save('launch.json',dict(task='39f',revision=request['revision'],session='giada-task39f-reserve',backend='gpu-t4-s-kkb-usw4a1-bk1b6qc2quya',account_profile='reserve',account='cinziamrg48@gmail.com',gpu=execution['gpu'],torch=execution['torch'],job_status=status,execution=execution,external_destination=DEST.relative_to(ROOT).as_posix(),thresholds_changed=False,scientific_result_claimed=False))
    print(json.dumps(dict(valid=True,step=step,finite=finite,gpu=execution['gpu'],projected_training_minutes=execution['projected_training_seconds']/60)),flush=True)
if __name__=='__main__':main()
