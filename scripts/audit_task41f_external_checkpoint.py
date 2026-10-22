"""Read-only SHA/CRC/finite audit of the latest off-runtime 41f checkpoint."""
import argparse,hashlib,io,json,sys,zipfile
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from research_memory.mirror import write_json


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--profile',choices=['primary','reserve'],default='reserve')
    parser.add_argument('--output',type=Path);args=parser.parse_args()
    folder=ROOT/f'artifacts/task41f_{args.profile}_external'
    receipt=json.loads((folder/'external_backup_receipt.json').read_text())
    request=json.loads((ROOT/'experiments/results/task41f_protocol_v1/dispatch_request.json').read_text())
    status=json.loads((folder/'job_status.json').read_text())
    assert receipt['valid'] and receipt['runtime_independent'] and status['revision']==request['revision']
    archive=Path(receipt['local']);h=hashlib.sha256()
    with archive.open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
    assert h.hexdigest()==receipt['remote']['sha256']
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        names=[n for n in z.namelist() if n.startswith('checkpoint_') and n.endswith('.pt')]
        name=max(names,key=lambda n:int(n.split('_')[1].split('.')[0]))
        state=torch.load(io.BytesIO(z.read(name)),map_location='cpu',weights_only=False)
    assert int(state['step'])==int(name.split('_')[1].split('.')[0])
    tensors=[]
    def walk(v):
        if torch.is_tensor(v):tensors.append(v)
        elif isinstance(v,dict):
            for child in v.values():walk(child)
        elif isinstance(v,(tuple,list)):
            for child in v:walk(child)
    walk(state)
    assert tensors and all(bool(torch.isfinite(v).all()) for v in tensors)
    report=dict(valid=True,checked_utc=datetime.now(timezone.utc).isoformat(),revision=request['revision'],
        checkpoint=name,step=int(state['step']),all_tensors_finite=True,tensor_count=len(tensors),
        archive=str(archive),archive_bytes=archive.stat().st_size,sha256=h.hexdigest(),CRC_valid=True,
        runtime_independent=True,status_observed=status,scientific_result_not_inferred=True)
    if args.output:write_json(args.output,report)
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
