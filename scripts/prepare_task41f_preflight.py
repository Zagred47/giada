"""Pin committed code for the actual reserve/primary GPU preflight."""
import argparse,hashlib,json,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];folder=ROOT/'experiments/results/task41f_protocol_v1'
parser=argparse.ArgumentParser();parser.add_argument('--profile',choices=['primary','reserve'],default='reserve');args=parser.parse_args()
revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
spec=json.loads((ROOT/'experiments/task41f_preregistration_v1.json').read_text())
for name,sha in spec['sources'].items():
    raw=subprocess.check_output(['git','show',revision+':'+name],cwd=ROOT)
    assert raw==(ROOT/name).read_bytes() and hashlib.sha256(raw).hexdigest()==sha,name
request=dict(revision=revision,profile=args.profile,protocol_sha256=hashlib.sha256((ROOT/'experiments/task41f_preregistration_v1.json').read_bytes()).hexdigest())
(folder/'preflight_request.json').write_text(json.dumps(request,indent=2)+'\n');print(json.dumps(request))
