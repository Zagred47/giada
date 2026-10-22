"""Pin the committed scientific sources for the actual Colab GPU preflight."""
import hashlib
import argparse
import json
from pathlib import Path
import subprocess
ROOT=Path(__file__).resolve().parents[1]
folder=ROOT/'experiments/results/task41e_protocol_v1'
parser=argparse.ArgumentParser();parser.add_argument('--profile',choices=['primary','reserve'],default='primary');args=parser.parse_args()
revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
spec=json.loads((ROOT/'experiments/task41e_preregistration_v1.json').read_text())
for name,expected in spec['sources'].items():
    committed=subprocess.check_output(['git','show',revision+':'+name],cwd=ROOT)
    assert committed==(ROOT/name).read_bytes() and hashlib.sha256(committed).hexdigest()==expected,name
request=dict(revision=revision,profile=args.profile,protocol_sha256=hashlib.sha256((ROOT/'experiments/task41e_preregistration_v1.json').read_bytes()).hexdigest())
(folder/'preflight_request.json').write_text(json.dumps(request,indent=2)+'\n')
print(json.dumps(request))
