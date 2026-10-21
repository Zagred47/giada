"""Install the preregistered native runtime and verify it without fresh data."""
import json,subprocess,sys
from pathlib import Path
r=subprocess.run([sys.executable,'-m','pip','install','--quiet','neuron==8.2.7'],text=True,capture_output=True)
assert r.returncode==0,r.stderr[-3000:]
code="import json,sys,shutil,neuron,numpy;from neuron import h;h.CVode().active(0);print(json.dumps(dict(neuron=neuron.__version__,numpy=numpy.__version__,python=sys.version,nrnivmodl=shutil.which('nrnivmodl'),cvode_available=True)))"
r=subprocess.run([sys.executable,'-c',code],text=True,capture_output=True);assert r.returncode==0,r.stderr[-3000:]
report=json.loads(r.stdout.strip().splitlines()[-1]);assert report['neuron'].split('+')[0]=='8.2.7' and report['nrnivmodl']
report.update(valid=True,fresh_confirmation_accessed=False)
Path('/content/task41_environment_preflight.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
