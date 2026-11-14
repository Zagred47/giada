"""Same paired audit definitions as46, explicitly scoped to independent47."""
import inspect,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import compare_task46 as prior
from research_memory.mirror import write_json
source=inspect.getsource(prior.main).replace('artifacts/task46_kaggle_external','artifacts/task47_kaggle_external').replace('results/task46_protocol_v1','results/task47_protocol_v1')
exec(compile(source,'independent47audit','exec'),prior.__dict__)
if __name__=='__main__':
    prior.main()
    import json
    path=prior.ROOT/'experiments/results/task47_protocol_v1/comparison.json';r=json.loads(path.read_text())
    r['limits']='Independent47geometries/input seeds, same frozen ensemble; quality and runtime separate.45e remains incomplete. Native-invalid or unsupported event cases explicitly excluded, no global biological equivalence/release/speedup/48claim. Component timings nonadditive.'
    write_json(path,r)
