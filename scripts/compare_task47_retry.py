"""Audit only the corrected47 protocol run, never the misrouted original."""
import inspect,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import compare_task46 as prior
from research_memory.mirror import write_json
source=inspect.getsource(prior.main).replace('artifacts/task46_kaggle_external','artifacts/task47_retry_external').replace('results/task46_protocol_v1','results/task47_retry_v1')
exec(compile(source,'corrected47audit','exec'),prior.__dict__)
if __name__=='__main__':
    prior.main()
    path=prior.ROOT/'experiments/results/task47_retry_v1/comparison.json';r=json.loads(path.read_text())
    r['limits']='Same preregistered independent47 protocol after technical routing correction. Original misrouted47 attempt excluded from confirmation. Frozen P and unchanged gates; native-invalid/event-unsupported cases explicit. Quality/cost separate; no45ecompletion,48authorization,release or speedup claim.'
    write_json(path,r)
