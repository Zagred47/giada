"""Same46paired audit for the isolated-cache technical retry."""
import inspect,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import compare_task46 as previous
source=inspect.getsource(previous.main).replace('artifacts/task46_kaggle_external','artifacts/task46_retry_external').replace('results/task46_protocol_v1','results/task46_retry_v1')
exec(compile(source,'same46audit_after_cache_fix','exec'),previous.__dict__)
if __name__=='__main__':previous.main()
