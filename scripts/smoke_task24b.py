"""Reduced CPU pipeline check only, never registered as scientific evidence."""
from pathlib import Path
import json
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.giada_teacher.sodium_control_diagnosis import config,run
cfg=config(ROOT);cfg.update(require_cuda=False,pool_size=128,batch_size=16,widths=[16],checkpoints=[0,1],rollout_horizons_steps=[1,10])
out=Path(sys.argv[1]);out.mkdir(exist_ok=False)
r=run(ROOT,out,cfg,'cpu-smoke-only')
assert r['valid'] and not r['fresh_used_for_selection'] and not r['old_fresh_used_for_selection']
assert not r['oracle_probes_selectable'] and len(r['learned'])==18
print(json.dumps(dict(valid=True,scope='Reduced CPU pipeline only, not scientific success')))
