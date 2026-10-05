"""Small CPU integration preflight without opening Task27 fresh seeds."""
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.giada_teacher.current_supervision_comparison import config,run
from src.giada_teacher.hh_family_transfer import write
from scripts.audit_task27_result import audit

root=Path(__file__).resolve().parents[1]
cfg=config(root)
cfg.update(require_cuda=False,checkpoints=[0,1,2],pool_size=64,batch_size=16,
           development_count=16,sample_count=16,extrema_nodes=2,
           dt_values_ms=[.025,1],held_horizons_ms=[1,2],path_dt_ms={'fast':.025})
cfg['fresh_seeds']=[270991,270992,270993,270994]
with tempfile.TemporaryDirectory(prefix='giada_task27_cpu_') as location:
    output=Path(location)
    write(output/'native_audit.json',{'valid':True,'preflight_only':True})
    result=run(root,output,cfg,'cpu-preflight')
    assert result['valid'] and len(result['learned'])==24
    assert not result['fresh_used_for_selection']
    assert json.loads((output/'selection_freeze.json').read_text())['fresh_accessed'] is False
    write(output/'process_status.json',{'phase':'gpu','returncode':0})
    write(output/'code_provenance.json',{'code_revision':'cpu-preflight','dirty_runtime':False})
    write(output/'run_contract.json',cfg)
    assert audit(output)['valid']
    print(json.dumps({'valid':True,'rows':len(result['learned']),
                      'freeze':(output/'selection_freeze.json').exists()}))
