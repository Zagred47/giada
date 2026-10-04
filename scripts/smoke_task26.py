"""Reduced CPU end-to-end integration; not scientific confirmation."""
import json
import argparse
import sys
import tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.giada_teacher.joint_heterogeneous_comparison import config,run
p=argparse.ArgumentParser();p.add_argument('--native-root');args=p.parse_args()
cfg=config(ROOT)
cfg.update(require_cuda=False,widths=[4],checkpoints=[0,1,2],decay_start=0,pool_size=64,batch_size=16,
           development_count=24,sample_count=24,extrema_nodes=3,extrema_node_offset=.21,
           held_horizons_ms=[1,10],path_dt_ms={'fast':.025},fresh_seeds=[269921,269922,269923,269925])
cfg['data_seeds']={**cfg['data_seeds'],'fit':269901,'fit_regions':269902,'minibatches':269903,'development':[269911,269912],'held_development':269915}
with tempfile.TemporaryDirectory(prefix='task26-smoke-') as d:
    out=Path(d);(out/'native_audit.json').write_text('{"valid":true}')
    report=run(ROOT,out,cfg,'preflight-not-scientific')
    assert report['valid'] and len(report['learned'])==6 and not report['fresh_used_for_selection']
    assert not report['current_supervision_used'] and not report['gate_d_completed']
    freeze=json.loads((out/'selection_freeze.json').read_text());assert not freeze['fresh_accessed']
    assert len(json.loads((out/'development_ladder.json').read_text()))==18
    dest=ROOT/'experiments/preflight/task26_joint';dest.mkdir(parents=True,exist_ok=True)
    (dest/'cpu_integration.json').write_text(json.dumps(dict(valid=True,scientific_confirmation=False,rows=6,scope='CPU reduced orchestration; genuine native audit performed separately; no fresh2604xx used.'),indent=2)+'\n')
    if args.native_root:
        native=json.loads((Path(args.native_root)/'native_audit.json').read_text());assert native['valid']
        (dest/'native_audit.json').write_text(json.dumps(native,indent=2)+'\n')
    print(json.dumps(dict(valid=True,rows=6)))
