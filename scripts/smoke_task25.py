"""End-to-end CPU integration preflight, not scientific confirmation."""
import argparse
import json
import sys
import tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.giada_teacher.heterogeneous_mechanism_composition import run
cfg=json.loads((ROOT/'experiments/task25_heterogeneous_mechanisms.json').read_text())
p=argparse.ArgumentParser();p.add_argument('--native-root');args=p.parse_args()
cfg.update(require_cuda=False,sample_count=32,extrema_nodes=3,extrema_node_offset=.19,
    fresh_seeds=[259901,259902,259903,259905],held_horizons_ms=[1,10],path_dt_ms={'fast':.025})
with tempfile.TemporaryDirectory(prefix='task25-smoke-') as d:
    out=Path(d);(out/'native_audit.json').write_text('{"valid":true}')
    report=run(ROOT,out,cfg,'preflight-not-scientific')
    assert report['valid'] and len(report['learned'])==12 and not report['retraining_performed']
    target=ROOT/'experiments/preflight/task25_heterogeneous';target.mkdir(parents=True,exist_ok=True)
    (target/'cpu_integration.json').write_text(json.dumps(dict(valid=True,scientific_confirmation=False,rows=len(report['learned']),scope='CPU small-support orchestration only; genuine native oracle audited separately.'),indent=2)+'\n')
    if args.native_root:
        native=json.loads((Path(args.native_root)/'native_audit.json').read_text());assert native['valid']
        (target/'native_audit.json').write_text(json.dumps(native,indent=2)+'\n')
    print(json.dumps(dict(valid=True,rows=len(report['learned']))))
