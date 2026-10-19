"""Small independent support-generation check, before any scientific training."""
import json,tempfile
from pathlib import Path
import numpy as np
from src.giada_teacher import task39i_coupled_relaxation as i
ROOT=Path(__file__).resolve().parents[1]
spec=json.loads((ROOT/i.CONFIG).read_text());quality=json.loads((ROOT/i.p.CONFIG).read_text())
quality['families']={'steps':quality['families']['steps']};spec.update(recovery_ticks=320,recovery_nodes=128,windows_per_family=8)
with tempfile.TemporaryDirectory(dir=ROOT/'artifacts') as t:
    out=Path(t)/'run/artifacts';out.mkdir(parents=True)
    i.h.recovery_fit(ROOT,out,spec,quality)
    arrays,anchors=i.independent_history(ROOT,out,spec,quality)
    manifest=json.loads((out/'history_fit_manifest.json').read_text())
    assert all(x.shape==(8,256,71) and np.isfinite(x).all() for x in arrays)
    assert all(((x[...,20:38]>=0)&(x[...,20:38]<=1)).all() for x in arrays)
    assert manifest['valid'] and manifest['disjoint_from_all_original_roles']
    assert anchors.shape==(16,107)
    report=dict(valid=True,FIT_only=True,rows=manifest['files'],anchors_shape=anchors.shape,
        gates_bounded=True,finite=True,source_sha256=i.p.digest(ROOT/'src/giada_teacher/task39i_coupled_relaxation.py'))
    i.p.save_json(ROOT/'experiments/results/task39i_protocol_v1/support_preflight.json',report)
print(json.dumps(dict(valid=True,windows=[list(x.shape) for x in arrays],anchors=list(anchors.shape))))
