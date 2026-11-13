"""Bind the tested45d evaluation-only engine to the frozen45e protocol.

Only protocol/context and provenance labels change. The rollout, scorer,
skip-completed-block logic and freeze/weight assertions are reused verbatim.
No call to45e.evaluate/fit/choose is permitted; only its registered cost closure
is extracted, as in the original tested recovery engine.
"""
import ast,inspect,json,types
from . import task45d_evaluation_recovery as previous
from . import task45e_anchored_repair as e
d=e.d;c=e.c;n=e.n
engine=types.SimpleNamespace(**e.__dict__)
engine.adapter=d.remap(d.adapter,ARMS=e.ARMS)
engine.crossing_and_gate_probes=d.remap(d.crossing_and_gate_probes,ARMS=e.ARMS)
engine.remap=d.remap;engine.control_check=d.control_check;engine.ORIGINAL_BENCHMARK=d.ORIGINAL_BENCHMARK
def forbidden(*args,**kwargs):raise AssertionError('45e recovery forbids training and selection')
engine.fit=forbidden;engine.choose=forbidden
def load_plan(root):return json.loads((root/'experiments/task45e_evaluation_recovery_v1.json').read_text())
summarize=d.remap(previous.summarize,d=engine)
# Explicit, narrow provenance substitutions; no equation/threshold substitutions.
source=inspect.getsource(previous.recover).replace('GPU_training_transition_preflight.json','GPU_cost_preflight.json').replace('[GIADA45d recovery]','[GIADA45e recovery]').replace("task='45d evaluation recovery'","task='45e evaluation recovery'")
tree=ast.parse(source)
calls={q.func.attr for q in ast.walk(tree) if isinstance(q,ast.Call) and isinstance(q.func,ast.Attribute)}
assert not calls&{'fit','choose','nominate','backward','step','train_loss'}
for dependency in ('copy','hashlib','io','os','zipfile','np'):
    globals()[dependency]=getattr(previous,dependency)
d=engine
exec(compile(tree,'45e_unchanged_evaluation_engine','exec'),globals())
