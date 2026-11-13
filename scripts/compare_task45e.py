"""Same paired comparison logic as45d; source paths explicit, thresholds unchanged."""
import inspect,sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import compare_task45d_recovered as previous
code=inspect.getsource(previous.main).replace('task45d_evaluation_external','task45e_kaggle_external').replace('task45d_evaluation_v1','task45e_protocol_v1')
# A45e run performs new preregistered FIT/DEV, unlike45d evaluation recovery.
code=code.replace('selection_unchanged=True,training_repeated=False','selection_unchanged=False,training_repeated=False,new_preregistered_training=True')
exec(compile(code,'45e_same_paired_comparator','exec'),previous.__dict__)
if __name__=='__main__':
    previous.main()
    path=previous.ROOT/'experiments/results/task45e_protocol_v1/comparison.json';report=json.loads(path.read_text())
    if report['complete']:
        measured=set.intersection(*(set(r['median_wall_ms']) for r in report['cost'])) if report['cost'] else set()
        report['individual_cost_not_measured']=[a for a in ('P','N00','N01','N10','N11','Iz','Iu','R41','R111') if a not in measured]
        report['45d_context']='Compare frozenC1 across tasks;45e P changes composition and new fits start atC1. No rewritten45d nomination or retrospective qualification.'
        previous.write_json(path,report)
