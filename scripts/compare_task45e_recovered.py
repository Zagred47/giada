"""Merged45e audit and all-arm45d context; never compare incomplete denominators."""
import inspect,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import compare_task45d_recovered as previous
code=inspect.getsource(previous.main).replace('task45d_evaluation_external','task45e_evaluation_external').replace('task45d_evaluation_v1','task45e_evaluation_v1')
exec(compile(code,'45e_recovery_same_paired_comparator','exec'),previous.__dict__)
def main():
    previous.main();path=previous.ROOT/'experiments/results/task45e_evaluation_v1/comparison.json';report=json.loads(path.read_text())
    if report['complete']:
        measured=set.intersection(*(set(q['median_wall_ms']) for q in report['cost'])) if report['cost'] else set()
        report['individual_cost_not_measured']=[a for a in ('P','N00','N01','N10','N11','Iz','Iu','R41','R111') if a not in measured]
        old=json.loads((previous.ROOT/'experiments/results/task45d_evaluation_v1/comparison.json').read_text());assert old['complete']
        report['all45d_arms']=old
        report['cross_task_limits']='Same geometry/case/seed/prefix/evaluator, frozenC1 control counts must reproduce.45dNcap/Nline startf011;45eP transplants onlyNap ontoC1. Training arms are not one-to-one architecture matches. Use within-run speed ratios; no independent release claim.'
        receipt=json.loads((previous.ROOT/'artifacts/task45e_kaggle_external/external_backup_receipt.json').read_text())
        import zipfile
        with zipfile.ZipFile(receipt['local']) as z:
            freeze=json.loads(z.read('nomination_freeze.json'));report['frozen_nomination']=freeze['chosen'];report['fresh_DEV_choices']=freeze['per_arm_choices']
        previous.write_json(path,report)
if __name__=='__main__':main()
