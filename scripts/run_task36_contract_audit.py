"""Execute the documentary/source reconciliation of the NEW roadmap Task36."""
import ast
import hashlib
import json
from pathlib import Path
import re
import statistics
import subprocess

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'experiments/results/task36_ionic_contract_v1'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def inventory_eval(node):
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        return inventory_eval(node.left) + inventory_eval(node.right)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult):
        return inventory_eval(node.left) * inventory_eval(node.right)
    return ast.literal_eval(node)


def main():
    specpath = ROOT / 'experiments/task36_ionic_abstraction_contract_v1.json'
    spec = read(specpath)
    tree = ast.parse((ROOT / 'src/giada_teacher/ionic_block_teacher_forced.py').read_text())
    literals = {n.targets[0].id: inventory_eval(n.value) for n in tree.body
                if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)
                and n.targets[0].id in ('CHANNELS', 'STATE_NAMES')}
    assert list(literals['CHANNELS']) == spec['channels']
    # STATE_NAMES uses tuple multiplication; inspect its exact AST through a
    # restricted expression evaluator rather than importing scientific runtimes.
    assert sum(map(len, literals['STATE_NAMES'])) == 18
    assert [list(x) for x in literals['STATE_NAMES']] == spec['gate_names']
    prefix = (ROOT / 'src/giada_teacher/ie_residual_cost.py').read_text()
    assert prefix.index('next_ca =') < prefix.index('inf = 1/(1+(.00043/next_ca') < prefix.index('next_sk =')
    assert 'g[..., OFFSETS[k]].square()' in prefix and 'for k in (0,1)' in prefix
    p7root = ROOT / 'experiments/results/ie_p7_kaggle_v4'
    audit, p7 = read(p7root / 'result_audit.json'), read(p7root / 'final_report.json')
    assert audit['valid'] and audit['core_cells'] == audit['operational_cells'] == 56
    for name, info in audit['files'].items():
        path = p7root / name
        assert hashlib.sha256(path.read_bytes()).hexdigest() == info['sha256']
    assert len(spec['hypotheses']) == 7 and len(spec['evaluation_requirements']) == 8
    roadmap = (ROOT / 'experiments/giada_modular_surrogate_roadmap.md').read_text(encoding='utf-8')
    task_titles = {int(n): title for n, title in re.findall(r'^- \*\*Task (\d+) — (.*?)\*\*', roadmap, re.M)}
    assert all(n in task_titles for n in range(36, 86))
    sources = list(spec['evidence_documents']) + [
        'experiments/giada_modular_surrogate_roadmap.md',
        'experiments/task36_ionic_abstraction_contract_v1.json',
        'experiments/task36_ionic_abstraction_contract_v1.md',
        'src/giada_teacher/ionic_block_teacher_forced.py',
        'src/giada_teacher/ie_discrete_ionic.py',
        'src/giada_teacher/ie_residual_cost.py',
        'src/giada_teacher/ie_readout_comparison.py',
        'src/neuronio/neuronio_rapid_fire_eval_utils.py',
        'src/neuronio/neuronio_eval_utils.py',
        'scripts/audit_ie_p7_result.py', 'scripts/run_task36_contract_audit.py']
    sources += [b['checkpoint'] for b in spec['baselines'] if 'checkpoint' in b]
    for b in spec['baselines']:
        if 'checkpoint' in b:
            assert hashlib.sha256((ROOT / b['checkpoint']).read_bytes()).hexdigest() == b['sha256']
    index = {rel: dict(sha256=hashlib.sha256((ROOT / rel).read_bytes()).hexdigest(), bytes=(ROOT / rel).stat().st_size) for rel in sources}
    stage_rows = [x for x in p7['operational']['rows'] if x['batch'] == 642 and x['mode'] == 'cuda_graph']
    formula = next(x for x in stage_rows if x['name'] == 'formula_vectorized')
    profile = {key: statistics.median(s[key] for s in formula['samples']) for key in
               ('wall_ms', 'rng_events_rpc_ms', 'h2d_reset_ms', 'rollout_ms', 'd2h_ms')}
    report = dict(valid=True, schema_version=spec['schema_version'], task=36,
        title=spec['title'], mode=spec['mode'], training_performed=False,
        new_test_accessed=False, new_native_simulation_performed=False,
        code_revision_at_execution=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        original_task36_current_number=72, channels=spec['channels'], physical_gate_count=18,
        voltage_dependent_gate_count=17, sk_gate_index=16,
        phase_order_verified_in_existing_source=True, sources=index,
        ie_p7_audit=audit['code_revision'], ie_p7_primary=audit['primary'],
        ie_p7_native_behavior_support=audit['native_behavior_support'],
        ie_p7_formula_operational_stage_medians_ms=profile,
        baseline_workload_limits='IE-P7 resident core is not isolated ionic latency; operational host producer/RPC is implementation-specific. No new profiling performed in Task36.',
        open_evaluation_gaps=spec['future_run_gate']['task37_required_before_training'],
        hypotheses_registered_as_open=[h['id'] for h in spec['hypotheses']],
        evaluation_specifications=[e['id'] for e in spec['evaluation_requirements']],
        roadmap_actions={str(k): v for k, v in task_titles.items() if 36 <= k <= 85},
        historical_number_mapping={str(k): k + 36 for k in range(36, 50)},
        decision='GO_TASK37_PREPARATION_ONLY', task37_training_authorized=False,
        candidate_promoted=False,
        completion_scope='Task36 source/evidence contract audit completed; SQLite registration/verify/packing reported separately in memory receipt.')
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'contract_audit.json').write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(json.dumps(dict(valid=True, task=36, sources=len(index), hypotheses=7, evaluation_specs=8,
                          decision=report['decision'], training_performed=False)))


if __name__ == '__main__':
    main()
