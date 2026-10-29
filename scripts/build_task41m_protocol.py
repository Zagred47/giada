"""Seal an intentionally small frozen diagnostic, not another training search."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def main():
    old=json.loads((ROOT/'experiments/task41l_preregistration_v1.json').read_text())
    audit=json.loads((ROOT/'experiments/results/task41l_audited_v1/audit.json').read_text())
    reuse=json.loads((ROOT/'experiments/results/task41l_audited_v1/reuse_manifest.json').read_text())
    reuse.pop('local')
    hypotheses=[
      dict(id='selective_transfer',statement='The frozen Ca_HVA-only repair transfers across full panels without the other donor-channel regressions.',contrast='swap100 versus swap000 versus K_full under identical primary policy',falsifier='New autonomous/imposed/recurrence failures remain; a local stress repair alone is insufficient.'),
      dict(id='numerical_contract',statement='A coherent voltage/solve numerical policy removes autonomous failures across whole panels, not only selected cases.',contrast='primary versus voltage_solve64 for every frozen model, both fixed and matched references',falsifier='Remaining/new full-panel failures, bad canonical floors or lack of continuous-error reduction.'),
      dict(id='reference_confound',statement='Some apparent gains depend on changing the canonical numerical reference.',contrast='Same exact prediction versus original FP32 reference and versus policy-matched canonical reference',falsifier='Both references yield the same conclusions and comparable continuous errors; pass flips alone are not mechanistic proof.'),
      dict(id='fp32_algebra',statement='Recentered but mathematically identical FP32 solve can capture the mixed-precision benefit without its numerical cost.',contrast='recentered32 versus primary and voltage_solve64, full paired quality and integrated timing',falsifier='Autonomous failures persist or new ones appear; no deployment claim from faster inaccurate results.')]
    spec=dict(task='41m',version='v1',purpose='Frozen selective repair plus full numerical-contract verification; repair and compression remain separate.',
        models=['swap000','swap100','K_full'],policies=['primary','voltage_solve64','recentered32'],seeds=old['seeds'],dt_ms=.1,
        training_updates=0,selection_performed=False,max_gpu_seconds=3600,gpu='Tesla T4',torch='2.11.0+cu130',
        quality=old['quality'],compartment_gates=old['compartment_gates'],cost=old['cost'],reuse=reuse,
        frozen_diagnostic_input=old['frozen_diagnostic_input'],hypotheses=hypotheses,
        control_failures={panel:[dict(case=r['case'],seed=r['seed']) for r in audit['failures']['H_nominee'][panel+'/autonomous/responses']] for panel in ('41','41c','historical')},
        outcomes=dict(primary_repair='All newly evaluated primary panels plus reused imposed stress/recurrence pass; still requires independent confirmation and deployment-cost assessment.',
            numerical_repair='Improvement only under diagnostic numerics is evidence for a concrete contract change, not a waiver of primaryFP32 or Task42.',
            insufficient='If no full-panel repair and no new actionable causal distinction, stop this repair family; no automatic training extension.'),
        stop_rules=['60-minute scientific-worker wall budget, excluding environment setup; preserve partials; exhaustion is not scientific NO-GO.',
                    'Nonfinite parameters/traces, SHA mismatch, primary H control mismatch: stop and preserve technical failure.',
                    'No automatic retraining, reselection, new capacity, threshold changes, independent-confirmation claim or Task42.'],
        limits=['All panels already opened: diagnostic only; seeds17/29/43 not new independent biological replicas.',
                'Three model interventions share frozen weights; swap100 evaluates two networks, not a compressed deployment.',
                'Voltage/Ca/gate/current traces and response differentials use exact previous alignment and thresholds.',
                'Imposed stress and recurrence reused byte-identically from41l because numerical voltage policy is absent in those tests.',
                'No new development selection; no complete release qualification is claimed by this diagnostic.',
                'Training horizon/precision mismatch recorded but not directly changed or causally tested by41m.',
                'DB backup local only; scientific archive/logs copied off-runtime with SHA/CRC.'])
    cells=[dict(cell_type='markdown',metadata={},source=['# GIADA41m — Verifica congelata del contratto numerico\n','Nessun training o selezione. Tre modelli × tre politiche numeriche; riferimenti originale e allineato. Soglie invariate, limite60min.\n','La console esegue lo stesso worker di questo notebook; nessuna duplicazione del run.']),
        dict(cell_type='code',metadata={},execution_count=None,outputs=[],source=[
            'from pathlib import Path\nimport os, subprocess, sys\n',
            "REPO=Path('/content/giada_task41m/giada') # checkout fissata dal manager console\n",
            "OUTPUT=REPO.parent/'artifacts'\nassert not OUTPUT.exists(), 'Non duplicare il run console.'\n",
            "subprocess.run([sys.executable,'-u',str(REPO/'scripts/run_task41m_frozen_contract.py'),str(OUTPUT)],cwd=REPO,check=True)\n"])]
    notebook=ROOT/'notebooks/giada_task41m_frozen_contract.ipynb'
    notebook.write_text(json.dumps(dict(cells=cells,metadata=dict(kernelspec=dict(display_name='Python3',language='python',name='python3')),nbformat=4,nbformat_minor=5),indent=2),encoding='utf-8',newline='\n')
    names=['src/giada_teacher/task41m_frozen_contract.py','scripts/run_task41m_frozen_contract.py','scripts/build_task41m_protocol.py','notebooks/giada_task41m_frozen_contract.ipynb','tests/test_task41m_frozen_contract.py']
    spec['sources']={n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in names}
    (ROOT/'experiments/task41m_preregistration_v1.json').write_text(json.dumps(spec,indent=2),encoding='utf-8',newline='\n')
    print('Sealed41m protocol, no training or launch')

if __name__=='__main__':main()
