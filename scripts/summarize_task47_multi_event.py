"""Separate implementation validation from biological event interpretation."""
import gzip,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from research_memory.mirror import write_json
F=ROOT/'experiments/results/task47_multi_event_v1'
if __name__=='__main__':
    s=json.loads((F/'summary.json').read_text());assert s['valid'] and s['complete'];cases=json.loads((F/'case_results.json').read_text());new=json.loads((F/'new_support_audit.json').read_text())
    details=json.loads(gzip.decompress((F/'event_details.json.gz').read_bytes()));controls=json.loads(gzip.decompress((F/'saved_reference_controls.json.gz').read_bytes()))
    bad=[r for r in controls if not r['expected_pass']];identity=[r for r in controls if r['expected_pass']]
    assert all(not r['observed_pass'] for r in bad) and all(r['observed_pass'] for r in identity)
    pnew=[r for r in cases if r['arm']=='P' and r['horizon_ms']==1000 and r['original_passed'] and not r['multi_event_passed']]
    assert {(r['geometry'],r['case'],r['seed'],r['mode']) for r in pnew}=={('47_chain5',9,seed,'autonomous') for seed in (17,29,43)}
    boundary=[r for r in new['rows'] if r['arm']=='P'];assert [len(r['mask_disagreement_samples']) for r in boundary]==[2,1,1]
    study=dict(valid=True,original_verdict_unchanged=True,training_performed=False,thresholds_changed=False,study_kind='Post-hoc frozen evaluator comparison on the existing47cohort; not independent confirmation',case_comparisons=s['case_comparisons'],channel_comparisons=s['channel_comparisons'],negative_mutants_rejected=len(bad),negative_mutants_total=len(bad),identity_controls_passed=len(identity),identity_controls_total=len(identity),original_also_rejected_all_constructed_negative_mutants=all(not r['original_passed'] for r in bad),
        P_counts={key:value for key,value in s['summary'].items() if key.endswith('/P')},
        conclusion='The frozen multi-event threshold-episode evaluator reproduces original timing and gate results exactly and passes the specified mutation/null controls. It retains the Ca_HVA support failure without a fictitious44.9ms matched-event delay, and exposes three additional P autonomous SK_E2 threshold-fragmentation cases. The model has not changed. These are added failures of a new stricter observable, not new errors induced in its trajectories.',
        Ca_HVA='Original 3reference threshold episodes versus2candidate episodes, one unmatched reference single-sample excursion; matched peak/onset/offset errors0. Still fails presence under the frozen strict definition.',
        SK_E2='One continuous reference above-threshold episode becomes a1or2sample initial candidate island, a1or2sample gap, and the main episode. The discrepancy consists of2samples forseed17 and1sample for29/43 (0.1or0.2ms); original0.3ms onset tolerance is met for the main matched episode, but the extra-component rule fails. No smoothing or merging was introduced after seeing this.',
        interpretation='Spurious/missing in this study refers to threshold-connected components, not proof of extra/missing biological responses. Passing negative controls is necessary but not sufficient to establish biological validity of event identity. This v1 remains sensitive to near-threshold topology. Do not silently reinterpret these components as spurious action potentials or waive them to claim144/144.',
        next_step_not_executed='Decide and preregister a domain-appropriate event-identity/support contract (including near-threshold fragmentation and rapid legitimate events) before further metric changes. Do not add training solely to chase aggregate offsets or grant task48 automatically.',task48_authorized=False,speedup_claimed=False,source_freeze='freeze.json')
    write_json(F/'interpretation.json',study);print(json.dumps(study),flush=True)
