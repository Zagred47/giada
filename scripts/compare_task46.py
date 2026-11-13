"""Paired46 accuracy and intervention audit; descriptive scoped cost frontier."""
import collections,json,statistics,sys,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from research_memory.mirror import write_json
from src.giada_teacher.task46_spatial_cost import y
def main():
    r=json.loads((ROOT/'artifacts/task46_kaggle_external/external_backup_receipt.json').read_text());assert r['valid']
    if r['technical_failure']:
        write_json(ROOT/'experiments/results/task46_protocol_v1/comparison.json',dict(complete=False,technical_failure=True,report=r['report'],no_scientific_success_claim=True));return
    with zipfile.ZipFile(r['local']) as z:
        rows=json.loads(z.read('results.json'));group=collections.defaultdict(list)
        for q in rows:
            for key in ('all',q['geometry'],'block='+q['block'],'density='+q['density'],'site='+q['site_label'],'coupling='+str(q['coupling'])):group[(key,q['horizon_ms'],q['mode'],q['arm'])].append(q)
        summary={};pairs=[];control={(q['geometry'],q['case'],q['seed'],q['horizon_ms'],q['mode']):q for q in rows if q['arm']=='C1'}
        for key,pool in group.items():
            eligible=[q for q in pool if q['reference_valid']]
            summary['/'.join(map(str,key))]=dict(total=len(pool),eligible=len(eligible),passed=sum(q['current_passed'] and q['compartment_passed'] for q in eligible),current_failures=sum(not q['current_passed'] for q in eligible),compartment_failures=sum(not q['compartment_passed'] for q in eligible),worst_voltage_rmse_mv=max(q['voltage_rmse_mv'] for q in pool),worst_calcium_rmse_mM=max(q['calcium_rmse_mM'] for q in pool))
        for q in rows:
            if q['arm'] not in ('P','R41') or not q['reference_valid']:continue
            before=control[(q['geometry'],q['case'],q['seed'],q['horizon_ms'],q['mode'])];a=before['current_passed'] and before['compartment_passed'];b=q['current_passed'] and q['compartment_passed']
            if a!=b:pairs.append(dict(geometry=q['geometry'],case=q['case'],seed=q['seed'],arm=q['arm'],horizon_ms=q['horizon_ms'],mode=q['mode'],protocol=q['protocol'],new_failure_vsC1=bool(a and not b),recovered_vsC1=bool(b and not a)))
        channels={}
        for arm in ('C1','P','R41'):
            for mode in ('imposed','autonomous'):
                for channel in sorted({c['channel'] for q in rows for node in q['per_node'] for c in node['channels']}):
                    pool=[c for q in rows if q['arm']==arm and q['mode']==mode and q['horizon_ms']==1000 and q['reference_valid'] for node in q['per_node'] for c in node['channels'] if c['channel']==channel]
                    if pool:channels[arm+'/'+mode+'/'+channel]=dict(count=len(pool),failed=sum(not c['passed'] for c in pool),max_nrmse=max(c['nrmse'] for c in pool),max_nmax=max(c['nmax'] for c in pool),max_offset_ms=max((c['event'].get('offset_error_ms') or 0) for c in pool),max_onset_ms=max((c['event'].get('onset_error_ms') or 0) for c in pool),max_peak_ms=max((c['event'].get('peak_error_ms') or 0) for c in pool),event_status=dict(collections.Counter(c['event']['status'] for c in pool)))
        cost=[]
        for q in json.loads(z.read('cost.json')):
            med={name:statistics.median(v['wall_ms']) for name,v in q['samples'].items()};cost.append(dict(batch=q['batch_per_seed'],scope=q['scope'],median_wall_ms=med,formula32_over_model={name:med['formula32']/v for name,v in med.items()},formula64_over_model={name:med['formula64']/v for name,v in med.items()} if 'formula64' in med else {}))
        components=[]
        for q in json.loads(z.read('coupled_component_cost.json')):
            components.append(dict(geometry=q['geometry'],sites=q['actual_sites_per_model'],median_wall_ms={name:{scope:statistics.median(v) for scope,v in values.items()} for name,values in q['samples_wall_ms'].items()},scope=q['scope'],non_additive=True))
        propagation={}
        for name in z.namelist():
            if name.startswith('propagation_') and name.endswith('.json'):
                values=json.loads(z.read(name));propagation[name]=dict(evaluated=sum('counts_match' in q for q in values),crossing_count_mismatches=sum(not all(q['counts_match']) for q in values if 'counts_match' in q),diagnostic_only=True)
        feedback=[];coupling=[]
        for h in sorted({q['horizon_ms'] for q in rows}):
            pool=[q for q in rows if q['horizon_ms']==h]
            feedback.extend(dict(horizon_ms=h,**q) for q in y.paired_contrasts(pool));coupling.extend(dict(horizon_ms=h,**q) for q in y.coupling_contrasts(pool))
        write_json(ROOT/'experiments/results/task46_protocol_v1/comparison.json',dict(complete=True,source_sha256=r['remote']['sha256'],summary=summary,paired_changes_vsC1=pairs,imposed_autonomous=feedback,disconnected_coupled=coupling,channels=channels,cost=cost,components=components,propagation=propagation,interventions=json.loads(z.read('intervention_pairs.json')),limits='P frozen from incomplete45e. No reconstructed success of missing45e stress/cost/regression44. Different46protocols are not a one-to-one45e pass-count comparison. Native-invalid cases excluded. Component timings nonadditive; no release/47/speedup claim.'))
if __name__=='__main__':main()
