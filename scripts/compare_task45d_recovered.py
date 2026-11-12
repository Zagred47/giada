"""Audit all45d arms against frozen45cJ; never treat completion as a pass."""
import collections,hashlib,json,statistics,sys,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from research_memory.mirror import write_json
def main():
    r=json.loads((ROOT/'artifacts/task45d_evaluation_external/external_backup_receipt.json').read_text());assert r['valid']
    with zipfile.ZipFile(r['local']) as z:
        if r['technical_failure']:
            write_json(ROOT/'experiments/results/task45d_evaluation_v1/comparison.json',dict(complete=False,technical_failure=True,report=r['report']));return
        report=json.loads(z.read('final_report.json'));rows=[]
        for panel in ('original45','safe_hyper','regression44'):rows.extend(json.loads(z.read(panel+'/results.json')))
        arms=sorted({q['arm'] for q in rows if q['arm']!='none'});paired={};timing={}
        for panel in ('original45','safe_hyper','regression44'):
            paired[panel]={}
            for h in sorted({q['horizon_ms'] for q in rows if q['panel']==panel}):
                paired[panel][str(h)]={}
                for mode in ('imposed','autonomous'):
                    pool=[q for q in rows if q['panel']==panel and q['horizon_ms']==h and q['mode']==mode];controls={(q['geometry'],q['case'],q['seed']):q for q in pool if q['arm']=='C1'}
                    for arm in arms:
                        selected=[q for q in pool if q['arm']==arm and q['reference_valid']];new=[];fixed=[];tm=collections.defaultdict(lambda:dict(onset=0.,offset=0.,peak=0.,missing=0))
                        for q in selected:
                            key=(q['geometry'],q['case'],q['seed']);before=controls[key];passed=q['current_passed'] and q['compartment_passed'];was=before['current_passed'] and before['compartment_passed']
                            if was and not passed:new.append(key)
                            if passed and not was:fixed.append(key)
                            for node in q['per_node']:
                                for channel in node['channels']:
                                    e=channel['event'];t=tm[channel['channel']]
                                    if e['status']=='missing_event':t['missing']+=1
                                    for name in ('onset','offset','peak'):
                                        if e.get(name+'_error_ms') is not None:t[name]=max(t[name],e[name+'_error_ms'])
                        paired[panel][str(h)][mode+'/'+arm]=dict(eligible=len(selected),passed=sum(q['current_passed'] and q['compartment_passed'] for q in selected),new_failures_vs45cJ=new,recovered_vs45cJ=fixed)
                        timing[panel+'/'+str(h)+'/'+mode+'/'+arm]=dict(tm)
        stress=report['stress'];baseline=set(map(tuple,stress['C1']['failed']));st={name:dict(passed=q['passed'],total=q['total'],new_failures_vs45cJ=sorted(set(map(tuple,q['failed']))-baseline),recovered_vs45cJ=sorted(baseline-set(map(tuple,q['failed'])))) for name,q in stress.items()}
        cost=[]
        for q in json.loads(z.read('cost.json')):
            med={name:statistics.median(v['wall_ms']) for name,v in q['samples'].items()};cost.append(dict(batch=q['batch_per_seed'],scope=q['scope'],median_wall_ms=med,formula32_over_model={k:med['formula32']/v for k,v in med.items()},formula64_over_model={k:med['formula64']/v for k,v in med.items()} if 'formula64' in med else {},C1_over_model={k:med['C1']/v for k,v in med.items()}))
        write_json(ROOT/'experiments/results/task45d_evaluation_v1/comparison.json',dict(complete=True,source_archive_sha256=r['remote']['sha256'],reference='Frozen C1 =45cJ, unchanged thresholds; original45c archived separately',paired=paired,timing=timing,stress=st,cost=cost,execution=json.loads(z.read('execution_variant_parity.json')),selection_unchanged=True,training_repeated=False,individual_cost_not_measured=[a for a in arms if a not in ('C0','C1','J')],limits='No release or automatic advancement; max event errors are diagnostic and missing events remain failures. Cost ratios descriptive, preserve original precision comparators.'))
if __name__=='__main__':main()
