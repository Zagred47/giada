"""All49 initialization panels: unchanged amplitude/charge gates, no release claim."""
import gzip,hashlib,json,zipfile,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from research_memory.mirror import write_json

def main():
    path=ROOT/'artifacts/task49_kaggle_external/task49_final.zip';receipt=json.loads((path.parent/'kaggle_receipt.json').read_text())
    with path.open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==receipt['sha256']
    with zipfile.ZipFile(path) as z:
        assert z.testzip() is None;report=json.loads(z.read('final_report.json'));assert report['complete'] and report['valid'];plan=json.loads(z.read('preregistration.json'));summary={};slow={};numeric=[]
        for future in plan['future_protocols']:
            rows=json.loads(gzip.decompress(z.read('metrics_'+future+'.json.gz')))
            for row in rows:
                if row.get('kind'):numeric.append(row);continue
                key=f"{row['role']}/{row['cut_ms']}/{row['arm']}";r=summary.setdefault(key,dict(passed=0,total=0,max_current_nrmse=0.,max_current_nmax=0.,max_paired_nmax=0.,max_gate_error=0.))
                r['passed']+=sum(row['all_channel_amplitude_charge_pass_by_history']);r['total']+=len(row['history_ids'])
                r['max_current_nrmse']=max(r['max_current_nrmse'],float(np.max(row['metrics']['current_nrmse'])))
                r['max_current_nmax']=max(r['max_current_nmax'],float(np.max(row['metrics']['current_nmax'])))
                r['max_paired_nmax']=max(r['max_paired_nmax'],float(np.max(row['paired_nmax_to_P_history'])))
                r['max_gate_error']=max(r['max_gate_error'],float(np.max(row['metrics']['gate_max'])))
                if row['role']=='fresh':
                    sk=f"{row['cut_ms']}/{row['arm']}/{row['horizon_ms']}";s=slow.setdefault(sk,dict(Nap_h=0.,K_Pst_h=0.,Ca_HVA_h=0.))
                    for name,gate in [('Nap_h',9),('K_Pst_h',13),('Ca_HVA_h',1)]:s[name]=max(s[name],float(np.max(np.array(row['metrics']['gate_rmse'])[:,gate])))
        costs=json.loads(z.read('initialization_cost.json'));cost_medians={k:float(np.median(v)) for k,v in costs['samples_ms'].items()}
        output=dict(valid=True,complete=True,archive_sha256=receipt['sha256'],archive_bytes=path.stat().st_size,report=report,summary=summary,slow_gate_rmse=slow,cost_median_ms=cost_medians,numeric_controls=numeric,scope='Counts pool3seeds x3future protocols x6 horizons x24fresh histories=1296 per cut/arm. Repeated horizons/cases are not independent samples. Only unchanged current amplitude/charge checks, not full timing/gate/coupled release.',cost_scope=costs['scope'])
    folder=ROOT/'experiments/results/task49_audit_v1';folder.mkdir(exist_ok=True);write_json(folder/'audit.json',output)
    print('FRESH amplitude/charge counts pooled across horizons (NOT independent):')
    for cut in plan['cuts_ms']:
        print('CUT',cut)
        for arm in plan['arms']:
            r=summary[f'fresh/{cut}/{arm}'];print(arm,r)
    print('COST',cost_medians)
if __name__=='__main__':main()
