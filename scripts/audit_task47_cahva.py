"""Forensic audit of saved Ca_HVA traces; original evaluator remains unchanged."""
import hashlib,io,json,sys,zipfile
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from research_memory.mirror import write_json
from src.giada_teacher.task37_ionic_playground import event_metrics

F=ROOT/'experiments/results/task47_cahva_audit_v1'
def intervals(mask,dt=.1):
    changes=np.diff(np.r_[False,mask,False].astype('int8'));starts=np.flatnonzero(changes==1);ends=np.flatnonzero(changes==-1)-1
    return [dict(start_index=int(a),end_index=int(b),start_ms=float(a*dt),end_ms=float(b*dt),samples=int(b-a+1)) for a,b in zip(starts,ends)]
def describe(reference,prediction,dt=.1):
    base=float(reference[0]);rr=np.abs(reference-base);pp=np.abs(prediction-base);amp=float(rr.max());threshold=.1*amp
    r=rr>=threshold;p=pp>=threshold
    return dict(original_event=event_metrics(reference,prediction,dt,1e-7,.001),reference_baseline_ma_cm2=base,amplitude_ma_cm2=amp,threshold_ma_cm2=threshold,reference_active_intervals=intervals(r,dt),candidate_active_intervals=intervals(p,dt),reference_peak_index=int(rr.argmax()),candidate_peak_index=int(pp.argmax()),peak_ratio=float(pp.max()/amp),reference_above_threshold_margin=rr-threshold,candidate_above_threshold_margin=pp-threshold,active_mask_disagreement_indices=np.flatnonzero(r!=p).tolist(),max_abs_current_error_ma_cm2=float(abs(prediction-reference).max()),current_nrmse=float(np.sqrt(np.mean((prediction-reference)**2))/max(float(abs(reference).max()),1e-7)))
def member_arrays(z,name):
    # Read one saved NPZ at a time, not every large trace array simultaneously.
    return np.load(io.BytesIO(z.read(name)))
def main():
    F.mkdir(exist_ok=True);archive=ROOT/'artifacts/task47_retry_external/task47_final.zip';p=json.loads((ROOT/'experiments/task47_preregistration_v1.json').read_text());q=p['quality']
    with zipfile.ZipFile(archive) as z:
        report=json.loads(z.read('final_report.json'));assert report['executed_protocol_verified']
        rows=json.loads(z.read('results.json'));fail=[r for r in rows if r['arm']=='P' and not(r['current_passed'] and r['compartment_passed'])]
        assert {(r['geometry'],r['case'],r['seed']) for r in fail}=={('47_chain5',8,17)}
        members=['reference_47_chain5.npz','traces_47_chain5_imposed.npz','traces_47_chain5_autonomous.npz']
        provenance={name:dict(uncompressed_bytes=z.getinfo(name).file_size,zip_crc32=f'{z.getinfo(name).CRC:08x}') for name in members}
        evidence={}
        with member_arrays(z,members[0]) as src:
            evidence['reference_current']=src['current'][:,0,0,8,4,:].copy();evidence['reference_voltage']=src['voltage'][:,0,0,8,4].copy();evidence['reference_gates']=src['gates'][:,0,0,8,4,:2].copy()
        print('[47 audit] reference case extracted',flush=True)
        for mode,member in zip(('imposed','autonomous'),members[1:]):
            with member_arrays(z,member) as src:
                evidence[mode+'_current']=src['current'][:,:,:,8,4,:].copy();evidence[mode+'_voltage']=src['voltage'][:,:,:,8,4].copy();evidence[mode+'_gates']=src['gates'][:,:,:,8,4,:2].copy()
            print('[47 audit] '+mode+' case extracted',flush=True)
    np.savez_compressed(F/'case8_node4_evidence.npz',**evidence)
    audits=[]
    for mode in ('imposed','autonomous'):
        for seed_index,seed in enumerate((17,29,43)):
            for arm_index,arm in enumerate(('none','C1','P','R41')):
                for horizon in (100.,500.,1000.):
                    count=round(horizon/.1);r=evidence['reference_current'][:count,0];v=evidence[mode+'_current'][:count,seed_index,arm_index,0]
                    detail=describe(r,v);detail.pop('reference_above_threshold_margin');detail.pop('candidate_above_threshold_margin')
                    audits.append(dict(mode=mode,seed=seed,arm=arm,horizon_ms=horizon,**detail))
    for row in fail:
        want=next(c['event'] for n in row['per_node'] for c in n['channels'] if c['channel']=='Ca_HVA' and not c['passed'])
        got=next(v for v in audits if all(v[k]==row[k] for k in ('mode','seed','arm','horizon_ms')))['original_event'];assert want==got
    result=dict(valid=True,archive_sha256=json.loads((archive.parent/'kaggle_receipt.json').read_text())['sha256'],archive_provenance=provenance,case=fail[0]['protocol'],case_index=8,node_index=4,channel='Ca_HVA',dt_ms=.1,thresholds=q,original_failure_reproduced_exactly=True,training_performed=False,thresholds_changed=False,original_report_reclassified=False,audits=audits)
    write_json(F/'trace_audit_initial.json',result)
    print(json.dumps([r for r in audits if r['arm']=='P' and r['seed']==17 and r['horizon_ms']==1000.]),flush=True)
if __name__=='__main__':main()
