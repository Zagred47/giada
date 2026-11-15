"""Read-only archive audit and descriptive48 comparisons; no requalification."""
import io,json,hashlib,zipfile,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from research_memory.mirror import write_json

def main():
    folder=ROOT/'artifacts/task48_kaggle_external';path=folder/'task48_final.zip';receipt=json.loads((folder/'kaggle_receipt.json').read_text())
    with path.open('rb') as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
    assert digest==receipt['sha256'] and path.stat().st_size==receipt['bytes']
    with zipfile.ZipFile(path) as z:
        assert z.testzip() is None
        report=json.loads(z.read('final_report.json'));rows=json.loads(z.read('diagnostic_rows.json'));p=json.loads(z.read('preregistration.json'));design=json.loads(z.read('probe_design.json'))
        assert report['complete'] and report['valid'] and report['controls']['weights_hash_before']==report['controls']['weights_hash_after']==p['weights_sha256']
        collisions={};recovery={};neighbours={};spectra={};errors={}
        for model in ('canonical64','canonical32','P'):
            per_channel={}
            for ch in ('Ca_HVA','Ca_LVAst','NaTa_t','NaTs2_t','Nap_Et2','K_Pst','K_Tst'):
                rr=[r for r in rows if r['kind']=='exact_opening_collision' and r['model']==model and r['channel']==ch and r['horizon_ms']==1000]
                per_channel[ch]=dict(max_gap=max(max(r['max_future_gap_normalized']) for r in rr),min_gap=min(min(r['max_future_gap_normalized']) for r in rr),pairs_above_shared_scale_10percent=sum(max(r['max_future_gap_normalized'])>.1 for r in rr),total_pair_drive_combinations=len(rr))
            collisions[model]=per_channel
            for family in p['families']:
                rr=[r for r in rows if r['kind']=='stacked_horizon_observability' and r['model']==model and r['family']==family]
                matrix=np.array([r['singular_values'] for r in rr])
                spectra[model+'/'+family]=dict(largest_by_direction=matrix.max(axis=(0,1)).tolist(),smallest_by_direction=matrix.min(axis=(0,1)).tolist(),relative_rank_at_posthoc_rtol_1e_6=[int(sum(np.array(seed)>max(seed)*1e-6)) if max(seed)>0 else 0 for r in rr for seed in r['singular_values']],posthoc_descriptive_only=True)
            for future in p['future_protocols']:
                with np.load(io.BytesIO(z.read(f'panel_{future}_{model}.npz'))) as values:
                    trace=values['witness_current'];end={}
                    for ch in per_channel:
                        ratios={}
                        for horizon in (100,500,1000):
                            gaps=[]
                            for pair in design['collision_pairs']:
                                if pair['channel']!=ch:continue
                                l,r=pair['left'],pair['right'];stop=round(horizon/.1)+1
                                scale=np.maximum(np.maximum(abs(trace[:stop,:,l]).max(0),abs(trace[:stop,:,r]).max(0)),1e-7)
                                gaps.extend((abs(trace[stop-1,:,l]-trace[stop-1,:,r])/scale).tolist())
                            ratios[str(horizon)]=max(gaps)
                        end[ch]=ratios
                    recovery[future+'/'+model]=end
            for view in ('full_gates','channel_openings','family_currents'):
                rr=[r for r in rows if r['kind']=='reachable_neighbours' and r['model']==model and r['view']==view]
                ranked=sorted(rr,key=lambda r:r['input_distance'])
                neighbours[model+'/'+view]=dict(pair_count=len(design['neighbours']),input_distance_min=min(r['input_distance'] for r in rr),input_distance_max=max(r['input_distance'] for r in rr),max_future_endpoint_gap=max(max(r['endpoint_future_gap_normalized']) for r in rr),closest_examples=ranked[:3],not_impossibility_proof=True)
        for future in p['future_protocols']:
            with np.load(io.BytesIO(z.read(f'panel_{future}_canonical64.npz'))) as ref:
                for model in ('canonical32','P'):
                    with np.load(io.BytesIO(z.read(f'panel_{future}_{model}.npz'))) as pred:
                        normal=[];absolute=[]
                        for h in p['horizons_ms']:
                            target=ref[str(h)][:,:p['history_count']];got=pred[str(h)][:,:p['history_count']];scale=np.maximum(abs(target),1e-7)
                            normal.append(abs(got-target)/scale);absolute.append(abs(got-target))
                        normal=np.stack(normal);absolute=np.stack(absolute)
                        ix=np.unravel_index(np.argmax(normal),normal.shape)
                        errors[future+'/'+model]=dict(max_endpoint_relative_to_instantaneous_floor_normalized=float(normal.max()),max_absolute_current_error_ma_cm2=float(absolute.max()),worst=dict(horizon=p['horizons_ms'][ix[0]],seed=(17,29,43)[ix[1]],history=int(ix[2]),channel=int(ix[3])),not_registered_trajectory_nmax=True)
        output=dict(valid=True,complete=True,archive_sha256=digest,archive_bytes=path.stat().st_size,backup_runtime_independent=True,elapsed_seconds=report['elapsed_seconds'],controls=report['controls'],collision_summary=collisions,recovery_endpoint_gaps=recovery,local_observability_spectra=spectra,reachable_neighbours=neighbours,paired_endpoint_errors=errors,history_gate_max_rmse={k:float(np.max(v)) for k,v in report['history_gate_rmse_by_seed'].items()},family_pca=report['families'],limitations=['72 histories end with common -65mV for50–100ms and calcium5e-5mM for50ms: fast-gate and SK variation is largely washed out. PCA cannot establish global state reduction.','Exact constructed opening collisions have no demonstrated voltage-history reachability.','P future probes start from canonical reachable endpoints, not P-conditioned endpoints; history error measured separately.','Local finite-difference rank is sampled, scale-dependent and numerically conditioned; not a globally minimal dimension.','V/Ca externally imposed, no coupled voltage, calcium feedback, synapses or new native validation.','Endpoint error normalization here is diagnostic, not the preregistered full-trajectory quality gate; thresholds unchanged.','No new model trained and no speedup demonstrated; Task47 limits retained.'])
    dest=ROOT/'experiments/results/task48_audit_v1';dest.mkdir(exist_ok=True);write_json(dest/'audit.json',output)
    print(json.dumps({k:output[k] for k in ('collision_summary','recovery_endpoint_gaps','paired_endpoint_errors','history_gate_max_rmse','local_observability_spectra')},indent=2))
if __name__=='__main__':main()
