"""Read-only50 artifact audit; distinguish feasibility, loss and timing scope."""
import gzip,hashlib,io,json,sys,zipfile
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from research_memory.mirror import write_json

def main():
    import torch
    path=ROOT/'artifacts/task50_kaggle_external/task50_final.zip';receipt=json.loads((path.parent/'kaggle_receipt.json').read_text())
    with path.open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==receipt['sha256']
    with zipfile.ZipFile(path) as z:
        assert z.testzip() is None;report=json.loads(z.read('final_report.json'));assert report['valid'] and report['complete'];p=json.loads(z.read('preregistration.json'));rows=json.loads(gzip.decompress(z.read('evaluation.json.gz')));curve=json.loads(z.read('learning_curve.json'));freeze=json.loads(z.read('selection_freeze.json'));cost=json.loads(z.read('cost.json'));finite={}
        for name in ('selected.pt','checkpoint_250.pt','checkpoint_1000.pt','checkpoint_2000.pt','P.pt'):
            weights=torch.load(io.BytesIO(z.read(name)),map_location='cpu',weights_only=True);finite[name]=all(bool(torch.isfinite(v).all()) for v in weights.values() if torch.is_tensor(v));assert finite[name]
        summary={};channels={};reset={};horizons={}
        for row in rows:
            key=row['arm'];r=summary.setdefault(key,dict(passed=0,total=0,nrmse=[],nmax=[]));r['passed']+=sum(row['passed_by_history']);r['total']+=len(row['passed_by_history']);r['nrmse'].extend(np.array(row['nrmse']).flatten());r['nmax'].extend(np.array(row['nmax']).flatten())
            h=horizons.setdefault(f"{key}/{row['horizon_ms']}",dict(passed=0,total=0,median_nrmse=[]));h['passed']+=sum(row['passed_by_history']);h['total']+=len(row['passed_by_history']);h['median_nrmse'].extend(np.array(row['nrmse']).flatten())
            for i,ch in enumerate(('Ca_HVA','Ca_LVAst','NaTa_t','NaTs2_t','Nap_Et2','Ih','Im','K_Pst','K_Tst','SK_E2','SKv3_1')):
                k=key+'/'+ch;c=channels.setdefault(k,dict(passed=0,total=0,nrmse=[]));nr=np.array(row['nrmse'])[:,i];nm=np.array(row['nmax'])[:,i];cq=np.array(row['charge'])[:,i];aq=np.array(row['absolute_charge'])[:,i];c['passed']+=int(((nr<=.01)&(nm<=.05)&(cq<=.02)&(aq<=.02)).sum());c['total']+=len(nr);c['nrmse'].extend(nr)
            if row['reset_nrmse'] is not None and row['future']=='pulse_recovery' and row['horizon_ms']>=500:
                rr=reset.setdefault(key,dict(intact=[],reset=[]));rr['intact'].extend(np.array(row['nrmse']).flatten());rr['reset'].extend(np.array(row['reset_nrmse']).flatten())
        for collection in (summary,channels):
            for r in collection.values():
                for field in ('nrmse','nmax'):
                    if field in r:
                        values=r.pop(field);r['median_'+field]=float(np.median(values));r['max_'+field]=float(np.max(values))
        for r in horizons.values():r['median_nrmse']=float(np.median(r['median_nrmse']))
        for r in reset.values():
            for k,v in list(r.items()):r[k+'_median']=float(np.median(v));r[k+'_max']=float(np.max(v));del r[k]
        scaling={}
        for ai,arm in enumerate(p['arms']):scaling[arm['id']]={str(row['update']):float(np.mean(row['development'][3*ai:3*ai+3])) for row in curve}
        output=dict(valid=True,archive_sha256=receipt['sha256'],archive_bytes=path.stat().st_size,elapsed_seconds=report['elapsed_seconds'],finite_checkpoints=finite,summary=summary,channel_summary=channels,horizon_summary=horizons,reset_control=reset,scaling=scaling,cost_median_ms={k:float(np.median(v)) for k,v in cost['samples_ms'].items()},cost_scope=cost,selected_updates=freeze['updates'],limitations=['Counts pool repeated horizons,inputs,seeds; not independent cases.','Metric medians pooled across channels can hide severe failures.','Training chunks64ticks=6.4ms; development segments25.6ms, versus fresh1000ms. Budget improvements do not prove long-memory learnability.','No saved latent trajectories or dedicated slow-gate probes; current drift/reset only.','Cost compares15packed models against3P/canonical replicas, not per-model latency or speed at matched accuracy.','All tensors padded to18latent/32trunk in packed runtime; logical widths are masked, not real storage/compute compression.','Initial encoder sees physical gates; no voltage-only initializer or coupled release.'])
        from src.giada_teacher import task50_compact_memory as module
        net=module.model_class(torch)(p,'cpu');net.load_state_dict(torch.load(io.BytesIO(z.read('selected.pt')),map_location='cpu',weights_only=True))
        with torch.inference_mode():
            physical=torch.tensor(np.tile(module.q.steady(),(len(net.members),1,1)),dtype=torch.float32);pred=net.opening(net.initialize(physical))[:,0].numpy();target=module.q.openings(module.q.steady())
        output['posthoc_steady_initial_readout_probe']={ch:dict(target=float(target[k]),prediction_median=float(np.median(pred[:,k])),physical_voltage_mv=-65.,calcium_mM=5e-5,not_complete_rollout=True) for k,ch in enumerate(module.q.ionic.CHANNELS)}
        output['limitations'].append('Initialization encoder timing was not separately recorded by the benchmark; no fabricated per-arm cost or encoder latency.')
        folder=ROOT/'experiments/results/task50_audit_v1';folder.mkdir(exist_ok=True);write_json(folder/'audit.json',output)
        print(json.dumps({k:output[k] for k in ('summary','scaling','reset_control','cost_median_ms','horizon_summary')},indent=2))
if __name__=='__main__':main()
