"""Derived45b audit, streamed metrics and explicit correction of gate identity."""
import hashlib,io,json,sys,zipfile
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from research_memory.mirror import write_json
from src.giada_teacher import task45b_rate_coverage as b

def streamed_rows(z,name):
    with z.open(name) as f:
        buf=[]
        for line in f:
            if line.startswith(b'  {'):buf=[line]
            elif buf:
                if line.rstrip() in (b'  },',b'  }'):
                    yield json.loads(b''.join(buf)+b'  }');buf=[]
                else:buf.append(line)

def main():
    receipt=json.loads((ROOT/'artifacts/task45b_kaggle_external/external_backup_receipt.json').read_text());path=Path(receipt['local'])
    assert receipt['valid'] and receipt['runtime_independent'] and not receipt['technical_failure']
    with path.open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==receipt['remote']['sha256']
    assert path.stat().st_size==receipt['remote']['bytes']
    folder=ROOT/'experiments/results/task45b_audited_v1';folder.mkdir(exist_ok=True)
    flat=[(channel,state) for channel,states in zip(b.t.ionic.CHANNELS,b.t.ionic.STATE_NAMES) for state in states]
    assert flat[3]==('Ca_LVAst','h') and flat[7]==('NaTs2_t','h')
    panels={};failures=[];guards=[];timing=[]
    with zipfile.ZipFile(path) as z:
        assert z.testzip() is None
        report=json.loads(z.read('final_report.json'))
        for panel in ('original45','safe_hyper','regression44'):
            pool={};maxima={}
            for r in streamed_rows(z,panel+'/results.json'):
                if not r['reference_valid'] or r['arm']=='none':continue
                key=(r['arm'],r['mode'],r['horizon_ms']);d=pool.setdefault(key,dict(total=0,passed=0,regressions=0,gate_failed_cases=0,current_failed_cases=0))
                d['total']+=1;d['passed']+=int(r['current_passed'] and r['compartment_passed']);d['regressions']+=int(r['regression_vs_f000'])
                gate_bad=any(not node['gate']['passed'] for node in r['per_node'])
                current_bad=any(not c['passed'] for node in r['per_node'] for c in node['channels'])
                d['gate_failed_cases']+=int(gate_bad);d['current_failed_cases']+=int(current_bad)
                for j,node in enumerate(r['per_node']):
                    for c in node['channels']:
                        if c['channel'] not in ('Ih','Nap_Et2'):continue
                        k=key+(c['channel'],);m=maxima.setdefault(k,dict(nmax=0.,onset_offset_ms=0.,peak_ms=0.,failed_node_cases=0))
                        m['nmax']=max(m['nmax'],c['nmax']);m['failed_node_cases']+=int(not c['passed'])
                        e=c['event'];m['onset_offset_ms']=max(m['onset_offset_ms'],e.get('onset_error_ms') or 0,e.get('offset_error_ms') or 0);m['peak_ms']=max(m['peak_ms'],e.get('peak_error_ms') or 0)
                        if r['arm'] in ('f011','f111') and not c['passed']:
                            failures.append(dict(panel=panel,**{k:r[k] for k in ('geometry','case','seed','coupling','mode','horizon_ms','arm')},node=j,channel=c['channel'],event=e,nmax=c['nmax']))
            panels[panel]=dict(summaries=[dict(arm=k[0],mode=k[1],horizon_ms=k[2],**v) for k,v in pool.items()],channels=[dict(arm=k[0],mode=k[1],horizon_ms=k[2],channel=k[3],**v) for k,v in maxima.items()])
            print('[GIADA45b audit] streamed '+panel,flush=True)
        # Read full18-state tensors, not just the overall gate error statistic.
        for geo in json.loads(z.read('preregistration.json'))['geometries']:
            gid=geo['id']
            with np.load(io.BytesIO(z.read('safe_hyper/reference_'+gid+'.npz'))) as rr,np.load(io.BytesIO(z.read('safe_hyper/traces_'+gid+'_autonomous.npz'))) as pp:
                reference=rr['gates'];pred=pp['gates_f000_f111']
                for si,seed in enumerate((17,29,43)):
                    error=np.max(abs(pred[:,si,1,2,0]-reference[:,0,0,2,0]),0)
                    if error.max()>.03:
                        index=int(np.argmax(error));guards.append(dict(geometry=gid,case=2,protocol='45b_safe_hyper_130',coupling=0.,seed=seed,node=0,dominant_gate_index=index,dominant_gate='.'.join(flat[index]),max_gate_error=float(error.max()),
                            Ca_LVAst_h_error=float(error[3]),NaTs2_t_h_error=float(error[7]),Nap_Et2_h_error=float(error[9]),Ih_m_error=float(error[10])))
            del reference,pred
        import torch
        def tensors(value):
            if isinstance(value,torch.Tensor):yield value
            elif isinstance(value,dict):
                for v in value.values():yield from tensors(v)
            elif isinstance(value,(list,tuple)):
                for v in value:yield from tensors(v)
        checkpoints=[]
        for name in z.namelist():
            if name.endswith('.pt'):
                data=z.read(name);values=list(tensors(torch.load(io.BytesIO(data),map_location='cpu',weights_only=True)))
                checkpoints.append(dict(member=name,sha256=hashlib.sha256(data).hexdigest(),tensors=len(values),finite=all(bool(torch.isfinite(v).all()) for v in values)))
        assert all(r['finite'] for r in checkpoints)
        freeze=json.loads(z.read('repair_freeze.json'));calibration=json.loads(z.read('calibration_freeze.json'))
        stress=report['stress_regressions']
        audit=dict(valid=True,task='45b',source_archive_sha256=receipt['remote']['sha256'],source_archive_bytes=receipt['remote']['bytes'],sha_crc_verified=True,
            all_checkpoints_finite=True,checkpoints=checkpoints,all_traces_finite=report['all_traces_finite'],original45_reproduction=report['original45_reproduction'],
            panels=panels,failed_target_channels=failures,stress=stress,repair_freeze=freeze,calibration=calibration,
            corrected_gate_failures=guards,gate_index_mapping=[dict(index=i,channel=c,state=s) for i,(c,s) in enumerate(flat)],
            correction=dict(previous_assistant_statement='Residual gate index3 was described as NaTs2_t.h.',correct='Index3 is Ca_LVAst.h; NaTs2_t.h is index7.',why='Verified source STATE_NAMES/OFFSETS and per-gate trace errors. This corrects interpretation only, not measured metrics or gates.'),
            choice=dict(exploratory_start='f011',not_promoted_to_release=True,reason='Ih inf+rate repair improves original45 and preserves original stress66/72. Any Nap rate rollback introduces two new historical failures.'),
            interpretation=['Coverage at500 updates beat4x budget on nominal support; more steps were not a monotonic cure in this experiment.',
                'Nap rate rollback rescues negative-tail memory but globally destroys some compensated nominal timing; equilibrium and rate must be fitted with functional preservation guards.',
                'Ih rate correction alone lowers amplitude error but can worsen timing; both kinetic heads matter.',
                'Six safe-panel full-combination failures are Ca_LVAst.h hidden-state failures, not NaTs2_t.h, despite passing channel current metrics.'],
            thresholds_changed=False,abstraction_changed=False,independent47_accessed=False)
        write_json(folder/'audit.json',audit)
        # Reusable learned weights only, small and independently hashed; no need
        # to upload a4.6GB result archive just to initialize45c in the future.
        weights=torch.load(io.BytesIO(z.read('f011.pt')),map_location='cpu',weights_only=True)
        parent_weights=torch.load(io.BytesIO(z.read('frozen41r.pt')),map_location='cpu',weights_only=True)
        changed=[k for k in weights if not torch.equal(weights[k],parent_weights[k])]
        assert set(changed)<= {'parent.c','parent.d'}
        fit=np.concatenate((weights['parent.c'][:,:,[10,27]].numpy(),weights['parent.d'][:,None,[10,27]].numpy()),1)
        target=folder/'f011_ih_heads.npz';np.savez_compressed(target,weights=fit)
        write_json(folder/'f011_freeze.json',dict(valid=True,source_archive_sha256=receipt['remote']['sha256'],source_member='f011.pt',source_member_sha256=next(r['sha256'] for r in checkpoints if r['member']=='f011.pt'),
            logical_weights_sha256=b.n.weights_hash(type('StateView',(),{'state_dict':lambda self:weights})()),heads_path=target.relative_to(ROOT).as_posix(),heads_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
            base41r=json.loads(z.read('preregistration.json'))['freeze']['logical_weights_sha256'],changed_state_keys=changed,columns=[10,27],not_a_qualified_release=True))
    print('45b audit written, gate attribution corrected, f011 small freeze verified.',flush=True)

if __name__=='__main__':main()
