"""Persist initial45c probes and classify the disproved gate-name claim correctly."""
import argparse,io,json,os,zipfile,hashlib
import numpy as np
from .mirror import ROOT,write_json
from .incremental_batch import IncrementalBatchMirror
from .register_task41d_result import ResultPreview,PreviewComplete

def probes(repo):
    import torch
    from src.giada_teacher import task45b_rate_coverage as b
    torch.set_num_threads(1);os.environ['GIADA_TASK41N_FROZEN']=str(repo/'artifacts/task41n_frozen.zip.bin')
    folder=repo/'experiments/results/task45b_audited_v1';freeze=json.loads((folder/'f011_freeze.json').read_text())
    net=b.n.load_models(torch,repo,'cpu')[0]['dual']
    with zipfile.ZipFile(repo/'artifacts/task45b_kaggle_external/task45b_final.zip') as z:
        state=z.read('f011.pt');assert hashlib.sha256(state).hexdigest()==freeze['source_member_sha256']
        net.load_state_dict(torch.load(io.BytesIO(state),map_location='cpu',weights_only=True));assert b.n.weights_hash(net)==freeze['logical_weights_sha256']
        v=torch.full((3,1),-130.);ca=torch.full_like(v,1e-4);rates=[]
        with torch.inference_mode():
            u,zz=net.raw(v,ca)
            for channel,gate in [('Ca_LVAst',3),('NaTs2_t',7)]:
                inf,tau=b.t._rates(channel,v.double(),ca.double(),torch)
                for si,seed in enumerate((17,29,43)):
                    update=float(torch.sigmoid(zz[si,0,gate]));rates.append(dict(channel=channel,gate='h',index=gate,seed=seed,voltage_mv=-130.,
                        teacher_inf=float(inf[si,0,1]),teacher_tau_ms=float(tau[si,0,1]),learned_inf=float(torch.sigmoid(u[si,0,gate])),
                        learned_update=update,learned_tau_ms=None if update==0 else (0. if update==1 else float(-.1/np.log1p(-update))),
                        owner='donor' if bool(net.mask[gate]) else 'parent'))
        with np.load(io.BytesIO(z.read('original45/reference_45_chain6.npz'))) as rr,np.load(io.BytesIO(z.read('original45/traces_45_chain6_autonomous.npz'))) as pp:
            ref=rr['current'][:,0,0,3,1,5];pred=pp['current_Nap_Ih'][:,0,4,3,1,1]
            distance=abs(ref-ref[0]);pdistance=abs(pred-ref[0]);threshold=.1*float(distance.max())
            ri=int(np.flatnonzero(distance>=threshold)[-1]);pi=int(np.flatnonzero(pdistance>=threshold)[-1])
            slope=float(distance[min(len(distance)-1,ri+1)]-distance[max(0,ri-1)])/.2
            residual=float(pdistance[ri]-distance[ri]);timing=dict(geometry='45_chain6',case=3,node=1,seed=17,arm='f011',mode='autonomous',channel='Ih',
                reference_offset_ms=ri*.1,candidate_offset_ms=pi*.1,observed_signed_shift_ms=(pi-ri)*.1,reference_slope=slope,distance_residual=residual,
                linear_predicted_shift_ms=None if slope==0 else -residual/slope,unchanged_threshold_ms=.3,diagnostic_only=True)
    result=dict(valid=True,source_archive_sha256=freeze['source_archive_sha256'],f011_hash=freeze['logical_weights_sha256'],rates_at_minus130=rates,remaining_Ih_crossing=timing,
        interpretation='Ca_LVAst.h rate error at -130mV supports a targeted kinetic probe; NaTs2_t.h coefficients are not exact either, but its measured hidden-state error is not the dominant failure. Ih local linear shift agrees with observed -0.9ms; this is a diagnostic at one opened crossing, not proof of transfer.',
        training_performed=False,thresholds_changed=False,abstraction_changed=False)
    write_json(folder/'initial45c_probes.json',result);return result

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--validate-only',action='store_true');args=ap.parse_args();repo=ROOT.parent
    folder=repo/'experiments/results/task45b_audited_v1';prior=json.loads((folder/'memory_receipt.json').read_text());result=probes(repo)
    original=json.loads((repo/'experiments/results/task45b_protocol_v1/result_memory_receipt.json').read_text())
    m=ResultPreview() if args.validate_only else IncrementalBatchMirror();ids={}
    if not args.validate_only:assert m.baseline_hash==prior['verification']['snapshot_hash']
    def put(table,suffix,**fields):
        code=f'{table}-giada-task45b-audit-{suffix}-v1';rid=m.local_upsert(table,{'Codice stabile':code,**fields})['record_id'];ids[code]=rid;return rid
    put('claims','45c-gate_identity_negative_control',Stato='Contraddetta nel dominio')
    art=original['records']['artifacts-giada-task45b-final-v1'];exp=original['records']['experiments-giada-task45b-rate-coverage-v1']
    obs=put('observations','initial45c-probes',Nome='Initial45c constant-coefficient and intermediate Ih crossing probes',Descrizione=json.dumps(result),**{'Artefatti dettagliati':[art]})
    finding=put('findings','initial45c-probes',Nome='f011 Ca_LVAst.h tail kinetics and flat Ih crossing support focused45c tests',Esito='Misto',Esperimenti=[exp],Osservazioni=[obs],Risultato=json.dumps(result),Limitazioni='One held voltage and one opened crossing; no learned repair or new qualification. Flat-crossing tangent agreement need not hold at other events. NaTs2_t.h not universally exact, only non-dominant in the six tested failures.')
    put('evidence','45c-ih_crossing_residual',**{'Risultati a sostegno':[prior['records']['findings-giada-task45b-audit-ih-v1'],finding],'Argomentazione':'Opened f011 intermediate-node Ih crossing: measured shift -0.9ms, local deltaI/slope predicts approximately -0.934ms. This supports the sensitivity hypothesis; exact original timing failure remains counted.'})
    m.commit_batch()
    if args.validate_only:return
    verification=m.verify_incremental(prior['verification']);assert verification['valid']
    write_json(folder/'final_memory_receipt.json',dict(valid=True,verification=verification,records=ids,previous_audit_receipt='memory_receipt.json',negative_gate_claim_status='Contraddetta nel dominio',airtable_accessed=False,task45c_launched=False))
    m.export_snapshot_to(ROOT/'data/airtable_snapshot.json');print('45b audit addendum and contradicted gate attribution verified/exported.',flush=True)

if __name__=='__main__':
    try:main()
    except PreviewComplete:print('45b audit addendum schema valid; no writes')
