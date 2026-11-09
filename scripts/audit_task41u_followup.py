"""Read-only41u follow-up: backup verification and post-hoc Nap interaction probes."""
import sys,json,hashlib,zipfile,io,os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from research_memory.payload_store import open_readonly
from src.giada_teacher import task41u_objective_features as u

def main():
    torch.set_num_threads(1)
    receipt=json.loads((ROOT/'research_memory/data/local_backup_receipt.json').read_text());db=Path(receipt['local_sqlite'])
    with db.open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==receipt['sqlite_sha256']
    manifest=db.parent/receipt['payload_manifest'];assert hashlib.sha256(manifest.read_bytes()).hexdigest()==receipt['payload_manifest_sha256']
    files=json.loads(manifest.read_text())['files']
    for item in files:
        path=db.parent/'raw_reports'/item['path']
        with path.open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==item['compressed_sha256']
    with open_readonly(ROOT/'research_memory/data/research_memory.sqlite') as c:
        rows=c.execute('SELECT Nome, Stato FROM experiments WHERE [Codice stabile] LIKE ?',('%task41u%',)).fetchall()
        print('DB',rows)
    print('Backup verified: SQLite and',len(files),'payload files')
    archive=ROOT/'artifacts/task41u_kaggle_external/task41u_final.zip'
    with zipfile.ZipFile(archive) as z:
        data=dict(np.load(io.BytesIO(z.read('opened41n_incumbent/fresh_stress_inputs.npz'))))
        archived=np.load(io.BytesIO(z.read('opened41n_incumbent/stress_dual.npz')))['gates']
    os.environ['GIADA_TASK41N_FROZEN']=str(ROOT/'artifacts/task41n_frozen.zip.bin');base=u.n.load_models(torch,ROOT,'cpu')[0]['dual']
    with np.load(ROOT/'experiments/results/task41r_protocol_v1/heads_timing_frozen_k.npz') as a:net=u.r.materialize(torch,base,dict(a))
    print('Existing correction parameters',[(k,tuple(v.shape)) for k,v in net.named_parameters() if 'correction' in k and k.startswith('parent.')])
    v=data['voltage'];unique,ix=np.unique(v,return_inverse=True);ix=ix.reshape(v.shape)
    with torch.no_grad():
        ru,rz=net.raw(torch.tensor(unique)[None].expand(3,-1),torch.full((3,len(unique)),1e-4))
        uu=torch.sigmoid(ru[...,list(u.GATES)]).numpy();zz=torch.sigmoid(rz[...,list(u.GATES)]).numpy()
    tu,tz=u.n.p.coefficients(v.astype('float64'),data['calcium'].astype('float64'),.1)
    truth=u.n.p.readout(v,data['target_gates'],data['multipliers'],u.n.p.ionic.REVERSALS)[1]
    prior=u.n.contract(ROOT)[0]
    interventions={'control':(), 'both_equilibria':(('u',8),('u',9)), 'both_rates':(('z',8),('z',9)), 'activation_pair':(('u',8),('z',8)), 'inactivation_pair':(('u',9),('z',9)), 'all_Nap':(('u',8),('u',9),('z',8),('z',9))}
    if '--nap-factorial' in sys.argv:
        import itertools
        factors=(('u',8),('z',8),('u',9),('z',9))
        interventions={'none' if not subset else '+'.join(k+str(g) for k,g in subset):subset for size in range(5) for subset in itertools.combinations(factors,size)}
    for label,changes in interventions.items():
        state=np.broadcast_to(data['initial'][:,u.GATES],(3,len(data['initial']),4)).copy();trace=[]
        for tick in range(len(v)):
            a=uu[:,ix[tick]].astype('float64');b=zz[:,ix[tick]].astype('float64')
            for kind,gate in changes:(a if kind=='u' else b)[:,:,u.GATES.index(gate)]=(tu if kind=='u' else tz)[tick,:,gate]
            state+=b*(a-state);trace.append(state.copy())
        trace=np.stack(trace);failed=[]
        for si,seed in enumerate(prior['seeds']):
            gates=archived[:,si].astype('float64');gates[...,list(u.GATES)]=trace[:,si]
            current=u.n.p.readout(v,gates,data['multipliers'],u.n.p.ionic.REVERSALS)[1]
            for case in range(v.shape[1]):
                m=u.n.p.evaluate_currents(truth[:,case],current[:,case],.1,prior['quality'],data['target_gates'][:,case],gates[:,case])
                if not m['passed']:failed.append((seed,case,[x['channel'] for x in m['channels'] if not x['passed']]))
        print('POSTHOC',label,'passed',72-len(failed),'failed',failed)

if __name__=='__main__':main()
