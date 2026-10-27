"""Complete immutable 41k results and prospective 41l design, SQLite only."""
import gc
import hashlib
import json
import sys
import subprocess
import zipfile
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from .mirror import ROOT, write_json
from .incremental_batch import IncrementalBatchMirror


def main():
    repo = ROOT.parent
    folder = repo / 'experiments/results/task41k_audited_v1'
    audit = json.loads((folder / 'comparison_with_J.json').read_text())
    report = audit['report']
    assert audit['valid'] and audit['complete'] and report['valid']
    assert report['training_updates_performed'] == 0 and not report['selection_recomputed']
    assert not report['thresholds_changed'] and not report['task42_authorized']
    archive = Path(audit['backup']['local'])
    with archive.open('rb') as stream:
        assert hashlib.file_digest(stream, 'sha256').hexdigest() == audit['backup']['remote']['sha256']
    progress_path = folder / 'registration_progress.json'
    progress = json.loads(progress_path.read_text()) if progress_path.exists() else dict(raw=[], records={}, batches=[])
    if progress.get('complete') and '--refresh-structured' not in sys.argv:
        print('Task41k results and proposed41l already registered'); return
    baseline = progress.get('verification', json.loads((repo / 'experiments/results/task41k_protocol_v1/recovery_memory_receipt.json').read_text())['verification'])
    m = IncrementalBatchMirror(); ids = progress['records']
    assert m.baseline_hash == baseline['snapshot_hash']

    def put(table, suffix, **fields):
        code = f'{table}-giada-task41k-result-{suffix}-v1'
        rid = m.local_upsert(table, {'Codice stabile': code, **fields})['record_id']
        ids[code] = rid
        return rid

    @lru_cache(None)
    def ref(code):
        if code in ids: return ids[code]
        rows = m.query("SELECT record_id FROM v_records WHERE stable_code='" + code + "'")['rows']
        assert len(rows) == 1, code
        return rows[0]['record_id']

    def flush():
        nonlocal m, baseline
        print('Committing bounded batch and verifying complete reciprocal readback', flush=True)
        change = m.commit_batch(); baseline = m.verify_incremental(baseline)
        assert baseline['valid']
        progress['verification'] = baseline; progress['batches'].append(change)
        write_json(progress_path, progress)
        m = IncrementalBatchMirror(); gc.collect()
        print('Verified original reports: ' + str(len(progress['raw'])), flush=True)

    aid = put('artifacts', 'archive', Nome='Task41k complete final recovery v5', Tipo='Report', Versione='v5',
              Percorso=archive.relative_to(repo).as_posix(), **{'SHA-256': audit['backup']['remote']['sha256'], 'Dimensione byte': archive.stat().st_size})
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        members = sorted((x for x in z.infolist() if x.filename.endswith('.json')), key=lambda x: x.file_size)
        done = {r['member'] for r in progress['raw']}; staged = 0
        for member in members:
            if member.filename in done: continue
            if staged and staged + member.file_size > 300_000_000:
                flush(); staged = 0
            raw = z.read(member); digest = hashlib.sha256(raw).hexdigest(); name = member.filename
            art = put('artifacts', 'member-' + name, Nome='Task41k original ' + name, Tipo='Report', Versione='v1',
                      Percorso=archive.relative_to(repo).as_posix() + '#' + name,
                      **{'SHA-256': digest, 'Dimensione byte': len(raw)})
            obs = put('observations', 'raw-' + name, Nome='Task41k complete original JSON ' + name,
                      Descrizione=raw.decode('utf-8'), **{'Artefatti dettagliati': [aid, art]})
            progress['raw'].append(dict(member=name, observation=obs, sha256=digest, bytes=len(raw)))
            staged += len(raw); del raw
            print('Staged ' + name + ' ' + str(member.file_size), flush=True)
        if staged: flush()
        assert {r['member'] for r in progress['raw']} == {x.filename for x in members}
    audit_arts=[aid]
    for path in (folder/'comparison_with_J.json',repo/'scripts/audit_task41k_final.py'):
        content=path.read_bytes()
        audit_arts.append(put('artifacts','audit-'+path.name,Nome='Task41k '+path.name,Tipo='Codice' if path.suffix=='.py' else 'Report',Versione='v1',Percorso=path.relative_to(repo).as_posix(),
            **{'SHA-256':hashlib.sha256(content).hexdigest(),'Dimensione byte':len(content)}))
    obs = put('observations', 'paired-audit', Nome='Task41k complete same-panel audit against H and J',
              Descrizione=json.dumps(audit, ensure_ascii=False), **{'Artefatti dettagliati': audit_arts})
    exp = ref('experiments-giada-task41k-comparison-v1')
    limits = 'Opened panels diagnostic, not independent confirmation. Development differs J/K: use same-run H. Failed evaluations are not unique biological cases. 1024 updates cannot exclude data/budget limitations. Thresholds and abstraction unchanged.'
    statements = {
        'outcome': 'Technical valid, scientific LEARNING_ALIGNMENT_NO_GO. H6,point6,oldmix6,uniform6,uniform-retain8,target8,target-retain7,copy-coeff496,copy-dynamic623 failed evaluations. No qualified model or Task42 authorization.',
        'local-repair': 'K_dyn_target fixes both Ca_HVA slow_boundary_return stress41c failures seed17/29, but introduces Nap_Et2 stress41/seed17 and autonomous41/case120/seed29, autonomous41c/case122/seed29, plus K_Tst recurrence development/case0/seed29. Two fixed and four introduced; no net global gain.',
        'retention': 'Unit retention copies H on lower-half FIT query error, not guaranteed complete passed trajectories; uniform-retain8 and target-retain7 do not provide global repair. This does not disprove canonical worst-group protection.',
        'selection': 'Point/oldmix select0/0/0,uniform and uniform-retain0/16/0,target256/1024/0,target-retain0/1024/0. Compact coeff64/1024/16,dynamic16/0/16. Later training exists even where selection returns0. All source checkpoints finite.',
        'probe': 'Initial coefficient/dynamic gradient cosines 0.03225645,0.02461166,0.00288776 on small probe: near orthogonal, not negative. Normalized dynamic step1e-4 improves probe loss on all seeds,1e-3 worsens all. Supports testing step amplitude/curvature, not proof of actual Adam overshoot or universal conflict.',
        'numerical-interaction': 'Frozen voltage_precision.json:41c autonomous failed responses H/nominee primary1/2,solve64-only2/3,voltage64-only1/2,both64 0/0,recentered1/3. Panel41 nominee primary1,solve64 0,voltage64-only1,both64 0,recentered0. Both new autonomous regressions disappear with joint voltage/solve precision; not proof stress/recurrence pass or justification for waiving FP32. Need paired canonical numerical-policy reference/floor to attribute learning versus numerical interaction. Calcium/gates/network are not all promoted toFP64.',
        'compression': 'Pure parent-copy compression remains inaccurate even relative to H, which already fails canonical gates. K compact budget1024 versus J distillation4096+60000 refinement is not budget matched. Capacity impossibility not established; repair and compression remain separate.',
        'cost': 'Batch642 per seed ionic formula/model ratios H0.839,repair0.834-0.838,compact4.007/4.054; integrated H0.828,repair0.828-0.831,compact1.987/1.989. Fast compact models fail quality; no qualified speedup. Within-run ratios, not paired cross-run hardware claim.',
        'recovery': 'Recovered full frozen evaluation without retraining or reselection; version5 completed all suites,cost and parity. Historical failure_report.json retained as provenance, superseded by final_report.json, not evidence final run failed. All PT tensors finite; archive SHA/CRC verified outside runtime.',
    }
    numeric_obs=put('observations','numerical-response-summary',Nome='Task41k frozen numerical-policy response counts',
        Descrizione=json.dumps(dict(policy_order=['primary','solve64_only','voltage64_only','voltage_and_solve64','recentered'],
            failed_responses={'41/H':[0,0,0,0,0],'41/nominee':[1,0,1,0,0],'41c/H':[1,2,1,0,1],'41c/nominee':[2,3,2,0,3]},
            scope='Autonomous response counts only, not full-suite qualification. Nominee=K_dyn_target; unchanged source thresholds; source numerical carrier leaves network/calcium/gates FP32. Matched canonical-policy reference control still needed for attribution.')),
        **{'Artefatti dettagliati':[aid,ref('artifacts-giada-task41k-result-member-selected_numerical_audit/voltage_precision.json-v1')]})
    findings = {}
    for key, text in statements.items():
        evidence_obs=[obs]
        if key=='numerical-interaction':
            evidence_obs.append(numeric_obs)
        if key=='probe':
            evidence_obs.append(ref('observations-giada-task41k-result-raw-gradient_alignment_probe.json-v1'))
        findings[key] = put('findings', key, Nome='Task41k ' + key, Esito='Negativo' if key=='outcome' else 'Misto',
                            Esperimenti=[exp], Osservazioni=evidence_obs, Risultato=text, Limitazioni=limits)
    date = datetime.now(timezone.utc).date().isoformat()
    for hypothesis, finding in [('objective_conflict','probe'), ('forgetting','retention'), ('coverage','local-repair'), ('compression_objective','compression')]:
        outcome='Contraddice' if hypothesis=='compression_objective' else 'Indeterminata'
        argument=statements[finding]
        if hypothesis=='compression_objective':
            argument+=' Selected parent dynamic scores, lower better: coeff[0.0797165,0.0371090,0.0390911],dynamic[0.4795429,0.3894201,0.2940421]. Contradiction scoped to selected models under this1024-update protocol, not all budgets/architectures.'
            m.local_upsert('claims',{'Codice stabile':'claims-giada-task41k-compression_objective-v1','Stato':'Contraddetta nel dominio','Limiti':argument+' '+limits})
        put('evidence', hypothesis, Nome='Task41k scoped assessment ' + hypothesis, Esito=outcome,
            **{'Affermazione valutata':[ref('claims-giada-task41k-'+hypothesis+'-v1')], 'Risultati a sostegno':[findings[finding]],
               'Argomentazione':argument, 'Limiti e spiegazioni alternative':limits, 'Data valutazione':date})
    spec = json.loads((repo / 'experiments/task41k_preregistration_v1.json').read_text())
    run_ids = []
    for arm in spec['arms']:
        name = arm['name']
        for seed in spec['seeds']:
            code = f'runs-giada-task41k-{name}-{seed}-v1'
            run_ids.append(ref(code))
            m.local_upsert('runs', {'Codice stabile':code, 'Stato':'Completata', 'Artefatti prodotti':[aid],
                'Validità tecnica':'All saved PT tensors finite;1024 training updates;complete frozen evaluation with0 additional training updates and no reselection. Scientific NO-GO; see per-arm full observations.'})
        put('observations','aggregate-'+name,Nome='Task41k detailed aggregate '+name,
            Descrizione=json.dumps(dict(summary=report['summaries'][name],failures=audit['failures'][name],changes=audit['changes_vs_H'].get(name),selected_steps=audit['chosen_steps'].get(name),contributing_runs=run_ids[-3:])),
            **{'Artefatti dettagliati':[aid], 'Specifica di valutazione':[ref('evaluations-giada-task41k-quality-v1')]})
    put('analyses','full-audit',Nome='Task41k full repair/compression audit',Origine='Post hoc',Versione='v1',
        Procedura='scripts/audit_task41k_final.py; full source JSON plus paired case identities',Configurazione=limits,
        **{'Run analizzati':run_ids,'Input artefatti':[aid],'12 · Risultati scientifici':list(findings.values())})
    m.local_upsert('experiments', {'Codice stabile':'experiments-giada-task41k-comparison-v1','Stato':'Concluso'})
    design = repo / 'experiments/task41l_design.md'; raw = design.read_bytes()
    did = put('artifacts','41l-design',Nome='Task41l prospective regression attribution design',Tipo='Report',Versione='v1',Percorso=design.relative_to(repo).as_posix(),**{'SHA-256':hashlib.sha256(raw).hexdigest(),'Dimensione byte':len(raw)})
    put('observations','41l-design',Nome='Task41l complete proposed scientific protocol',Descrizione=raw.decode('utf-8'),**{'Artefatti dettagliati':[did]})
    claims = []
    for key, statement, falsifier in [
        ('step','Some regressions arise from excessive finite update size, not a useless local direction.','Matched lower-LR arms and fractional-update probes do not reduce regressions at equal progress; slower learning alone is not support.'),
        ('coverage','Current FIT lacks sufficient diversity in rare sensitive regimes.','Nested4x support at matched updates gives no consistent benefit; this weakens this coverage change, not all possible data interventions.'),
        ('protection','Mean dynamic objective sacrifices channel/regime groups; canonical worst-group protection can preserve them.','Protection loses the Ca correction or introduces as many new failures; merely staying at H is not repair.'),
        ('feedback','Some Nap/K failures originate from altered feedback trajectories rather than intrinsic channel errors.','Defects persist with reference-imposed V/Ca and follow the channel replacement independently of Ca feedback; use factorial interactions, not single swaps alone.'),
        ('numerical-interaction','Voltage accumulation and membrane-solve precision interact with learned corrections in autonomous response failures.','Matched canonical-policy floors and continuous-error audits show no causal numerical interaction; no claim from a changed pass/fail alone, no primary-policy waiver.')]:
        code='claims-giada-task41l-'+key+'-v1'
        rid=m.local_upsert('claims',{'Codice stabile':code,'Nome':'Task41l '+key,'Tipo':'Ipotesi','Stato':'Aperta','Enunciato':statement,'Condizioni di falsificazione':falsifier,'Limiti':limits+' Prospective, untested.'})['record_id'];claims.append(rid)
        put('evidence','41l-'+key,Nome='Task41l motivation '+key,Esito='Indeterminata',**{'Affermazione valutata':[rid],'Risultati a sostegno':list(findings.values()),'Argomentazione':'Motivation, not causal confirmation. '+statement,'Limiti e spiegazioni alternative':falsifier,'Data valutazione':date})
    next_exp=m.local_upsert('experiments',{'Codice stabile':'experiments-giada-task41l-regression-attribution-v1','Nome':'Task41l regression causes, repair only','Stato':'Progettato','Ipotesi':claims,'Descrizione':'PROPOSED DESIGN ONLY. Frozen channel-factorial attribution plus2x2x2 learning-rate/support/canonical-protection matrix. No executable preregistration, no training launched.','Obiettivo informativo':'Separate feedback mediation, finite step size, coverage and group sacrifice; compression parked.'})['record_id']
    def prospective(table, suffix, **fields):
        code=f'{table}-giada-task41l-{suffix}-v1'
        rid=m.local_upsert(table,{'Codice stabile':code,**fields})['record_id'];ids[code]=rid
        return rid
    protocol=prospective('protocols','design',Nome='Task41l proposed matched regression attribution',Versione='draft-v1',Modalità='Ricomposizione',Esperimento=[next_exp],
        Procedura=raw.decode('utf-8'),**{'Criteri di successo':'Exact41k quality/compartment/cost predicates; all gates required. No independent confirmation on opened panels.',
        'Regole di arresto':'No launch before executable preregistration and CPU/GPU preflight. Nonfinite parameters, missing source hashes or loss of paired controls stop interpretation.'})
    levels={}
    for label,values,meaning in [
        ('amplitude',('1e-5','1e-6'),'Initial Adam learning rate; identical exponential decay through1024 then floor through4096. Finite-step amplitude versus slow learning.'),
        ('coverage',('1','4'),'Nested FIT support multiplier, fixed family probabilities and targeting cutoff from1x; diversity versus update budget.'),
        ('protection',('off','canonical-worst-group-weight1'),'Add canonical worst channel/family group loss; do not copy H or change scientific thresholds.')]:
        factor=prospective('factors',label,Nome='Task41l '+label,Descrizione=meaning,Protocollo=[protocol],**{'Interpretazione causale':meaning+' Proposed, not executed; interactions measured in complete2x2x2.'})
        for index,value in enumerate(values):
            levels[(label,index)]=prospective('factorlevels',label+'-'+str(index),Nome='Task41l '+label+' '+value,Valore=value,Fattore=[factor],Descrizione=meaning)
    aa={}
    for amplitude in (0,1):
        for data in (0,1):
            for protect in (0,1):
                name=f'L_A{amplitude}D{data}P{protect}'
                config=dict(initial_lr=1e-5 if amplitude==0 else 1e-6,support_multiplier=1 if data==0 else 4,
                    canonical_worst_group_protection=bool(protect),parent='H_nominee',compression=False,
                    checkpoints=[0,16,64,256,1024,4096],status='proposed_not_implemented')
                aa[(amplitude,data,protect)]=prospective('arms',name,Nome='Task41l '+name,Ruolo='Trattamento',Protocollo=[protocol],**{'Configurazione residua':json.dumps(config)})
                for label,index in [('amplitude',amplitude),('coverage',data),('protection',protect)]:
                    prospective('assignments',name+'-'+label,Nome='Task41l '+name+' '+label,Braccio=[aa[(amplitude,data,protect)]],Livello=[levels[(label,index)]],Descrizione='Prospective factorial assignment, no execution yet.')
                for seed in (17,29,43):
                    prospective('runs',name+'-'+str(seed),Nome=f'Task41l {name} seed{seed}',Stato='Pianificata',Seed=str(seed),Braccio=[aa[(amplitude,data,protect)]],
                        **{'Configurazione effettiva':json.dumps(config),'Validità tecnica':'Not executed; CPU/GPU preflight and executable preregistration pending.'})
    for vertex,arm in aa.items():
        for axis,label in enumerate(('amplitude','coverage','protection')):
            if vertex[axis]: continue
            other=list(vertex);other[axis]=1
            prospective('contrasts',label+'-'+''.join(map(str,vertex)),Nome='Task41l '+label+' paired '+str(vertex),Protocollo=[protocol],Bracci=[arm,aa[tuple(other)]],
                **{'Contrasto e coefficienti':'Treatment minus control at fixed other factors; seed/case paired metrics, new/resolved/remaining failures and training exposure.',
                   'Soglia interpretativa':'Original scientific thresholds unchanged. Report effect sizes and all3seeds; no global causal claim from aggregate pass counts.',
                   'Confondenti controllati':'Same H,architecture,optimizer schedule,selection,evaluation; unequal data contents only on D and unequal objective compute on P reported explicitly.'})
    put('decisions','next41l',Nome='Retain H; prepare41l, park compression',Esito='Continuare',Risultati=list(findings.values()),Motivazione='User requests recording and next experiment preparation. Scientific design prepared; implementation/preflight/preregistration required before launch. No push or launch performed.',**{'Data':date})
    put('lessons','distinguish-regression',Nome='Separate repaired channels from induced feedback regressions',Lezione='Use paired learned-coefficient transplants and imposed/autonomous feedback controls; inspect update amplitude and group effects. Do not equate local repair with global GO, gradient cosine with causality, or fast inaccurate compression with qualified acceleration.',Risultati=list(findings.values()),Limiti=limits)
    flush()
    progress['complete']=True;write_json(progress_path,progress)
    write_json(folder/'memory_receipt.json',dict(valid=True,verification=baseline,records=ids,raw_reports=progress['raw'],completed_runs=24,
        raw_original_bytes_verified=True,airtable_accessed=False,next_experiment=next_exp,next_status='scientific_design_only_not_launched',DB_backup='local refresh pending'))
    m.export_snapshot_to(ROOT/'data/airtable_snapshot.json')
    print('Complete41k and prospective41l registered, verified and exported',flush=True)
    if '--backup-after' in sys.argv:
        subprocess.run([sys.executable,'-m','research_memory.prepare_drive_backup'],cwd=repo,check=True)
        subprocess.run([sys.executable,'-m','research_memory.record_local_backup','--raw-sqlite','--scope',
            'Complete41k final49original JSON reports,24completed runs,paired audit and scoped evidence including numerical-policy interaction;41l proposed design,5hypotheses,8planned arms and12matched contrasts. No41l training, no remote launch;thresholds and abstraction unchanged.'],cwd=repo,check=True)
        receipt=json.loads((folder/'memory_receipt.json').read_text())
        receipt['DB_backup']=json.loads((ROOT/'data/local_backup_receipt.json').read_text())
        assert receipt['DB_backup']['valid'] and not receipt['DB_backup']['drive_updated']
        write_json(folder/'memory_receipt.json',receipt)
        print('Final full SQLite backup verified locally; no Drive access',flush=True)


if __name__=='__main__': main()
