"""Finalize45c numeric contract and immutable compact native-fixture transport."""
import hashlib,json,sys,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from research_memory.mirror import write_json

def main():
    d=json.loads((ROOT/'experiments/task45c_design_v1.json').read_text());old=json.loads((ROOT/'experiments/task45b_preregistration_v1.json').read_text())
    receipt=json.loads((ROOT/'artifacts/task45b_kaggle_external/external_backup_receipt.json').read_text())
    source=Path(receipt['local']);folder=ROOT/'artifacts/task45c_input';folder.mkdir(exist_ok=True)
    destination=folder/'native45c.zip.bin';members={}
    with source.open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==d['baseline']['source_archive_sha256']
    with zipfile.ZipFile(source) as src:
        names=[n for n in src.namelist() if n=='calibration_freeze.json' or n in ('frozen41r.pt','f011.pt','f111.pt') or (n.endswith('.npz') and '/native_' in n) or n.endswith('/native_floor.json')]
        if destination.exists():
            with zipfile.ZipFile(destination) as z:
                assert set(z.namelist())==set(names) and z.testzip() is None
                for name in names:
                    digest=hashlib.sha256(src.read(name)).hexdigest();assert hashlib.sha256(z.read(name)).hexdigest()==digest;members[name]=digest
        else:
            with zipfile.ZipFile(destination,'w',compression=zipfile.ZIP_STORED,allowZip64=True) as z:
                for name in names:
                    data=src.read(name);members[name]=hashlib.sha256(data).hexdigest();z.writestr(name,data)
    with destination.open('rb') as f:digest=hashlib.file_digest(f,'sha256').hexdigest()
    transport=dict(valid=True,mode='existing_Kaggle_output_readonly_no_payload_upload',source45b_sha256=d['baseline']['source_archive_sha256'],
        source45b_bytes=receipt['remote']['bytes'],kernel_source='alessandrobelli/giada-task45b-rate-coverage',mounted_archive='task45b_final.zip',
        archive='native45c.zip.bin',local=destination.relative_to(ROOT).as_posix(),bytes=destination.stat().st_size,sha256=digest,members=members,
        local_payload_upload_blocked=True,local_payload_uploaded=False,
        rationale='Conservative alternative to denied local archive upload: read existing user-owned45b persisted output within Kaggle. No native payload exported from local machine. CPU mounted preflight mandatory.')
    write_json(ROOT/'experiments/task45c_transport_v1.json',transport)
    p=dict(d,version='v1',status='PREREGISTERED_EXECUTABLE',remote_execution_authorized=True,notebook_implemented=True,
        execution_authority='User explicitly authorized notebook implementation, DB registration, Git push and one Kaggle run; local-only DB backup.',
        cost=dict(batches=[1,642,4096],ticks=80,repetitions=30,minimum_paired_saving=.1,
            scope='Original resident single-compartment ionic/integrated CUDA Graph benchmark, all materialized candidates; formula32/formula64 and H controls. Separate from13-arm coupled evaluator overhead; no automatic speed qualification.'),
        source_transport=transport,prior45b='experiments/task45b_preregistration_v1.json',
        panels=old['panels'],duration_ms=1000.,prefix_horizons_ms=[100.,500.,1000.],fitting_horizons_ms=[1.,8.,40.,100.,500.,1000.],geometries=old['geometries'],coupling=old['coupling'],
        original45_protocols=old['original45_protocols'],calibration=old['calibration'],
        model_arms=['C0','N00','N01','N10','N11','I0','I1','L0','L1','J','R41','R111'],
        coupled_arms=['none','C0','N00','N01','N10','N11','I0','I1','L0','L1','J','R41','R111'],
        fitting=dict(points=2048,nominal_fraction=.5,coefficient_seed=9453001,paths_per_role=24,duration_ms=1000.,
            fit_seeds=[9453101,9453113,9453127],dev_seeds=[9453201,9453211,9453229],
            families=['doublepulse','staircase','tail_hold','tail_ramp'],initial_voltages_mv=[-95.,-42.],
            checkpoints=[1,4],sample_ticks=128,coefficient_scale=1.,ridge=1e-3,whitening_floor=1e-5,max_whitened_step_norm=2.,
            current_weight=1.,gate_weight=1.,crossing_weight=1.,imitation_weight=.1,
            precision='FP64 NumPy causal sensitivity/linear solves; frozen FP32 quantized features/logits. Check actual FP32 materialized forward on DEV at checkpoints1/4. No optimizer state sharing or hidden resets.',
            objective='Equal macro-normalized coefficient rows plus selected path rows. N00/N10 coefficient-only; N01/N11 add nominal teacher gate/current preservation and bounded f011-current imitation on nominal FIT histories. I0 coefficient-only; I1 adds original10%-transient crossing-neighbor current rows. L0/L1 add gate and current path rows on all FIT histories. Same data and unchanged evaluator.',
            sensitivities='Exact causal recurrence Jacobian for existing h/u/z readout heads on imposed FIT paths, with frozen other gates. Regularized whitened Gauss-Newton, independent seed solves, step norm cap2. No new features, layers or runtime coefficient formulas.',
            nominal_guard='Teacher-current/gate FIT rows, not a promise of no regressions. Actual all-channel previously-passing DEV identities must remain passed; reject candidates introducing any new DEV failure.',
            nomination='For each family: candidates with no new all-channel/prefix failure vs C0 on fresh DEV. Rank highest all-prefix TARGET current+gate pass count, then smallest target worst normalized severity, earlier checkpoint/registered order. C0 fallback. Freeze component choices before coupled audits. J combines only these frozen heads. Report whole-model DEV outcomes as well; target acquisition never presented as a global pass.'),
        implementation_scope='New fixed-V/Ca histories are synthetic FIT/DEV only. Opened45b native/calibration fixtures reused verbatim, checksummed. No native regeneration, no Task47 data. Native reference floors and physical-domain checks still required for interpreting all coupled results.',
        regression_guards=['C0 must exactly reconstruct f011 logical hash; six41r failures and FP32 controls218/234 remain explicit.',
            'No NaTs2_t.h weight changes; its index7 is probed as attribution control, Ca_LVAst.h is index3.',
            'Report all100/500/1000ms prefixes; no averaging away intermediate failures.',
            'Old original45 invalid physical regimes stay diagnostic, never counted as model passes.',
            'All13 arms evaluated at identical arm/case/seed batch shape; all-formula null control and packed/materialized parity mandatory.',
            '41 stress and44 regressions are opened audits after nomination, never selection input.'],
        persistence='Persist source/member checks, fitted checkpoints and nomination before coupled phase; results per geometry/mode. Native source cache excluded from final ZIP. Full metrics plus V/Ca and target gate3/7/9/10/current1/4/5 for all arms; full18/11 tensors for C0/J. Partial/final ZIP on failure/success, Kaggle persisted output and local SHA/CRC readback. DB local2 verified bundles.',
        runtime_budget='One private T4 run, worker5400seconds, outer7200seconds. No automatic relaunch on failure. FIT/DEV Jacobian solves on CPU avoid long-BPTT overhead; CUDA resident comparisons and shared seed/arm batches. Eager execution timings descriptive, no speedup claim.',
        limits=['Exploratory45c, not independent47 or qualified release.','No thresholds, state representation, inference equations or abstraction changes.','Fresh DEV labels only nominate targeted repairs; all-model coupled and historical regressions can still reject them.','Original single-transient evaluator unchanged; signed/slope probes diagnostic only.'])
    write_json(ROOT/'experiments/task45c_preregistration_v1.json',p)
    print('45c immutable fixtures '+str(transport['bytes'])+' bytes; protocol finalized.',flush=True)

if __name__=='__main__':main()
