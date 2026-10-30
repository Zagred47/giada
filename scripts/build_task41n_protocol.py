"""Seal new native protocols, frozen candidate and bitwise execution contrasts."""
import hashlib,json,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def main():
    prior=json.loads((ROOT/'experiments/task41m_preregistration_v1.json').read_text())
    native=json.loads((ROOT/'experiments/task41c_preregistration_v1.json').read_text())
    audit=json.loads((ROOT/'experiments/results/task41m_audited_v1/audit.json').read_text())
    source=ROOT/audit['archive'];assert sha(source)==audit['sha256']
    with zipfile.ZipFile(source) as z:raw=z.read('frozen_models.pt')
    package=ROOT/'artifacts/task41n_frozen.zip.bin'
    with zipfile.ZipFile(package,'w',zipfile.ZIP_DEFLATED,compresslevel=1) as z:z.writestr('frozen_models.pt',raw)
    protocols=['jittered_triplet','late_rebound','notched_plateau','spaced_rapid_train']
    pulses=dict(jittered_triplet=[(4.4,6.1,.0098),(11.6,14.1,.0102),(20.8,22.,.0096)],late_rebound=[(3.2,8.6,-.0031),(12.3,14.9,.0103),(20.1,21.7,.0101)],notched_plateau=[(4.1,7.4,.0032),(7.4,10.6,.0088),(10.6,13.2,.0042),(17.4,19.6,.0102)],spaced_rapid_train=[(5.2,5.9,.011),(7.8,8.5,.0108),(11.9,12.6,.0112),(17.1,17.8,.0106)])
    schedules={'calibration':native['schedules_ms']['calibration']}
    for k,name in enumerate(protocols):
        offset=.17*k
        schedules[name]={receptor:[round(x+offset,3) for x in times] for receptor,times in dict(exc_1=[2.65,3.55,11.25,20.65],exc_2=[2.95,3.85,11.65,21.05],inh_1=[6.35,8.65,17.75,26.25],inh_2=[6.75,9.05,18.15,26.65]).items()}
    excluded=set()
    def walk(value,seed_context=False):
        if isinstance(value,dict):
            for key,v in value.items():walk(v,seed_context or 'seed' in key.lower())
        elif isinstance(value,list):
            for v in value:walk(v,seed_context)
        elif seed_context and isinstance(value,int):excluded.add(value)
    for path in ROOT.joinpath('experiments').glob('*preregistration*.json'):
        if path.name!='task41n_preregistration_v1.json':walk(json.loads(path.read_text()))
    new=[9410117,9410129,9410143,9410159,9410173,9410003];assert not set(new)&excluded
    spec=dict(task='41n',version='v1',candidate='Frozen41m Ca_HVA-only (swap100)',declared_candidate_policy='voltage_solve64',
        training_updates=0,selection_performed=False,learned_compression=False,model_seeds=[17,29,43],seeds=[17,29,43],
        native_seeds=new[:2],stress_seeds=new[2:5],calibration_seed=new[5],parity_seed=9419001,excluded_input_seeds=sorted(excluded),
        dt_ms=.1,duration_ms=40.,stress_duration_ms=1200.,initial_voltages_mv=[-77.,-52.],protocols=protocols,pulses=pulses,schedules_ms=schedules,
        panels=native['panels'],interventions=native['interventions'],neuron_version=native['neuron_version'],teacher_revision=native['teacher_revision'],
        receptor_time_semantics=native['receptor_time_semantics'],quality=prior['quality'],current_quality=prior['quality'],compartment_gates=prior['compartment_gates'],cost=prior['cost'],
        stress_families=['doublepulse_recovery','slow_boundary_return','calcium_swing','near_threshold_staircase'],
        execution_variants=['dual','donor_pruned','shared_features'],parity=dict(atol=0,rtol=0,required='Bitwise logits,gate step,coupled traces and stress states. Source eligibility frozen before fresh data. No acceptance from a quality tolerance.'),
        frozen_input=dict(local=package.relative_to(ROOT).as_posix(),sha256=sha(package),bytes=package.stat().st_size,checkpoint_sha256=hashlib.sha256(raw).hexdigest(),origin_archive_sha256=audit['sha256']),
        gpu='Tesla T4',torch='2.11.0+cu130',max_worker_seconds=5400,
        hypotheses=[dict(id='independent',statement='Frozen Ca-only mixed-precision function generalizes to new native protocols,initial voltages and RNG seeds.',falsifier='Any registered fresh trajectory,response or stress gate fails against either originalFP32 or matched canonical reference; no fitting after opening.'),
                    dict(id='pruning',statement='Unused donor SK/Nap precision branches can be skipped while retaining identical logits and dynamics.',falsifier='Any bit differs on preflight or fresh parity; implementation becomes ineligible regardless of speed.'),
                    dict(id='feature_reuse',statement='Input normalization/identical hat features can be computed once across frozen parent/donor without sharing their different weighted activations.',falsifier='Any bit differs or no paired speed benefit; preserve full GEMM widths and reduction order.'),
                    dict(id='cost',statement='Output-preserving execution changes remove measurable duplicate work without new learned compression.',falsifier='No improvement >=10% with positive paired uncertainty versus dual at642/4096; separate original formula >=10% deployment speed gate remains unchanged.')],
        qualification='Primary independent quality requires ALL176x3 trajectories and112x3 responses in BOTH modes and24x3stress histories, original AND matched reference. Native/shadow/phase/canonical floors required. Implementation admissibility requires exact parity. Original cost thresholds unchanged and separate; Task42 never automatic.',
        limits=['Modelseeds17/29/43 remain frozen; only stimulus/release seeds are new. No model ranking on fresh outcomes.',
                'Primary mixed voltage/solve64 is declared prospectively, not an unnoticed waiver of oldFP32. Gate network,calcium and18occupancies stayFP32.',
                'No new analytic kinetic fallback,optimizer,training,weight rounding or representation change.',
                'Single isolated compartment,40msnative and1200ms imposed histories; no multicompartment or untested-horizon claim.',
                'All implementations benchmarked even if inaccurate only as diagnostics; no speedup claim without independent quality and parity.',
                'NEURON8.2.7 runs in a separate Python3.12 CPU environment; GPU/Torch runs in current Colab Python. No native/CUDA memory sharing.',
                '90-minute bounded worker;partial archives every180seconds,final traces included;DBbackup local only.'])
    notebook=ROOT/'notebooks/giada_task41n_independent_exact_execution.ipynb'
    cells=[dict(cell_type='markdown',metadata={},source=['# GIADA41n — Conferma indipendente ed esecuzione equivalente\n','176 nuovi casi nativi,72valutazioni stress; pesi congelati. Tre implementazioni, soglie originali. Nessun training.\n','La console esegue questo stesso worker; non lanciare un secondo processo.']),dict(cell_type='code',metadata={},execution_count=None,outputs=[],source=["from pathlib import Path\nimport json,subprocess,sys\nrequest=json.loads(Path('/content/giada_task41n_launch.json').read_text())\nroot=Path(request['remote_root'])\nassert not (root/'artifacts/execution_freeze.json').exists(), 'Run console già avviato'\nsubprocess.run([sys.executable,'-u',str(root/'giada/scripts/run_task41n_independent.py'),str(root/'artifacts'),'--phase','preflight'],check=True)\n# Native e valutazione sono orchestrati dal manager console, con backup esterni.\n"])]
    notebook.write_text(json.dumps(dict(cells=cells,metadata=dict(kernelspec=dict(display_name='Python3',language='python',name='python3')),nbformat=4,nbformat_minor=5),indent=2),encoding='utf-8',newline='\n')
    sources=['src/giada_teacher/task41n_exact_execution.py','src/giada_teacher/task41n_independent_confirmation.py','scripts/run_task41n_independent.py','scripts/build_task41n_protocol.py','notebooks/giada_task41n_independent_exact_execution.ipynb','tests/test_task41n.py']
    spec['sources']={n:sha(ROOT/n) for n in sources}
    (ROOT/'experiments/task41n_preregistration_v1.json').write_text(json.dumps(spec,indent=2),encoding='utf-8',newline='\n')
    print('Sealed41n, new input seeds disjoint from previous protocols; fixed model seeds.')

if __name__=='__main__':main()
