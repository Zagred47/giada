"""CPU-only clean Python/NEURON compile+native smoke before46GPU dispatch."""
import json,subprocess,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import task46_kaggle as launch
from research_memory.mirror import write_json
ROOT=launch.ROOT;F=launch.F
def main():
    from kaggle.api.kaggle_api_extended import KaggleApi
    api=KaggleApi();api.authenticate();folder=ROOT/'artifacts/task46_native_env';folder.mkdir(exist_ok=True);marker=F/'native_environment_launch.json'
    if not marker.exists():
        assert not (F/'native_environment_pending.json').exists(),'Inspect uncertain CPU preflight; do not duplicate'
        revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
        probe="""import sys,json,numpy as np,neuron
from pathlib import Path
from neuron import h,load_mechanisms
from src.giada_teacher import task46_spatial_cost as a
from src.giada_teacher import task32_dynamic_calcium_feedback as cad
root=Path.cwd();teacher=root.parent/'teacher';out=Path('/kaggle/working/native_probe');out.mkdir()
assert sys.version_info[:3]==(3,12,11) and neuron.__version__.split('+')[0]=='8.2.7'
p,cfg,ca,base=a.spec(root);build,hashes=cad.compile_native(teacher,out,ca);load_mechanisms(str(build.resolve()))
checks=[]
for geo in p['geometries']:
 rows,inj,mult=a.design(geo,dict(p,duration_ms=1.))
 for coupling in (0.,.25):
  index=next(i for i,r in enumerate(rows) if r['coupling']==coupling)
  trace=a.x.native_episode(h,geo,rows[index],inj[:,index],mult[index],cfg,ca,base)
  assert all(np.isfinite(v).all() for v in trace.values())
  checks.append(dict(geometry=geo['id'],coupling=coupling,finite=True,voltage_shape=list(trace['voltage'].shape)))
Path('/kaggle/working/native_environment46.json').write_text(json.dumps(dict(valid=True,Python=sys.version,NEURON=neuron.__version__,numpy=np.__version__,checks=checks,mechanism_hashes=hashes,GPU_used=False,scientific_task46_results=False)))
print('Clean native46environment compiled and smoke passed')
"""
        code="""import sys,subprocess,os
from pathlib import Path
work=Path('/tmp/native46_env');work.mkdir();repo=work/'giada';teacher=work/'teacher';native=work/'native'
def run(cmd,**kw):subprocess.run(list(map(str,cmd)),check=True,**kw)
run([sys.executable,'-m','pip','install','--quiet','uv==0.9.5'])
run([sys.executable,'-m','uv','venv','--python','3.12.11',native])
python=native/'bin/python'
run([sys.executable,'-m','uv','pip','install','--python',python,'numpy==1.26.4','neuron==8.2.7'])
run(['git','clone','--quiet','--depth','1','--branch','codex/surrogate-validity-audit','https://github.com/Zagred47/giada.git',repo])
run(['git','-C',repo,'fetch','--quiet','--depth','1','origin',%r]);run(['git','-C',repo,'checkout','--quiet','--detach','FETCH_HEAD'])
run(['git','clone','--quiet','--depth','1','https://github.com/SelfishGene/neuron_as_deep_net.git',teacher])
run(['git','-C',teacher,'fetch','--quiet','--depth','1','origin','074c4666300a8ad246601dab179a97a6942f0f29']);run(['git','-C',teacher,'checkout','--quiet','--detach','FETCH_HEAD'])
run([python,'-c',%r],cwd=repo,env=dict(os.environ,PATH=str(python.parent)+os.pathsep+os.environ['PATH']))
"""%(revision,probe)
        write_json(folder/'notebook.ipynb',dict(nbformat=4,nbformat_minor=5,metadata=dict(kernelspec=dict(name='python3',display_name='Python3',language='python')),cells=[dict(cell_type='code',id='native46env',execution_count=None,metadata={},outputs=[],source=code.splitlines(keepends=True))]))
        meta=launch.metadata(False,'alessandrobelli/giada-task46-native-environment','GIADA Task46 native environment preflight');meta.update(enable_internet=True,dataset_sources=[]);write_json(folder/'kernel-metadata.json',meta)
        write_json(F/'native_environment_pending.json',dict(CPU_only=True,revision=revision));r=api.kernels_push(str(folder),timeout='1800');assert not r.error,r.error
        write_json(marker,dict(ref=r.ref,version=r.version_number,revision=revision));(F/'native_environment_pending.json').unlink()
    receipt=json.loads(marker.read_text());ref=receipt['ref'].lstrip('/').removeprefix('code/')+'/'+str(receipt['version'])
    for _ in range(100):
        status=str(api.kernels_status(ref).status).lower();print('[46nativeenv] '+status,flush=True)
        if 'complete' in status:break
        if any(q in status for q in ('error','fail','cancel')):raise RuntimeError('Clean native preflight failed; do not launchGPU')
        time.sleep(15)
    else:raise RuntimeError('CPU native preflight deadline, inspect existing run')
    dest=ROOT/'artifacts/task46_native_env_readback';dest.mkdir(exist_ok=True);api.kernels_output(ref,str(dest),file_pattern=r'native_environment46\.json$',force=True,quiet=True)
    proof=json.loads((dest/'native_environment46.json').read_text());assert proof['valid'];write_json(F/'native_environment_verified.json',proof)
if __name__=='__main__':main()
