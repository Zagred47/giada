import hashlib,io,json,sys,unittest,zipfile
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from research_memory.mirror import write_json
if __name__=='__main__':
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.discover(str(ROOT/'tests'),pattern='test_task52f_resume.py'));assert result.wasSuccessful()
    spec=json.loads((ROOT/'experiments/task52f_resume_v1.json').read_text());path=ROOT/'artifacts/task52f_kaggle_external/task52f_final.zip'
    with path.open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==spec['source_sha256']
    with zipfile.ZipFile(path) as z:
        assert z.testzip() is None;state=torch.load(io.BytesIO(z.read('resumable_training.pt')),map_location='cpu',weights_only=True);exact=torch.load(io.BytesIO(z.read('checkpoint_600.pt')),map_location='cpu',weights_only=True)
        assert state['update']==600 and all(torch.equal(v,exact[k]) for k,v in state['model'].items())
        assert all(int(x['step'])==600 for x in state['optimizer']['state'].values())
    names=['src/giada_teacher/task52f_encoder_credit.py','scripts/run_task52f_resume.py','scripts/task52f_resume_kaggle.py','scripts/task52f_resume_input.py','experiments/task52f_resume_v1.json','tests/test_task52f_resume.py']
    write_json(ROOT/'experiments/results/task52f_resume_v1/local_preflight.json',dict(valid=True,split_continuous_bitwise_equal=True,optimizer_step=600,source_sha256=spec['source_sha256'],files_sha256={n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in names}))
    sys.path.insert(0,str(ROOT/'scripts'));import task52f_resume_kaggle as k
    doc=k.notebook('a'*40)
    for cell in doc['cells']:
        if cell['cell_type']=='code':compile(''.join(cell['source']),'52f-resume','exec')
    assert 'run_task52f_resume.py' in json.dumps(doc) and 'str(resumes[0])' in json.dumps(doc)
    write_json(ROOT/'notebooks/giada_task52f_resume.ipynb',doc);print('Resume preflight passed',flush=True)
