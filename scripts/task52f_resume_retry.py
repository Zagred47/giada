"""Single user-authorized retry after explicit cancellation of queued version1."""
import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import task52f_resume_kaggle as r
from research_memory.mirror import write_json
old=r.F;new=r.ROOT/'experiments/results/task52f_resume_retry_v1';new.mkdir(exist_ok=True)
if sys.argv[1]=='launch':
    proof=json.loads((old/'cancelled_before_retry.json').read_text());assert proof['ui_confirmed']=='Cancelled after 0 seconds'
    for name in ('package.json','input_verified.json','recovery_input_verified.json','preregistration_receipt.json'):
        if not(new/name).exists():write_json(new/name,json.loads((old/name).read_text()))
r.F=new;r.prior.F=new;r.k.FOLDER=new;r.k.DEST=r.ROOT/'artifacts/task52f_resume_retry_external';r.k.SLUG='alessandrobelli/giada52f-resume600'
if sys.argv[1]=='supervise':r.prior.supervise()
else:r.k.main()
