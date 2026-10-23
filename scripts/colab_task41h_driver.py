import json,os,runpy,time
from pathlib import Path
os.environ['GIADA_COLAB_REQUEST']='/content/giada_task41h_launch.json'
request=json.loads(Path(os.environ['GIADA_COLAB_REQUEST']).read_text())
try:
    runpy.run_path('/content/colab_task39e_driver.py',run_name='__main__')
except SystemExit as exc:
    if exc.code not in (None,0):raise
if request.get('evaluation_only'):
    root=Path('/content')/('giada_task41h_'+request['revision'][:12])
    expected=json.loads((root/'recovery_manifest.json').read_text())['sha256']
    deadline=time.monotonic()+7200
    while True:
        ack=root/'external_final_ack.json'
        if ack.exists():
            receipt=json.loads(ack.read_text())
            assert receipt['valid'] and receipt['remote']['sha256']==expected
            print('[GIADA recovery] final external copy acknowledged',flush=True);break
        assert time.monotonic()<deadline,'Final external acknowledgement timeout'
        print('[GIADA recovery] keeping runtime active until final external backup is verified',flush=True)
        time.sleep(30)
