"""Read official live session logs; disconnecting never cancels the batch."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'artifacts/kaggle_cli_deps'))
from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.kernels.types.kernels_api_service import ApiGetKernelSessionLogsStreamRequest
api=KaggleApi();api.authenticate()
request=ApiGetKernelSessionLogsStreamRequest();request.user_name='alessandrobelli';request.kernel_slug='giada-task41h-evaluation-recovery';request.wait_for_logs_url_seconds=5
target=ROOT/'artifacts/task41h_kaggle_live.log'
with api.build_kaggle_client() as client:
    response=client.kernels.kernels_api_client.get_kernel_session_logs_stream(request)
    try:
        with target.open('a',encoding='utf-8') as stream:
            for line in response.iter_lines(chunk_size=1024,decode_unicode=True):
                if isinstance(line,bytes):line=line.decode('utf-8',errors='replace')
                if not line:continue
                stream.write(line+'\n');stream.flush()
                if 'END_OF_LOG' in line:break
    finally:response.close()
print(json.dumps(dict(log_saved=str(target),read_only=True)),flush=True)
