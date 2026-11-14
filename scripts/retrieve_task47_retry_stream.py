"""Bounded-memory output download; never buffer the multi-GB ZIP in RAM."""
import hashlib,io,json,os,sys,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
import task47_retry_kaggle as launch
from research_memory.mirror import write_json

class RemoteZip(io.RawIOBase):
    def __init__(self,url,size,session):self.url=url;self.size=size;self.position=0;self.session=session
    def seekable(self):return True
    def readable(self):return True
    def tell(self):return self.position
    def seek(self,offset,whence=0):
        self.position=offset if whence==0 else self.position+offset if whence==1 else self.size+offset
        return self.position
    def read(self,size=-1):
        count=min(self.size-self.position,size if size>=0 else self.size)
        if count<=0:return b''
        start=self.position;end=start+count-1
        with self.session.get(self.url,headers={'Range':f'bytes={start}-{end}'},stream=True,timeout=(15,40)) as r:
            if r.status_code!=206:raise RuntimeError('Range read unsupported; full streaming copy remains required')
            value=r.content
            assert len(value)==count and r.headers['Content-Range'].startswith(f'bytes {start}-{end}/')
        self.position+=count;return value

def main():
    import requests
    from kaggle.api.kaggle_api_extended import KaggleApi,ApiListKernelSessionOutputRequest
    folder=launch.k.DEST;folder.mkdir(exist_ok=True);api=KaggleApi();api.authenticate()
    receipt=json.loads((folder/'kaggle_receipt.json').read_text());job=json.loads((launch.F/'kaggle_launch.json').read_text())
    assert receipt['revision']==job['revision'];name=receipt['archive'];assert Path(name).name==name
    req=ApiListKernelSessionOutputRequest();req.user_name='alessandrobelli';req.kernel_slug='giada-task47-protocol-retry'
    with api.build_kaggle_client() as client:response=client.kernels.kernels_api_client.list_kernel_session_output(req)
    item=next(q for q in response.files if q.file_name==name);session=requests.Session()
    try:
        with zipfile.ZipFile(RemoteZip(item.url,receipt['bytes'],session)) as z:
            report=json.loads(z.read('final_report.json'))
            identities={phase:json.loads(z.read('executed_protocol_'+phase+'.json')) for phase in ('driver','preflight','native','gpu')}
            assert all(q==identities['driver'] for q in identities.values()) and identities['driver']['task']=='47'
            assert [g['nodes'] for g in identities['driver']['geometries']]==[5,9]
            write_json(folder/'remote_report.json',dict(provisional_until_archive_SHA=True,report=report,identities=identities))
            print(json.dumps(dict(remote_report=report,executed_protocol=identities['driver'],archive_SHA_pending=True)),flush=True)
    except RuntimeError as error:print(str(error),flush=True)
    archive=folder/name;partial=archive.with_suffix('.zip.part')
    if not archive.exists() or archive.stat().st_size!=receipt['bytes']:
        count=partial.stat().st_size if partial.exists() else 0;reported=count//(256*1024*1024)
        with partial.open('ab') as f:
            while count<receipt['bytes']:
                end=min(count+32*1024*1024,receipt['bytes'])-1
                with session.get(item.url,headers={'Range':f'bytes={count}-{end}'},stream=True,timeout=(15,40)) as r:
                    r.raise_for_status();assert r.status_code==206 and r.headers['Content-Range'].startswith(f'bytes {count}-{end}/')
                    for block in r.iter_content(4*1024*1024):
                        if not block:continue
                        f.write(block);count+=len(block)
                assert count==end+1
                if count//(256*1024*1024)>reported:
                    reported=count//(256*1024*1024);print(f'[47 backup] {count}/{receipt["bytes"]} bytes',flush=True)
        assert partial.stat().st_size==receipt['bytes'];partial.replace(archive)
    with archive.open('rb') as f:digest=hashlib.file_digest(f,'sha256').hexdigest()
    assert digest==receipt['sha256']
    print('[47 backup] SHA verified; checking ZIP CRC',flush=True)
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None;report=json.loads(z.read('final_report.json'))
        assert report.get('executed_protocol_verified') and report['task']=='47'
    write_json(folder/'external_backup_receipt.json',dict(valid=True,runtime_independent=True,local=str(archive),remote=dict(label='final',sha256=digest,bytes=archive.stat().st_size,platform='Kaggle persisted output'),launch=job,technical_failure=not(receipt['valid'] and report.get('complete')),report=report))
    print(json.dumps(dict(backup_verified=True,decision=report.get('decision'),quality_passed=report.get('independent_quality_confirmation_passed'))),flush=True)
if __name__=='__main__':main()
