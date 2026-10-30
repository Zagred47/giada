import hashlib,json,sys,tempfile,unittest,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from task41b_local_supervisor import recover_parts

class RecoveryPartsTests(unittest.TestCase):
    def test_archive_retained_parts_removed_and_restart_reuses(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'artifacts') as d:
            root=Path(d);source=root/'source.zip'
            with zipfile.ZipFile(source,'w') as z:z.writestr('checkpoint.pt',b'immutable-checkpoint')
            raw=source.read_bytes();parts=[]
            for n,block in enumerate((raw[:40],raw[40:])):
                p=root/f'recovery.zip.part{n:04d}';p.write_bytes(block)
                parts.append(dict(remote='/remote/'+p.name,bytes=len(block),sha256=hashlib.sha256(block).hexdigest()))
            receipt=dict(archive='/remote/recovery.zip',sha256=hashlib.sha256(raw).hexdigest(),parts=parts)
            target=root/'recovery.zip'
            def no_download(*a):raise AssertionError('Unexpected network access')
            self.assertTrue(recover_parts(receipt,target,root,no_download));self.assertEqual(target.read_bytes(),raw)
            self.assertTrue(all(not (root/Path(p['remote']).name).exists() for p in parts))
            self.assertTrue(recover_parts(receipt,target,root,no_download))
    def test_invalid_zip_does_not_remove_parts(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'artifacts') as d:
            root=Path(d);raw=b'not a ZIP';part=root/'recovery.zip.part0000';part.write_bytes(raw)
            receipt=dict(archive='recovery.zip',sha256=hashlib.sha256(raw).hexdigest(),parts=[dict(remote=part.name,bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())])
            with self.assertRaises(zipfile.BadZipFile):recover_parts(receipt,root/'recovery.zip',root,lambda *a:False)
            self.assertTrue(part.exists());self.assertFalse((root/'recovery.zip').exists())
if __name__=='__main__':unittest.main()
