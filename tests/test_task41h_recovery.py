import hashlib,tempfile,unittest
from pathlib import Path
from scripts.task41b_local_supervisor import recover_parts

class RecoveryTests(unittest.TestCase):
    def test_parts_resume_and_reject_corruption(self):
        blocks=[b'first'*100,b'second'*100];sha=lambda b:hashlib.sha256(b).hexdigest()
        receipt=dict(archive='/remote/recovery.zip',sha256=sha(b''.join(blocks)),parts=[dict(remote=f'/remote/part{i}',bytes=len(b),sha256=sha(b)) for i,b in enumerate(blocks)])
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]/'artifacts') as name:
            root=Path(name);calls=[]
            def download(name,target):
                calls.append(name);target.write_bytes(blocks[int(name[-1])]);return True
            self.assertTrue(recover_parts(receipt,root/'result.zip',root,download))
            self.assertEqual((root/'result.zip').read_bytes(),b''.join(blocks))
            calls.clear();self.assertTrue(recover_parts(receipt,root/'result.zip',root,download));self.assertEqual(calls,[])
            (root/'part0').write_bytes(b'bad')
            def corrupt(name,target):target.write_bytes(b'bad');return True
            self.assertFalse(recover_parts(receipt,root/'bad.zip',root,corrupt));self.assertFalse((root/'bad.zip').exists())

if __name__=='__main__':unittest.main()
