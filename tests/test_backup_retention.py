import hashlib,json,tempfile,unittest
from pathlib import Path
from research_memory.backup_retention import rotate
TEMP=Path(__file__).resolve().parents[1]/'artifacts'

class BackupRetentionTests(unittest.TestCase):
    def make(self,root,n):
        p=root/f'2026-10-{n:02d}_00-00-00Z';p.mkdir();db=p/'backup.sqlite';db.write_bytes(b'SQLite format 3\x00'+bytes([n])*1024)
        (p/'manifest.json').write_text(json.dumps(dict(filename=db.name,bytes=db.stat().st_size,sha256=hashlib.sha256(db.read_bytes()).hexdigest(),integrity_check='ok',foreign_key_violations=[])))
        return p
    def test_keep_two_and_preserve_unknown(self):
        with tempfile.TemporaryDirectory(dir=TEMP) as d:
            root=Path(d);folders=[self.make(root,n) for n in (1,2,3)];other=root/'not-a-backup';other.mkdir()
            r=rotate(root,apply=True,progress=lambda *a,**k:None)
            self.assertFalse(folders[0].exists());self.assertTrue(all(p.exists() for p in folders[1:]));self.assertTrue(other.exists());self.assertGreater(r['bytes_removed'],0)
    def test_corrupt_retained_never_deletes(self):
        with tempfile.TemporaryDirectory(dir=TEMP) as d:
            root=Path(d);folders=[self.make(root,n) for n in (1,2,3)];(folders[2]/'backup.sqlite').write_bytes(b'bad')
            with self.assertRaises(AssertionError):rotate(root,apply=True,progress=lambda *a,**k:None)
            self.assertTrue(all(p.exists() for p in folders))
    def test_preview_only(self):
        with tempfile.TemporaryDirectory(dir=TEMP) as d:
            root=Path(d);folders=[self.make(root,n) for n in (1,2,3)]
            r=rotate(root,progress=lambda *a,**k:None);self.assertEqual(r['bytes_removed'],0);self.assertTrue(all(p.exists() for p in folders))
if __name__=='__main__':unittest.main()
