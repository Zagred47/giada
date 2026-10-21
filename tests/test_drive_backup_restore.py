import json
import tempfile
import unittest
from pathlib import Path
from research_memory.packed_database import restore_if_missing

class DriveBackupRestore(unittest.TestCase):
    def test_local_snapshot_never_silently_uses_older_drive_copy(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as folder:
            db=Path(folder)/'research_memory.sqlite'
            db.with_suffix('.sqlite.manifest.json').write_text(json.dumps(dict(format='sqlite-local-backup-v1',local_archive='verified-local.7z',sqlite_sha256='b'*64)))
            with self.assertRaisesRegex(FileNotFoundError,'older Drive copy'):restore_if_missing(db)
            self.assertFalse(db.exists())
            db.write_bytes(b'existing authoritative local database')
            self.assertFalse(restore_if_missing(db))
            self.assertEqual(db.read_bytes(),b'existing authoritative local database')
    def test_missing_database_requires_explicit_download(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as folder:
            db=Path(folder)/'research_memory.sqlite'
            db.with_suffix('.sqlite.manifest.json').write_text(json.dumps(dict(format='sqlite-drive-backup-v1',drive_url='https://drive.google.com/file/d/example/view',sqlite_sha256='a'*64)))
            with self.assertRaisesRegex(FileNotFoundError,'Drive'):restore_if_missing(db)
            self.assertFalse(db.exists())
            db.write_bytes(b'original user database')
            self.assertFalse(restore_if_missing(db))
            self.assertEqual(db.read_bytes(),b'original user database')

if __name__=='__main__':unittest.main()
