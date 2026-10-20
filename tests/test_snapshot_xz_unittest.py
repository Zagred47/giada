import tempfile,unittest
from pathlib import Path
from research_memory.packed_snapshot import pack_snapshot,restore_snapshot_if_missing

class SnapshotTransport(unittest.TestCase):
    def test_both_formats_and_preservation(self):
        for compression in ('gzip','xz'):
            with tempfile.TemporaryDirectory() as folder:
                path=Path(folder)/'snapshot.json';original=b'{"preserve":"all records"}\r\n'*100
                path.write_bytes(original);pack_snapshot(path,compression)
                path.unlink();self.assertTrue(restore_snapshot_if_missing(path));self.assertEqual(path.read_bytes(),original)
                path.write_bytes(b'local newer');self.assertFalse(restore_snapshot_if_missing(path));self.assertEqual(path.read_bytes(),b'local newer')
    def test_corruption_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'snapshot.json';path.write_bytes(b'{}');pack_snapshot(path,'xz');path.unlink()
            path.with_suffix('.json.xz').write_bytes(b'corrupt')
            with self.assertRaises(ValueError):restore_snapshot_if_missing(path)

if __name__=='__main__':unittest.main()
