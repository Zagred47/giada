import tempfile
import unittest
from pathlib import Path
from research_memory.mirror import Mirror,ROOT


class StreamingVerifyTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(dir=ROOT.parent/'artifacts')
        self.addCleanup(self.tmp.cleanup)
        self.mirror=Mirror(Path(self.tmp.name)/'memory.sqlite')
        self.mirror.initialize()

    def test_verification_does_not_materialize_full_export(self):
        def forbidden():raise AssertionError('Whole DB export must not be used by verify')
        self.mirror.export_snapshot=forbidden
        self.assertTrue(self.mirror.verify()['valid'])

    def test_unknown_preserved_table_rejected(self):
        with self.mirror.connect() as conn:
            conn.execute('INSERT INTO _remote_records VALUES (?,?,?,?)',
                         ('unknown','rec00000000000001',None,'{}'))
        with self.assertRaises(ValueError):self.mirror.verify()


if __name__=='__main__':unittest.main()
