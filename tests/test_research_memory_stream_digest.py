import hashlib
import unittest
from research_memory.mirror import digest,dumps


class StreamingDigestTests(unittest.TestCase):
    def test_canonical_bytes_unchanged(self):
        values=[None,True,[],{},[1,1.,-0.,1e-30,1e30],
            {'unicode':'🧪 μ calcio è','escaped':'a\nb\\c\"d','nested':[{'z':1,'a':2}]},
            {'big':'x'*200000+'🧪','records':[{'i':i,'x':i/7} for i in range(1000)]}]
        for value in values:
            self.assertEqual(digest(value),hashlib.sha256(dumps(value).encode('utf-8')).hexdigest())
    def test_nonfinite_remains_rejected(self):
        for value in (float('nan'),float('inf'),float('-inf')):
            with self.assertRaises(ValueError):digest({'value':value})


if __name__=='__main__':unittest.main()
