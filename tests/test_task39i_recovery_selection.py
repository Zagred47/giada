import importlib.util
from pathlib import Path
import unittest
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('recovery',ROOT/'scripts/recover_task39i_evaluation.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
class SelectionTests(unittest.TestCase):
    def test_earliest_lexicographic_not_common_score(self):
        curves=[dict(step=100,panel=[dict(rank=[1,2.,.4]),dict(rank=[0,5.,.3])]),dict(step=0,panel=[dict(rank=[1,2.,.4]),dict(rank=[1,1.,.1])]),dict(step=300,panel=[dict(rank=[1,3.,.1]),dict(rank=None)])]
        ranks,steps=module.choose(curves,2)
        self.assertEqual(steps,[0,100]);self.assertEqual(ranks,[(1,2.,.4),(0,5.,.3)])
    def test_missing_member_fails_closed(self):
        with self.assertRaises(AssertionError):module.choose([dict(step=0,panel=[dict(rank=None)])],1)
if __name__=='__main__':unittest.main()
