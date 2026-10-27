import unittest
import weakref
from src.giada_teacher.task41k_recovery import managed_scanner_class


class Original:
    def __init__(self):
        self.count = 0
        def step(delta=1):
            self.count += delta
            return self.count
        self.step = step


class LifecycleTests(unittest.TestCase):
    def test_same_bytecode_reusable_and_immediate_release(self):
        old = Original()
        cls = managed_scanner_class(Original)
        obj = cls()
        self.assertIs(obj.step.__code__, old.step.__code__)
        self.assertEqual(obj.step(2), 2)
        self.assertEqual(obj.step(), 3)
        reference = weakref.ref(obj)
        del obj
        self.assertIsNone(reference())


if __name__ == '__main__':
    unittest.main()
