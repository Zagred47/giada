import unittest

import numpy as np

from src.giada_teacher.voltage_path_stress import VoltagePathStressConfig
from src.giada_teacher.voltage_step_sequences import FAMILIES, make_paired_step_paths


class RoadmapTask7Tests(unittest.TestCase):
    def test_paired_endpoints_and_initial_gates(self):
        paths = make_paired_step_paths(VoltagePathStressConfig(), role="development")
        self.assertEqual(tuple(paths), FAMILIES)
        for values in paths.values():
            self.assertEqual(values.shape, (256, 10))
            np.testing.assert_array_equal(values[0::2, 0], values[1::2, 0])
            np.testing.assert_array_equal(values[0::2, 7:], values[1::2, 7:])
            self.assertTrue(np.any(values[0::2, 1:7] != values[1::2, 1:7]))
            self.assertTrue(np.isfinite(values).all())

    def test_dev_and_sealed_independent(self):
        config = VoltagePathStressConfig()
        dev = make_paired_step_paths(config, role="development")
        sealed = make_paired_step_paths(config, role="sealed")
        self.assertEqual({k: v.shape for k, v in sealed.items()},
                         {k: (512, 10) for k in FAMILIES})
        self.assertFalse({tuple(row) for values in dev.values() for row in values} &
                         {tuple(row) for values in sealed.values() for row in values})


if __name__ == "__main__":
    unittest.main()
