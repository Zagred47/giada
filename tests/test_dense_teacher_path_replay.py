import unittest

from src.giada_teacher.dense_teacher_path_replay import select_replay_rows


class DenseReplaySelectionTests(unittest.TestCase):
    def test_fixed_support_and_order(self):
        rows = []
        for site in (0, 387, 460, 469):
            for regime, ids in (("spike", (9, 3, 7)), ("quiet", (8, 2))):
                rows.extend({"segment_id": site, "regime": regime,
                             "transition_index": site * 100 + i,
                             "trajectory_id": f"t-{site}-{i}"} for i in ids)
        selected = select_replay_rows(list(reversed(rows)))
        self.assertEqual(len(selected), 12)
        self.assertEqual([r["transition_index"] for r in selected[:3]], [3, 7, 2])

    def test_missing_support_fails_closed(self):
        with self.assertRaisesRegex(RuntimeError, "lacks preregistered"):
            select_replay_rows([])


if __name__ == "__main__":
    unittest.main()
