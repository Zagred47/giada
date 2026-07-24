import tempfile
import unittest
from pathlib import Path

try:
    import h5py
    import numpy as np
except ImportError:  # Lightweight contract-test environments omit HDF5.
    h5py = None
    np = None

from src.hayflow_data import TransitionH5Writer


@unittest.skipUnless(h5py is not None and np is not None, "h5py is required")
class TransitionH5WriterTest(unittest.TestCase):
    def test_extrema_times_allow_non_monotonic_segment_indices(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "transitions.h5"
            with TransitionH5Writer(
                path,
                {"voltage": 3, "rng_state": 1},
                microtrace_samples=3,
                microtrace_variable_count=1,
                segment_count=3,
                probe_count=1,
            ) as writer:
                writer.set_microtrace_grid([0.0, 0.5, 1.0])
                writer.append(
                    {
                        "state_t": {"voltage": [-70.0, -70.0, -70.0]},
                        "state_t_plus_1": {
                            "voltage": [-69.0, -69.0, -69.0]
                        },
                        "rng_t": [0.0],
                        "rng_t_plus_1": [0.0],
                        "micro_selected": [[0.0], [0.0], [0.0]],
                        "micro_probe_voltage": [[0.0], [0.0], [0.0]],
                        # argmin indices are [2, 0, 1], deliberately unsorted.
                        "micro_all_voltage": [
                            [0.0, -3.0, 0.0],
                            [-1.0, -2.0, -3.0],
                            [-2.0, -1.0, -2.0],
                        ],
                        "micro_somatic_current": [0.0, 0.0, 0.0],
                        "metadata": {
                            "transition_id": 0,
                            "trajectory_id": "test",
                            "category": "test",
                            "protocol": "test",
                            "split": "test",
                            "seed": 1,
                            "step_index": 0,
                            "start_time_ms": 0.0,
                            "native_snapshot_ref": "snapshot",
                        },
                        "inputs": [],
                    }
                )

            with h5py.File(path, "r") as handle:
                group = handle["microtraces/all_segment_voltage_summary"]
                np.testing.assert_allclose(
                    group["minimum_time_offset_ms"][0], [1.0, 0.0, 0.5]
                )
                np.testing.assert_allclose(
                    group["maximum_time_offset_ms"][0], [0.0, 1.0, 0.0]
                )

    def test_optional_release_contract_is_stored_without_changing_legacy_key(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "release.h5"
            with TransitionH5Writer(
                path,
                {"voltage": 1, "rng_state": 1},
                microtrace_samples=2,
                microtrace_variable_count=1,
                segment_count=1,
                probe_count=1,
                input_view_names=("U_scheduled", "U_rng", "U_realized"),
                store_release_outcomes=True,
            ) as writer:
                writer.set_microtrace_grid([0.0, 1.0])
                writer.append(
                    {
                        "state_t": {"voltage": [-70.0]},
                        "state_t_plus_1": {"voltage": [-69.0]},
                        "rng_t": [0.0],
                        "rng_t_plus_1": [1.0],
                        "micro_selected": [[0.0], [0.0]],
                        "micro_probe_voltage": [[-70.0], [-69.0]],
                        "micro_all_voltage": [[-70.0], [-69.0]],
                        "micro_somatic_current": [0.0, 0.0],
                        "metadata": {
                            "transition_id": 0,
                            "trajectory_id": "release",
                            "category": "local_synaptic",
                            "protocol": "release",
                            "split": "release_identifiability_test",
                            "seed": 1,
                            "step_index": 0,
                            "start_time_ms": 0.0,
                            "native_snapshot_ref": "snapshot",
                        },
                        "inputs": [{"kind": "synaptic_event"}],
                        "input_views": {
                            "U_scheduled": [{"kind": "synaptic_event"}],
                            "U_rng": [{"random123_stream_id": 1}],
                            "U_realized": [{"release_success": True}],
                        },
                        "release_outcomes": [{"release_success": True}],
                    }
                )
            with h5py.File(path, "r") as handle:
                self.assertIn("ordered_actions_json", handle["inputs"])
                self.assertIn("U_realized_json", handle["inputs"])
                self.assertIn("records_json", handle["release_outcomes"])


if __name__ == "__main__":
    unittest.main()
