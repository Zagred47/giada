import unittest

from src.hayflow_data import (
    CAUSAL_OBSERVATION_PHASE,
    CausalReleaseOutcome,
    InputAction,
    ProtocolTrajectory,
    TargetedRecipe,
    build_balanced_episode_plan,
    append_specialized_test_episodes,
    build_input_views,
    select_adaptive_recipe_brackets,
    summarize_independent_support,
)
from src.hayflow_teacher.event_extractor import (
    EventDefinition,
    annotate_backpropagation,
    extract_events,
)
from src.hayflow_teacher import TargetedDiagnosticDatasetSession


def outcome(success=True):
    increment = 2.0 if success else 0.0
    return CausalReleaseOutcome(
        transition_id=3,
        event_index=0,
        synapse_id=7,
        scheduled_time_ms=100.25,
        offset_ms=0.25,
        synapse_type="ProbAMPANMDA2",
        functional_type="excitatory_AMPA+NMDA",
        weight=1.0,
        random123_seed=11,
        random123_stream_id=7,
        random123_global_index=0,
        rng_sequence_before=0.0,
        rng_sequence_after=1.0,
        rng_distribution="negexp(1)",
        rng_preview_value=0.2,
        release_probability=0.4,
        release_success=success,
        released_quantity=1.0 if success else 0.0,
        ampa_state_increment=increment,
        nmda_state_increment=increment,
        inhibitory_state_increment=0.0,
        pre_synapse_state={"point_process.A_AMPA": 0.0},
        post_synapse_state={"point_process.A_AMPA": increment},
        observation_phase=CAUSAL_OBSERVATION_PHASE,
    )


class TargetedReleaseContractTest(unittest.TestCase):
    def test_three_input_views_keep_realized_release_causal(self):
        actions = [
            {
                "kind": "synaptic_event",
                "offset_ms": 0.25,
                "synapse_id": 7,
                "weight_multiplier": 1.0,
                "metadata": {},
            }
        ]
        views = build_input_views(actions, [outcome(True)])
        self.assertNotIn("release_success", views["U_scheduled"][0])
        self.assertIn("random123_stream_id", views["U_rng"][0])
        self.assertNotIn("release_success", views["U_rng"][0])
        self.assertTrue(views["U_realized"][0]["release_success"])

        failed = build_input_views(actions, [outcome(False)])
        self.assertEqual(failed["U_realized"], [])

    def test_release_contract_rejects_future_or_inconsistent_labels(self):
        valid = outcome(True)
        valid.validate()
        invalid = CausalReleaseOutcome(
            **{
                **valid.__dict__,
                "source": "inferred_from_t_plus_1_future",
            }
        )
        with self.assertRaisesRegex(ValueError, "future"):
            invalid.validate()

    def test_nmda_spike_can_be_separated_from_plateau_by_duration(self):
        definition = EventDefinition(
            "nmda_spike",
            "tuft",
            1,
            "tuft",
            -40.0,
            -50.0,
            min_duration_ms=1.0,
            maximum_event_duration_ms=9.0,
        )
        time = list(range(13))
        short = [-70, -35, -35, -55] + [-70] * 9
        long = [-70, -35] + [-35] * 10 + [-55]
        self.assertEqual(len(extract_events(time, {"tuft": short}, [definition])), 1)
        self.assertEqual(len(extract_events(time, {"tuft": long}, [definition])), 0)

    def test_bap_annotation_requires_outward_temporal_order(self):
        time = [0.0, 0.5, 1.0, 1.5, 2.0]
        event = {
            "kind": "backpropagating_ap",
            "onset_ms": 1.0,
            "linked_delay_ms": 0.5,
        }
        traces = {
            "soma": [-70, 10, -30, -60, -70],
            "trunk": [-70, -50, 0, -40, -70],
            "nexus": [-70, -70, -50, -10, -70],
            "tuft": [-70, -70, -60, -40, -10],
            "basal": [-70, -50, -10, -50, -70],
        }
        rows = annotate_backpropagation(
            time,
            traces,
            [event],
            regional_distances_um={
                "soma": 0,
                "trunk": 400,
                "nexus": 700,
                "tuft": 900,
                "basal": 150,
            },
        )
        self.assertEqual(rows[0]["origin"], "soma")
        self.assertEqual(rows[0]["maximum_distance_um"], 900)


class TargetedProtocolPlannerTest(unittest.TestCase):
    def test_v1_1_plan_hash_binds_snapshot_and_branch_metadata(self):
        first = ProtocolTrajectory(
            trajectory_id="episode",
            category="dendritic_events",
            protocol="ca",
            seed=1,
            duration_ms=2,
            split="train",
            snapshot_source="train-snapshot-a",
            metadata={"snapshot_id": "train-snapshot-a", "branch_id": "a"},
        )
        second = ProtocolTrajectory(
            trajectory_id="episode",
            category="dendritic_events",
            protocol="ca",
            seed=1,
            duration_ms=2,
            split="train",
            snapshot_source="train-snapshot-b",
            metadata={"snapshot_id": "train-snapshot-b", "branch_id": "b"},
        )
        self.assertNotEqual(
            TargetedDiagnosticDatasetSession._protocol_plan_sha256([first]),
            TargetedDiagnosticDatasetSession._protocol_plan_sha256([second]),
        )

    def test_adaptive_brackets_require_seed_robust_positive_and_negative(self):
        trials = []
        for candidate, stimulus, kinds in (
            ("below", 0.9, []),
            ("above", 1.1, ["calcium_spike"]),
        ):
            for seed in (1, 2):
                trials.append(
                    {
                        "candidate_id": candidate,
                        "stimulus_scalar": stimulus,
                        "seed": seed,
                        "event_kinds": kinds,
                    }
                )
        report = select_adaptive_recipe_brackets(trials, required_seed_count=2)
        selection = report["selections"]["calcium_spike"]
        self.assertEqual(selection["positive_candidate_id"], "above")
        self.assertEqual(selection["negative_candidate_id"], "below")
        self.assertFalse(report["valid"])  # other required classes are absent

    def test_episode_planner_counts_independent_episodes_not_transitions(self):
        action = InputAction(
            "somatic_current", 0.05, duration_ms=0.9, amplitude_na=1.0
        )
        all_classes = (
            "axonal_spike",
            "somatic_spike",
            "backpropagating_ap",
            "calcium_spike",
            "nmda_spike",
            "nmda_plateau",
        )
        recipes = [
            TargetedRecipe(
                "positive",
                "dendritic",
                "positive",
                2,
                {0: (action,)},
                positive_for=all_classes,
            ),
            TargetedRecipe(
                "negative",
                "dendritic",
                "hard_negative",
                2,
                {0: (action,)},
                hard_negative_for=all_classes,
            ),
        ]
        plans, rows = build_balanced_episode_plan(
            recipes,
            positive_targets={"train": 2},
            hard_negative_targets={"train": 3},
        )
        self.assertEqual(len(plans), 5)
        support = summarize_independent_support(rows)
        self.assertEqual(
            support["calcium_spike"]["train"]["positive_episode_count"], 2
        )
        self.assertEqual(
            support["calcium_spike"]["train"][
                "hard_negative_episode_count"
            ],
            3,
        )

    def test_specialized_pairs_share_only_the_intended_snapshot(self):
        synaptic = InputAction(
            "synaptic_event", 0.25, synapse_id=4
        )
        classes = (
            "axonal_spike",
            "somatic_spike",
            "backpropagating_ap",
            "calcium_spike",
            "nmda_spike",
            "nmda_plateau",
        )
        positive = TargetedRecipe(
            "positive",
            "dendritic",
            "positive",
            4,
            {1: (synaptic,)},
            positive_for=classes,
            branch_id="train-branch",
        )
        negative = TargetedRecipe(
            "negative",
            "dendritic",
            "negative",
            4,
            {1: (synaptic,)},
            hard_negative_for=classes,
            branch_id="train-branch",
        )
        heldout = TargetedRecipe(
            "heldout",
            "dendritic",
            "heldout",
            4,
            {1: (synaptic,)},
            hard_negative_for=classes,
            branch_id="heldout-branch",
            metadata={"train_eligible": False},
        )
        recovery = TargetedRecipe(
            "recovery",
            "dendritic",
            "recovery",
            6,
            {1: (synaptic,), 3: (synaptic,)},
            positive_for=classes,
            branch_id="train-branch",
            recovery_probe_delay_ms=2.0,
            metadata={
                "train_eligible": False,
                "recovery_probe": True,
                "pilot_validated": True,
            },
        )
        plans, rows = build_balanced_episode_plan(
            [positive, negative, heldout, recovery],
            positive_targets={"train": 1, "validation": 1, "deterministic_test": 1},
            hard_negative_targets={"train": 1, "validation": 1, "deterministic_test": 1},
        )
        plans, rows = append_specialized_test_episodes(
            plans, rows, [positive, negative, heldout, recovery]
        )
        near = [row for row in plans if row.split == "branching_near_test"]
        self.assertEqual(len({row.seed for row in near}), 1)
        self.assertEqual(
            len({row.metadata["snapshot_id"] for row in near}), 1
        )
        release = [
            row for row in plans if row.split == "release_identifiability_test"
        ]
        self.assertEqual(len({row.seed for row in release}), 2)
        self.assertEqual(
            len({row.metadata["snapshot_id"] for row in release}), 1
        )
        train_branches = {
            row.metadata["branch_id"] for row in plans if row.split == "train"
        }
        heldout_branches = {
            row.metadata["branch_id"]
            for row in plans
            if row.split == "held_out_branch_test"
        }
        self.assertTrue(train_branches.isdisjoint(heldout_branches))


if __name__ == "__main__":
    unittest.main()
