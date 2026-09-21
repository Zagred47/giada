from copy import deepcopy

from src.giada_teacher import (
    build_gpu_baseline_contract,
    paired_index_generator,
    paired_index_stream,
    validate_gpu_baseline_contract,
)


def test_common_gpu_contract_is_valid_and_paired() -> None:
    contract = build_gpu_baseline_contract()
    assert contract["validation"]["valid"]
    assert len(contract["paired_randomness"]["model_seeds"]) == 3
    assert contract["paired_randomness"]["same_minibatch_index_stream_per_seed_across_arms"]
    assert contract["data_contract"]["sealed_test_access_during_training"] is False


def test_test_firewall_and_precision_are_fail_closed() -> None:
    contract = build_gpu_baseline_contract()
    contract["model_selection_firewall"]["sealed_test_can_select"] = ["checkpoint"]
    contract["numeric_policy"]["automatic_mixed_precision_primary"] = True
    validation = validate_gpu_baseline_contract(contract)
    assert not validation["valid"]
    assert len(validation["blockers"]) == 2


def test_budget_and_timing_modes_cannot_silently_drift() -> None:
    contract = build_gpu_baseline_contract()
    drifted = deepcopy(contract)
    drifted["optimization"]["maximum_steps"] = 9000
    drifted["latency_benchmark"]["modes_reported_separately"] = False
    validation = validate_gpu_baseline_contract(drifted)
    assert not validation["valid"]
    assert any("final checkpoint" in item for item in validation["blockers"])
    assert any("latency" in item for item in validation["blockers"])


def test_paired_minibatch_stream_is_replayable_and_seeded() -> None:
    first = paired_index_stream(8096, 1024, 20, 100017)
    replay = paired_index_stream(8096, 1024, 20, 100017)
    different = paired_index_stream(8096, 1024, 20, 100029)
    assert first == replay
    assert first["sha256"] != different["sha256"]
    assert len(first["batches"]) == 20
    assert all(len(batch) == 1024 for batch in first["batches"])


def test_compact_generator_replays_the_materialized_stream_exactly() -> None:
    materialized = paired_index_stream(101, 32, 20, 100017)["batches"]
    compact = paired_index_generator(101, 32, 100017)
    assert [next(compact) for _ in range(20)] == materialized
