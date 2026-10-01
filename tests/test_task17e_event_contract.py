from src.giada_teacher.roadmap_task17e_independent_causal_confirmation import (
    Task17eConfig, _event_comparison, _protocols,
)


def _event(kind="somatic_spike", onset=4.0, censored=False):
    return {"kind": kind, "onset_ms": onset, "right_censored": censored}


def test_event_comparison_rejects_missing_or_late_events():
    config = Task17eConfig()
    reference = {"events": [_event()]}
    assert _event_comparison(reference, {"events": [_event(onset=4.2)]}, config)["somatic_spike"]["passed"]
    assert not _event_comparison(reference, {"events": [_event(onset=4.225)]}, config)["somatic_spike"]["passed"]
    assert not _event_comparison(reference, {"events": []}, config)["somatic_spike"]["passed"]
    assert not _event_comparison(reference, {"events": [_event(censored=True)]}, config)["somatic_spike"]["passed"]


def test_protocol_variants_are_distinct_schedules():
    quiet, nmda, calcium, nmda_variant, calcium_variant = _protocols(Task17eConfig())
    assert quiet is None
    assert nmda.synapse_count != nmda_variant.synapse_count
    assert calcium.synapse_count != calcium_variant.synapse_count
    assert nmda.event_window_ms != nmda_variant.event_window_ms
    assert calcium.event_window_ms != calcium_variant.event_window_ms
