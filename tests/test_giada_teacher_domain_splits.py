from src.giada_teacher import AtomicDomainSplitConfig, build_atomic_domain_splits


def test_atomic_domain_splits_are_disjoint_and_nonempty() -> None:
    report = build_atomic_domain_splits()
    validation = report["validation"]
    assert validation["valid"]
    assert validation["overlap_count"] == 0
    assert validation["fit_evaluation_overlap_count"] == 0
    assert validation["all_strata_nonempty"]
    assert report["embedded_confirmation"]["atomic_rows"] == 0


def test_train_excludes_windows_guards_singularity_and_edges() -> None:
    report = build_atomic_domain_splits()
    train = report["strata"]["train"]
    voltages = set(train["voltages_mv"])
    assert -100.0 not in voltages and 40.0 not in voltages
    assert not any(-28.0 <= voltage <= -26.0 for voltage in voltages)
    config = AtomicDomainSplitConfig()
    for low, high in (
        config.interpolation_voltage_windows_development
        + config.interpolation_voltage_windows_test
    ):
        assert not any(
            low - config.voltage_window_guard_mv <= voltage <= high + config.voltage_window_guard_mv
            for voltage in voltages
        )


def test_single_axis_ood_keeps_other_axes_in_support() -> None:
    report = build_atomic_domain_splits()
    voltage_ood = report["strata"]["ood_voltage_test"]
    dt_ood = report["strata"]["ood_dt_test"]
    train = report["strata"]["train"]
    assert voltage_ood["states"] == train["states"]
    assert voltage_ood["dt_ms"] == train["dt_ms"]
    assert dt_ood["states"] == train["states"]
    assert set(dt_ood["voltages_mv"]).issubset(set(train["voltages_mv"]))


def test_development_and_sealed_values_are_distinct() -> None:
    report = build_atomic_domain_splits()
    assert set(report["strata"]["interpolation_state_development"]["states"]).isdisjoint(
        report["strata"]["interpolation_state_test"]["states"]
    )
    assert set(report["strata"]["interpolation_dt_development"]["dt_ms"]).isdisjoint(
        report["strata"]["interpolation_dt_test"]["dt_ms"]
    )
