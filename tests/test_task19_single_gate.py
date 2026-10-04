import numpy as np
from src.giada_teacher.single_gate_transfer import CHANNELS, rates, update, data, state_grid, measurements, passes


def test_rates_singular_branch_and_analytic_im():
    for channel in CHANNELS:
        inf, tau = rates(channel, np.r_[np.linspace(-155, 95, 1001), [-154.90001, -154.9, -154.89999]])
        assert np.isfinite(tau).all() and np.all(tau > 0)
        assert np.all((inf >= 0) & (inf <= 1))
    assert rates('Ih', -154.9) == rates('Ih', -154.9 + .0001)
    inf, tau = rates('Im', -35)
    assert inf == .5
    assert abs(tau - 1 / .0066 / 2.3 ** 1.3) < 1e-12


def test_single_state_semigroup_and_extremal_bound():
    rng = np.random.default_rng(33)
    voltage = rng.uniform(-135, 75, 1000)
    for channel in CHANNELS:
        i, t = rates(channel, voltage)
        initial = rng.uniform(0, 1, len(voltage))
        assert np.max(abs(update(update(initial, i, t, .5), i, t, .5) - update(initial, i, t, 1))) < 1e-14
        pi, pt = np.clip(i + .02, 0, 1), t * 1.1
        err = abs(update(initial, pi, pt, 25) - update(initial, i, t, 25))
        bound = np.maximum(abs(update(0, pi, pt, 25)-update(0, i, t, 25)), abs(update(1, pi, pt, 25)-update(1, i, t, 25)))
        assert np.all(err <= bound + 1e-14)


def test_evaluation_grid_roles_disjoint_and_authentic_metrics():
    dev, fresh = state_grid(), state_grid(True)
    assert not set(dev[:, 0]) & set(fresh[:, 0])
    assert dev.shape == (2814, 3) and fresh.shape == (5642, 3)
    x = data(190401, 100)
    for channel in CHANNELS:
        i, t = rates(channel, x[:, 0])
        p = update(x[:, 1], i, t, x[:, 2])
        metrics = measurements(x, p, i, t, channel)
        assert metrics['current_rmse_ma_cm2'] == 0
        assert passes(metrics, dict(gate_rmse=.001, gate_max_error=.01, inf_rmse=.002, log_tau_rmse=.02, open_rmse=.002, occupancy_violations=0))


def test_vectorized_independent_adam_and_paired_initialization():
    import torch
    from src.giada_teacher.single_gate_transfer import model_factory, equivalent
    descriptors = [(c, o, s) for c in CHANNELS for o in ('transition_only', 'rate_supervised') for s in (17, 29, 43)]
    for width in (16, 32):
        model = model_factory(torch, width, descriptors)
        for value in model.parameters():
            assert torch.equal(value[0], value[3])
        x = torch.tensor(data(5, 32), dtype=torch.float32)[None].expand(len(descriptors), -1, -1)
        assert equivalent(torch, model, width, descriptors, x)['valid']
        assert sum(p.numel() for p in model.parameters()) // len(descriptors) == width*width+5*width+2
