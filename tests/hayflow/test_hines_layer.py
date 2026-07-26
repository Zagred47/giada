import importlib.util

import pytest


torch_available = importlib.util.find_spec("torch") is not None


@pytest.mark.skipif(not torch_available, reason="PyTorch is not installed locally")
def test_hines_matches_dense_solve_and_reports_safe_pivots():
    import torch

    from src.hayflow_model.hines_layer import DifferentiableHinesSolve

    #       0
    #     /   \
    #    1     2
    #   / \     \
    #  3   4     5
    parents = [0, 0, 0, 1, 1, 2]
    solver = DifferentiableHinesSolve(parents).double()
    coupling = torch.tensor(
        [[0.0, 0.7, 0.4, 0.3, 0.2, 0.5], [0.0, 0.2, 0.6, 0.1, 0.4, 0.3]],
        dtype=torch.double,
    )
    # Strict diagonal dominance guarantees a positive definite tree system.
    diagonal = torch.tensor(
        [[2.2, 2.1, 1.8, 1.2, 1.1, 1.5], [2.0, 1.7, 2.3, 1.0, 1.4, 1.2]],
        dtype=torch.double,
    )
    rhs = torch.tensor(
        [[1.0, -0.2, 0.5, 0.9, -0.7, 0.1], [-0.4, 0.3, 1.2, -0.1, 0.2, 0.8]],
        dtype=torch.double,
    )
    actual, report = solver(diagonal, coupling, rhs, return_diagnostics=True)
    expected = torch.linalg.solve(solver.dense_matrix(diagonal, coupling), rhs.unsqueeze(-1)).squeeze(-1)
    torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-10)
    assert report["positive_diagonal"]
    assert report["well_conditioned"]


@pytest.mark.skipif(not torch_available, reason="PyTorch is not installed locally")
def test_hines_gradcheck():
    import torch

    from src.hayflow_model.hines_layer import DifferentiableHinesSolve

    solver = DifferentiableHinesSolve([0, 0, 1, 1]).double()
    diagonal = torch.tensor(
        [[2.5, 2.0, 1.4, 1.2]], dtype=torch.double, requires_grad=True
    )
    coupling = torch.tensor(
        [[0.0, 0.5, 0.2, 0.3]], dtype=torch.double, requires_grad=True
    )
    rhs = torch.tensor(
        [[1.0, 0.2, -0.3, 0.7]], dtype=torch.double, requires_grad=True
    )
    assert torch.autograd.gradcheck(
        lambda d, g, b: solver(d, g, b),
        (diagonal, coupling, rhs),
        eps=1e-6,
        atol=1e-5,
        rtol=1e-4,
    )


@pytest.mark.skipif(not torch_available, reason="PyTorch is not installed locally")
def test_hines_rejects_non_positive_diagonal():
    import torch

    from src.hayflow_model.hines_layer import DifferentiableHinesSolve

    solver = DifferentiableHinesSolve([0, 0])
    with pytest.raises(RuntimeError, match="diagonal"):
        solver(
            torch.tensor([[1.0, 0.0]]),
            torch.tensor([[0.0, 0.1]]),
            torch.ones(1, 2),
        )


def test_tree_depths_rejects_cycles():
    from src.hayflow_model.hines_layer import tree_depths

    with pytest.raises(ValueError, match="exactly one"):
        tree_depths([1, 0])
