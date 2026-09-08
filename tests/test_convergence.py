"""Order-of-accuracy verification on a smooth solution.

Discontinuous benchmarks cannot distinguish a first-order from a second-order
scheme (both converge at rate 1 in :math:`L^1` near a shock), so the design
order is measured on the smooth advected density wave instead.
"""

import numpy as np
import pytest

from euler1d import ConvergenceTable, get_problem, l1_error, observed_order, preset, solve

RESOLUTIONS = [40, 80, 160]


def _convergence(scheme: str, resolutions=None) -> ConvergenceTable:
    resolutions = resolutions or RESOLUTIONS
    problem = get_problem("smooth")
    errors = []
    for nx in resolutions:
        config = preset(
            scheme,
            nx=nx,
            cfl=0.4,
            x_min=problem.x_min,
            x_max=problem.x_max,
            t_final=problem.t_final,
            boundary="periodic",
        )
        solution = solve(config, problem.initial_condition)
        exact, _, _ = problem.exact(solution.x, solution.t)
        errors.append(l1_error(solution.density, exact, config.dx))
    return ConvergenceTable(list(resolutions), errors, label=scheme)


@pytest.mark.parametrize(
    "scheme, expected",
    [
        ("local_lax_friedrichs", 1.0),
        ("second_order_combined", 2.0),
        ("hllc_muscl", 2.0),
    ],
)
def test_observed_order_matches_design_order(scheme, expected):
    table = _convergence(scheme)
    assert table.asymptotic_order == pytest.approx(expected, abs=0.25)


def test_lax_friedrichs_reaches_first_order_only_slowly():
    """Its viscosity coefficient is dx/dt, so the leading error term is large.

    The scheme is first-order accurate, but on the grids used here the measured
    rate is still climbing towards 1; asserting a clean 1.00 would be wrong.
    """
    table = _convergence("lax_friedrichs", resolutions=[100, 200, 400, 800])
    orders = table.orders[1:]
    assert 0.7 < table.asymptotic_order < 1.05
    assert all(later > earlier for earlier, later in zip(orders, orders[1:]))


def test_errors_decrease_monotonically_under_refinement():
    table = _convergence("second_order_combined")
    assert all(np.diff(table.errors) < 0.0)


def test_second_order_beats_first_order_on_the_same_grid():
    first = _convergence("local_lax_friedrichs")
    second = _convergence("second_order_combined")
    assert all(s < f for s, f in zip(second.errors, first.errors))


def test_observed_order_formula():
    """Halving the error under grid doubling is exactly first order."""
    orders = observed_order([1.0, 0.5, 0.25], [10, 20, 40])
    assert np.isnan(orders[0])
    assert orders[1] == pytest.approx(1.0)
    assert orders[2] == pytest.approx(1.0)


def test_convergence_table_rendering():
    table = ConvergenceTable([10, 20], [1e-2, 5e-3], label="demo")
    markdown = table.to_markdown()
    assert "| 10 |" in markdown and "1.00" in markdown
    assert "L1 error" in str(table)


def test_error_norms_agree_on_a_known_deviation():
    """A constant error e on a unit interval gives L1 = L2 = Linf = e."""
    from euler1d import l2_error, linf_error

    numeric = np.full(50, 1.25)
    exact = np.full(50, 1.0)
    dx = 1.0 / 50
    assert l1_error(numeric, exact, dx) == pytest.approx(0.25)
    assert l2_error(numeric, exact, dx) == pytest.approx(0.25)
    assert linf_error(numeric, exact) == pytest.approx(0.25)


def test_observed_order_is_undefined_for_a_vanishing_error():
    """An exactly-zero error carries no information about the rate."""
    orders = observed_order([1e-3, 0.0, 1e-4], [10, 20, 40])
    assert np.isnan(orders[1])
    assert np.isnan(orders[2])
