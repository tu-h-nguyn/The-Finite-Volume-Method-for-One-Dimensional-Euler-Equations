"""End-to-end behaviour of the finite-volume driver."""

import numpy as np
import pytest

from euler1d import PRESETS, SolverConfig, get_problem, preset, primitive_to_conservative, solve
from euler1d.solver import _apply_boundary

ALL_SCHEMES = sorted(PRESETS)


def _uniform(nx, rho=1.3, u=0.7, p=2.1):
    return primitive_to_conservative(
        np.full(nx, rho), np.full(nx, u), np.full(nx, p)
    )


@pytest.mark.parametrize("scheme", ALL_SCHEMES)
def test_uniform_flow_is_preserved_exactly(scheme):
    """Free-stream preservation: a constant state must stay constant."""
    config = preset(scheme, nx=40, t_final=0.3)
    solution = solve(config, _uniform(config.nx))
    np.testing.assert_allclose(solution.density, 1.3, rtol=1e-13)
    np.testing.assert_allclose(solution.velocity, 0.7, rtol=1e-12)
    np.testing.assert_allclose(solution.pressure, 2.1, rtol=1e-12)


@pytest.mark.parametrize("scheme", ALL_SCHEMES)
def test_periodic_runs_conserve_mass_momentum_and_energy(scheme):
    """On a periodic domain the scheme is exactly conservative."""
    problem = get_problem("smooth")
    config = preset(scheme, **{"nx": 64, "t_final": 0.2, "boundary": "periodic"})
    initial = problem.initial_condition(config.cell_centers)
    solution = solve(config, initial)
    np.testing.assert_allclose(
        solution.conserved_totals(), initial.sum(axis=1) * config.dx, rtol=1e-12
    )


@pytest.mark.parametrize("scheme", ALL_SCHEMES)
@pytest.mark.parametrize("name", ["sod", "lax", "toro2", "toro3", "toro4"])
def test_every_scheme_stays_positive_on_every_benchmark(scheme, name):
    """Robustness sweep: no negative density or pressure anywhere."""
    problem = get_problem(name)
    config = preset(
        scheme,
        nx=100,
        cfl=0.4,
        x_min=problem.x_min,
        x_max=problem.x_max,
        t_final=problem.t_final,
        gamma=problem.gamma,
        boundary=problem.boundary,
    )
    solution = solve(config, problem.initial_condition)
    assert np.all(solution.density > 0.0)
    assert np.all(solution.pressure > 0.0)


def test_sod_wave_positions_match_the_exact_solution():
    """The computed shock must sit within a few cells of the analytical one."""
    problem = get_problem("sod")
    config = preset("hllc_muscl", nx=400, t_final=problem.t_final)
    solution = solve(config, problem.initial_condition)

    from euler1d import solve_star_state

    star = solve_star_state(problem.left, problem.right)
    shock_exact = problem.x0 + star.speeds["right_shock"] * problem.t_final
    # Locate the steepest density drop on the right half of the domain.
    right = solution.x > 0.6
    index = int(np.argmin(np.diff(solution.density[right])))
    shock_numeric = solution.x[right][index]
    assert abs(shock_numeric - shock_exact) < 3 * config.dx


def test_accuracy_ranking_of_the_report_schemes():
    """The report's central claim: each refinement reduces the L1 error."""
    from euler1d import l1_error

    problem = get_problem("sod")
    errors = {}
    for scheme in ["lax_friedrichs", "local_lax_friedrichs", "second_order_combined"]:
        config = preset(scheme, nx=200, t_final=problem.t_final)
        solution = solve(config, problem.initial_condition)
        rho_exact, _, _ = problem.exact(solution.x, solution.t)
        errors[scheme] = l1_error(solution.density, rho_exact, config.dx)

    assert errors["lax_friedrichs"] > errors["local_lax_friedrichs"]
    assert errors["local_lax_friedrichs"] > errors["second_order_combined"]


def test_transmissive_boundary_copies_the_edge_cells():
    U = np.arange(3 * 5, dtype=float).reshape(3, 5)
    extended = _apply_boundary(U, "transmissive")
    assert extended.shape == (3, 9)
    np.testing.assert_array_equal(extended[:, 0], U[:, 0])
    np.testing.assert_array_equal(extended[:, 1], U[:, 0])
    np.testing.assert_array_equal(extended[:, -1], U[:, -1])


def test_periodic_boundary_wraps_around():
    U = np.arange(3 * 5, dtype=float).reshape(3, 5)
    extended = _apply_boundary(U, "periodic")
    np.testing.assert_array_equal(extended[:, :2], U[:, -2:])
    np.testing.assert_array_equal(extended[:, -2:], U[:, :2])


def test_solution_helpers_are_consistent():
    problem = get_problem("sod")
    config = preset("local_lax_friedrichs", nx=50, t_final=0.1)
    solution = solve(config, problem.initial_condition, record_totals=True)
    rho, u, p = solution.primitives
    np.testing.assert_allclose(solution.density, rho)
    np.testing.assert_allclose(
        solution.specific_internal_energy, p / ((config.gamma - 1.0) * rho)
    )
    assert len(solution.history) == solution.steps
    assert solution.mach_number.min() >= 0.0


def test_invalid_configuration_is_rejected():
    problem = get_problem("sod")
    with pytest.raises(ValueError, match="shape"):
        solve(SolverConfig(nx=10), problem.initial_condition(np.linspace(0, 1, 7)))
    with pytest.raises(ValueError, match="unknown time integrator"):
        solve(SolverConfig(nx=10, time_integrator="rk4"), problem.initial_condition)
    with pytest.raises(ValueError, match="unknown boundary"):
        solve(SolverConfig(nx=10, boundary="wall"), problem.initial_condition)
    with pytest.raises(ValueError, match="unknown scheme"):
        preset("godunov")


def test_step_limit_is_enforced():
    problem = get_problem("sod")
    with pytest.raises(RuntimeError, match="step limit"):
        solve(preset("local_lax_friedrichs", nx=50, max_steps=3), problem.initial_condition)
