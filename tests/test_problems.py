"""Benchmark definitions and their initial data."""

import numpy as np
import pytest

from euler1d import PROBLEMS, get_problem
from euler1d.thermo import conservative_to_primitive


@pytest.mark.parametrize("name", sorted(PROBLEMS))
def test_initial_condition_is_admissible(name):
    problem = get_problem(name)
    x = np.linspace(problem.x_min, problem.x_max, 101)
    rho, _, p = conservative_to_primitive(problem.initial_condition(x), problem.gamma)
    assert np.all(rho > 0.0)
    assert np.all(p > 0.0)


@pytest.mark.parametrize("name", sorted(PROBLEMS))
def test_config_inherits_the_problem_geometry(name):
    problem = get_problem(name)
    config = problem.config(nx=32)
    assert config.nx == 32
    assert config.x_min == problem.x_min
    assert config.t_final == problem.t_final
    assert config.boundary == problem.boundary
    assert len(config.cell_centers) == 32


def test_riemann_initial_condition_splits_at_x0():
    problem = get_problem("sod")
    U = problem.initial_condition(np.array([0.25, 0.75]))
    rho, _, p = conservative_to_primitive(U)
    np.testing.assert_allclose(rho, [1.0, 0.125])
    np.testing.assert_allclose(p, [1.0, 0.1])


def test_smooth_wave_returns_to_its_initial_shape_after_one_period():
    problem = get_problem("smooth")
    x = np.linspace(0.0, 1.0, 51)
    rho_0, _, _ = problem.exact(x, 0.0)
    rho_1, _, _ = problem.exact(x, 1.0)  # u0 = 1 on a unit-length periodic domain
    np.testing.assert_allclose(rho_0, rho_1, atol=1e-12)


def test_unknown_problem_is_rejected():
    with pytest.raises(ValueError, match="unknown problem"):
        get_problem("blast2d")
