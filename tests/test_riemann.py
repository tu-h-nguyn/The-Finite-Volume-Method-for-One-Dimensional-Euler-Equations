"""The exact Riemann solver, checked against tabulated and analytical results."""

import numpy as np
import pytest

from euler1d import GasState, exact_solution, solve_star_state
from euler1d.problems import PROBLEMS
from euler1d.thermo import GAMMA_AIR, sound_speed


def test_sod_star_state_matches_toro_table():
    """Reference values from Toro, *Riemann Solvers*, table 4.1."""
    star = solve_star_state(GasState(1.0, 0.0, 1.0), GasState(0.125, 0.0, 0.1))
    assert star.p == pytest.approx(0.30313, abs=1e-5)
    assert star.u == pytest.approx(0.92745, abs=1e-5)
    assert star.rho_left == pytest.approx(0.42632, abs=1e-5)
    assert star.rho_right == pytest.approx(0.26557, abs=1e-5)
    assert (star.left_wave, star.right_wave) == ("rarefaction", "shock")


@pytest.mark.parametrize("name", ["sod", "lax", "toro2", "toro3", "toro4"])
def test_star_state_is_a_root_of_the_pressure_function(name):
    """f_L(p*) + f_R(p*) + (u_R - u_L) must vanish, and u* agree on both sides."""
    from euler1d.riemann import _f_and_derivative

    problem = PROBLEMS[name]
    star = solve_star_state(problem.left, problem.right, problem.gamma)
    f_l, _ = _f_and_derivative(star.p, problem.left, problem.gamma)
    f_r, _ = _f_and_derivative(star.p, problem.right, problem.gamma)
    residual = f_l + f_r + (problem.right.u - problem.left.u)
    assert abs(residual) < 1e-10
    assert problem.left.u - f_l == pytest.approx(star.u, abs=1e-10)
    assert problem.right.u + f_r == pytest.approx(star.u, abs=1e-10)


def test_shock_satisfies_rankine_hugoniot():
    """Mass, momentum and energy jumps across Sod's right shock must balance."""
    problem = PROBLEMS["sod"]
    star = solve_star_state(problem.left, problem.right)
    s = star.speeds["right_shock"]
    gamma = GAMMA_AIR

    def state(rho, u, p):
        E = p / (gamma - 1.0) + 0.5 * rho * u**2
        U = np.array([rho, rho * u, E])
        F = np.array([rho * u, rho * u**2 + p, (E + p) * u])
        return U, F

    U_star, F_star = state(star.rho_right, star.u, star.p)
    U_r, F_r = state(problem.right.rho, problem.right.u, problem.right.p)
    np.testing.assert_allclose(F_r - F_star, s * (U_r - U_star), rtol=1e-9, atol=1e-12)


def test_rarefaction_is_isentropic():
    """Entropy p / rho^gamma is constant through the left fan of Sod's problem."""
    problem = PROBLEMS["sod"]
    x = np.linspace(0.27, 0.48, 60)
    rho, _, p = problem.exact(x, 0.2)
    entropy = p / rho**GAMMA_AIR
    np.testing.assert_allclose(entropy, entropy[0], rtol=1e-10)


def test_riemann_invariant_is_constant_through_the_fan():
    """u + 2a/(gamma-1) is constant across a left-moving rarefaction."""
    problem = PROBLEMS["sod"]
    x = np.linspace(0.27, 0.48, 60)
    rho, u, p = problem.exact(x, 0.2)
    invariant = u + 2.0 * sound_speed(rho, p) / (GAMMA_AIR - 1.0)
    np.testing.assert_allclose(invariant, invariant[0], rtol=1e-10)


def test_symmetric_data_give_a_symmetric_solution():
    """The 123 problem is mirror symmetric about the initial discontinuity."""
    problem = PROBLEMS["toro2"]
    star = solve_star_state(problem.left, problem.right)
    assert star.u == pytest.approx(0.0, abs=1e-12)
    assert star.rho_left == pytest.approx(star.rho_right, rel=1e-12)
    assert star.left_wave == star.right_wave == "rarefaction"

    x = np.linspace(0.0, 1.0, 201)
    rho, u, p = problem.exact(x, problem.t_final)
    np.testing.assert_allclose(rho, rho[::-1], rtol=1e-10)
    np.testing.assert_allclose(u, -u[::-1], atol=1e-10)
    np.testing.assert_allclose(p, p[::-1], rtol=1e-10)


@pytest.mark.parametrize("name", ["sod", "lax", "toro2", "toro3", "toro4"])
def test_exact_solution_stays_positive(name):
    problem = PROBLEMS[name]
    x = np.linspace(problem.x_min, problem.x_max, 400)
    rho, _, p = problem.exact(x, problem.t_final)
    assert np.all(rho > 0.0)
    assert np.all(p > 0.0)


def test_exact_solution_at_time_zero_is_the_initial_data():
    problem = PROBLEMS["sod"]
    x = np.array([0.25, 0.75])
    rho, u, p = problem.exact(x, 0.0)
    np.testing.assert_allclose(rho, [1.0, 0.125])
    np.testing.assert_allclose(p, [1.0, 0.1])


def test_uniform_state_has_no_waves():
    state = GasState(1.0, 0.3, 1.0)
    x = np.linspace(0.0, 1.0, 21)
    rho, u, p = exact_solution(x, 0.4, state, state)
    np.testing.assert_allclose(rho, 1.0, rtol=1e-12)
    np.testing.assert_allclose(u, 0.3, rtol=1e-12)
    np.testing.assert_allclose(p, 1.0, rtol=1e-12)


def test_vacuum_generation_is_reported():
    """Two strong rarefactions pulling apart must not silently return garbage."""
    left = GasState(1.0, -10.0, 0.4)
    right = GasState(1.0, 10.0, 0.4)
    with pytest.raises(ValueError, match="vacuum"):
        solve_star_state(left, right)
