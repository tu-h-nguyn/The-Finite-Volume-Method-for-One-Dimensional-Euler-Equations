"""Properties of the numerical fluxes, limiters and reconstruction."""

import numpy as np
import pytest

from euler1d import primitive_to_conservative
from euler1d.schemes import (
    FLUXES,
    LIMITERS,
    minmod,
    reconstruct,
    superbee,
)
from euler1d.thermo import flux

FLUX_NAMES = ["lax_friedrichs", "rusanov", "hll", "hllc"]


@pytest.fixture
def uniform_states():
    U = primitive_to_conservative([1.2] * 5, [0.4] * 5, [0.9] * 5)
    return U, U


@pytest.mark.parametrize("name", FLUX_NAMES)
def test_flux_is_consistent(name, uniform_states):
    """F(U, U) = F(U): every numerical flux must reduce to the physical one."""
    UL, UR = uniform_states
    numerical = FLUXES[name](UL, UR, 1.4, 2.0)
    np.testing.assert_allclose(numerical, flux(UL), rtol=1e-12)


@pytest.mark.parametrize("name", ["hll", "hllc"])
def test_godunov_type_fluxes_are_supersonic_upwind(name):
    """When every wave moves right, an upwind flux must return the left flux."""
    UL = primitive_to_conservative([1.0], [5.0], [1.0])
    UR = primitive_to_conservative([0.9], [5.1], [0.95])
    numerical = FLUXES[name](UL, UR, 1.4, 100.0)
    np.testing.assert_allclose(numerical, flux(UL), rtol=1e-12)


@pytest.mark.parametrize("name", ["lax_friedrichs", "rusanov"])
def test_central_fluxes_add_dissipation_proportional_to_the_jump(name):
    """LF and Rusanov are centred: they keep a dissipation term even upwind."""
    UL = primitive_to_conservative([1.0], [5.0], [1.0])
    UR = primitive_to_conservative([0.9], [5.1], [0.95])
    numerical = FLUXES[name](UL, UR, 1.4, 100.0)
    central = 0.5 * (flux(UL) + flux(UR))
    dissipation = numerical - central
    jump = UR - UL
    # The correction is -alpha * (U_R - U_L) / 2 with alpha > 0, so it always
    # points against the jump, and it never vanishes for distinct states.
    assert float(np.sum(dissipation * jump)) < 0.0
    assert not np.allclose(numerical, flux(UL))


def test_lax_friedrichs_is_more_diffusive_than_rusanov():
    """Its viscosity is dx/dt, which under the CFL bound exceeds max(|u| + a)."""
    UL = primitive_to_conservative([1.0], [0.0], [1.0])
    UR = primitive_to_conservative([0.125], [0.0], [0.1])
    dx_over_dt = 10.0  # far above the local wave speed ~1.18
    f_lf = FLUXES["lax_friedrichs"](UL, UR, 1.4, dx_over_dt)
    f_rusanov = FLUXES["rusanov"](UL, UR, 1.4, dx_over_dt)
    jump_lf = np.abs(f_lf - 0.5 * (flux(UL) + flux(UR)))
    jump_rusanov = np.abs(f_rusanov - 0.5 * (flux(UL) + flux(UR)))
    assert np.all(jump_lf >= jump_rusanov)


def test_hllc_resolves_a_stationary_contact_exactly():
    """Equal pressure and velocity, jumping density: HLLC must not smear it."""
    UL = primitive_to_conservative([1.0], [0.0], [1.0])
    UR = primitive_to_conservative([0.125], [0.0], [1.0])
    f_hllc = FLUXES["hllc"](UL, UR, 1.4, 10.0)
    # An exactly resolved stationary contact carries no mass and no energy.
    assert f_hllc[0, 0] == pytest.approx(0.0, abs=1e-14)
    assert f_hllc[2, 0] == pytest.approx(0.0, abs=1e-14)
    assert f_hllc[1, 0] == pytest.approx(1.0, rel=1e-12)
    # Rusanov, by contrast, diffuses mass across the same interface.
    f_rusanov = FLUXES["rusanov"](UL, UR, 1.4, 10.0)
    assert abs(f_rusanov[0, 0]) > 0.1


@pytest.mark.parametrize("name", sorted(LIMITERS))
def test_limiters_are_tvd_bounded(name):
    """Every limiter returns a slope between 0 and the smaller neighbouring one."""
    limit = LIMITERS[name]
    rng = np.random.default_rng(7)
    a = rng.uniform(-2.0, 2.0, 500)
    b = rng.uniform(-2.0, 2.0, 500)
    sigma = limit(a, b)
    opposite_signs = a * b <= 0.0
    assert np.all(sigma[opposite_signs] == 0.0)
    same = ~opposite_signs
    # 0 <= sigma / a <= 2 is the TVD region of the Sweby diagram.
    ratio = sigma[same] / a[same]
    assert np.all(ratio >= -1e-12)
    assert np.all(ratio <= 2.0 + 1e-12)


def test_minmod_picks_the_smaller_slope():
    assert minmod(np.array([1.0]), np.array([3.0]))[0] == 1.0
    assert minmod(np.array([-3.0]), np.array([-1.0]))[0] == -1.0
    assert minmod(np.array([1.0]), np.array([-1.0]))[0] == 0.0


def test_superbee_is_the_most_compressive():
    a, b = np.array([1.0]), np.array([2.0])
    assert superbee(a, b)[0] >= minmod(a, b)[0]


def test_reconstruction_is_exact_for_linear_data():
    """A linear profile must be reconstructed continuously across interfaces."""
    U = np.stack([np.arange(9, dtype=float) + 1.0] * 3)
    left, right = reconstruct(U, order=2, limiter="mc", positivity_fix=False)
    np.testing.assert_allclose(left, right, rtol=1e-14)


def test_reconstruction_is_flat_at_an_extremum():
    """Limiters must kill the slope at a local maximum (no new extrema)."""
    profile = np.array([1.0, 1.0, 2.0, 1.0, 1.0])
    U = np.stack([profile] * 3)
    left, right = reconstruct(U, order=2, positivity_fix=False)
    np.testing.assert_allclose(left[0], [1.0, 2.0], atol=1e-14)


def test_first_and_second_order_return_the_same_interfaces():
    U = np.stack([np.linspace(1.0, 2.0, 12)] * 3)
    l1, r1 = reconstruct(U, order=1)
    l2, r2 = reconstruct(U, order=2, positivity_fix=False)
    assert l1.shape == l2.shape == r1.shape == r2.shape == (3, U.shape[1] - 3)


def test_reconstruction_falls_back_to_first_order_near_vacuum():
    """A slope that would create a negative pressure must be discarded."""
    rho = np.array([1.0, 1.0, 1.0, 0.02, 0.01])
    u = np.array([-2.0, -2.0, -2.0, 2.0, 2.0])
    p = np.array([0.4, 0.4, 0.4, 1e-4, 1e-5])
    U = primitive_to_conservative(rho, u, p)
    left, right = reconstruct(U, order=2, positivity_fix=True)
    for edge in (left, right):
        density = edge[0]
        pressure = 0.4 * (edge[2] - 0.5 * edge[1] ** 2 / density)
        assert np.all(density > 0.0)
        assert np.all(pressure > 0.0)


def test_reconstruction_requires_ghost_cells():
    with pytest.raises(ValueError, match="ghost"):
        reconstruct(np.ones((3, 4)), order=2)


def test_unknown_names_are_rejected():
    from euler1d.schemes import get_flux, get_limiter

    with pytest.raises(ValueError, match="unknown numerical flux"):
        get_flux("roe")
    with pytest.raises(ValueError, match="unknown limiter"):
        get_limiter("nope")
