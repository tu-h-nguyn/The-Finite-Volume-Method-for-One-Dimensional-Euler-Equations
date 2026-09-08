"""Algebraic identities of the equation of state and the flux function."""

import numpy as np
import pytest

from euler1d import thermo


@pytest.fixture
def sample_state():
    rng = np.random.default_rng(20250908)
    rho = rng.uniform(0.1, 3.0, size=16)
    u = rng.uniform(-2.0, 2.0, size=16)
    p = rng.uniform(0.05, 5.0, size=16)
    return rho, u, p


def test_primitive_conservative_roundtrip(sample_state):
    rho, u, p = sample_state
    U = thermo.primitive_to_conservative(rho, u, p)
    rho_back, u_back, p_back = thermo.conservative_to_primitive(U)
    np.testing.assert_allclose(rho_back, rho, rtol=1e-14)
    np.testing.assert_allclose(u_back, u, rtol=1e-14)
    np.testing.assert_allclose(p_back, p, rtol=1e-13)


def test_flux_matches_closed_form(sample_state):
    rho, u, p = sample_state
    U = thermo.primitive_to_conservative(rho, u, p)
    F = thermo.flux(U)
    E = thermo.total_energy(rho, u, p)
    np.testing.assert_allclose(F[0], rho * u, rtol=1e-14)
    np.testing.assert_allclose(F[1], rho * u**2 + p, rtol=1e-13)
    np.testing.assert_allclose(F[2], (E + p) * u, rtol=1e-13)


def test_flux_is_galilean_consistent_for_zero_velocity():
    """With u = 0 only the momentum flux survives, and it equals the pressure."""
    U = thermo.primitive_to_conservative([1.0], [0.0], [2.5])
    F = thermo.flux(U)
    assert F[0, 0] == 0.0
    assert F[2, 0] == 0.0
    assert F[1, 0] == pytest.approx(2.5)


def test_max_wave_speed_is_the_largest_characteristic(sample_state):
    rho, u, p = sample_state
    U = thermo.primitive_to_conservative(rho, u, p)
    a = thermo.sound_speed(rho, p)
    assert thermo.max_wave_speed(U) == pytest.approx(np.max(np.abs(u) + a))


def test_sound_speed_of_air_at_reference_state():
    assert thermo.sound_speed(1.0, 1.0) == pytest.approx(np.sqrt(1.4))
