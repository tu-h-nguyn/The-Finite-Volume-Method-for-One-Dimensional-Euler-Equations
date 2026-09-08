r"""Numerical flux functions, slope limiters and time integrators.

Every numerical flux has the same signature

.. code-block:: python

    F = numerical_flux(U_left, U_right, gamma, dx_over_dt)

where ``U_left`` / ``U_right`` are the reconstructed states on either side of
each cell interface, stored as ``(3, n_interfaces)`` arrays.  ``dx_over_dt`` is
only used by the classical Lax-Friedrichs flux, whose numerical viscosity is
tied to the mesh ratio; the other fluxes ignore it.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from .thermo import GAMMA_AIR, conservative_to_primitive, flux, sound_speed

__all__ = [
    "FLUXES",
    "LIMITERS",
    "TIME_INTEGRATORS",
    "lax_friedrichs_flux",
    "rusanov_flux",
    "hll_flux",
    "hllc_flux",
    "minmod",
    "mc_limiter",
    "superbee",
    "van_leer",
    "reconstruct",
    "get_flux",
    "get_limiter",
]

NumericalFlux = Callable[[np.ndarray, np.ndarray, float, float], np.ndarray]


# ---------------------------------------------------------------------------
# Numerical fluxes
# ---------------------------------------------------------------------------
def lax_friedrichs_flux(
    UL: np.ndarray, UR: np.ndarray, gamma: float = GAMMA_AIR, dx_over_dt: float = 1.0
) -> np.ndarray:
    r"""Classical Lax-Friedrichs flux.

    .. math::

        \mathbf{F}^{LF}_{i+1/2} = \tfrac12\bigl(\mathbf{F}_L + \mathbf{F}_R\bigr)
        - \frac{\Delta x}{2\,\Delta t}\bigl(\mathbf{U}_R - \mathbf{U}_L\bigr)

    The dissipation coefficient is the mesh ratio itself, which is at least as
    large as any physical wave speed under the CFL condition — hence the scheme
    is very robust and very diffusive.
    """
    return 0.5 * (flux(UL, gamma) + flux(UR, gamma)) - 0.5 * dx_over_dt * (UR - UL)


def rusanov_flux(
    UL: np.ndarray, UR: np.ndarray, gamma: float = GAMMA_AIR, dx_over_dt: float = 1.0
) -> np.ndarray:
    r"""Local Lax-Friedrichs (Rusanov) flux.

    Identical in form to :func:`lax_friedrichs_flux`, but the viscosity uses the
    *local* maximum wave speed :math:`\alpha_{i+1/2} = \max(|u| + a)` taken over
    the two neighbouring states, which is much smaller than
    :math:`\Delta x/\Delta t` away from the fastest wave.
    """
    alpha = _local_max_speed(UL, UR, gamma)
    return 0.5 * (flux(UL, gamma) + flux(UR, gamma)) - 0.5 * alpha * (UR - UL)


def _local_max_speed(UL: np.ndarray, UR: np.ndarray, gamma: float) -> np.ndarray:
    rho_l, u_l, p_l = conservative_to_primitive(UL, gamma)
    rho_r, u_r, p_r = conservative_to_primitive(UR, gamma)
    a_l = sound_speed(rho_l, p_l, gamma)
    a_r = sound_speed(rho_r, p_r, gamma)
    return np.maximum(np.abs(u_l) + a_l, np.abs(u_r) + a_r)


def _wave_speed_estimates(
    UL: np.ndarray, UR: np.ndarray, gamma: float
) -> tuple[np.ndarray, np.ndarray, tuple]:
    """Pressure-based signal speeds (Toro, sec. 10.5.2)."""
    rho_l, u_l, p_l = conservative_to_primitive(UL, gamma)
    rho_r, u_r, p_r = conservative_to_primitive(UR, gamma)
    a_l = sound_speed(rho_l, p_l, gamma)
    a_r = sound_speed(rho_r, p_r, gamma)

    rho_bar = 0.5 * (rho_l + rho_r)
    a_bar = 0.5 * (a_l + a_r)
    p_star = np.maximum(0.0, 0.5 * (p_l + p_r) - 0.5 * (u_r - u_l) * rho_bar * a_bar)

    def q(p_k, a_k):
        ratio = p_star / p_k
        return np.where(
            ratio <= 1.0,
            1.0,
            np.sqrt(1.0 + (gamma + 1.0) / (2.0 * gamma) * np.maximum(ratio - 1.0, 0.0)),
        )

    s_l = u_l - a_l * q(p_l, a_l)
    s_r = u_r + a_r * q(p_r, a_r)
    return s_l, s_r, (rho_l, u_l, p_l, rho_r, u_r, p_r)


def hll_flux(
    UL: np.ndarray, UR: np.ndarray, gamma: float = GAMMA_AIR, dx_over_dt: float = 1.0
) -> np.ndarray:
    """Harten-Lax-van Leer flux: a two-wave approximate Riemann solver."""
    s_l, s_r, _ = _wave_speed_estimates(UL, UR, gamma)
    f_l, f_r = flux(UL, gamma), flux(UR, gamma)
    f_hll = (s_r * f_l - s_l * f_r + s_l * s_r * (UR - UL)) / (s_r - s_l)
    return np.where(s_l >= 0.0, f_l, np.where(s_r <= 0.0, f_r, f_hll))


def hllc_flux(
    UL: np.ndarray, UR: np.ndarray, gamma: float = GAMMA_AIR, dx_over_dt: float = 1.0
) -> np.ndarray:
    """HLLC flux — HLL restored with the missing contact wave.

    Not part of the original report; included to show what the contact-resolving
    upgrade buys over Rusanov/HLL on the same mesh.
    """
    s_l, s_r, (rho_l, u_l, p_l, rho_r, u_r, p_r) = _wave_speed_estimates(UL, UR, gamma)
    f_l, f_r = flux(UL, gamma), flux(UR, gamma)

    numerator = p_r - p_l + rho_l * u_l * (s_l - u_l) - rho_r * u_r * (s_r - u_r)
    denominator = rho_l * (s_l - u_l) - rho_r * (s_r - u_r)
    s_star = numerator / denominator

    def star_state(U, rho, u, p, s):
        factor = rho * (s - u) / (s - s_star)
        energy = U[2] / rho + (s_star - u) * (s_star + p / (rho * (s - u)))
        return factor * np.stack([np.ones_like(s_star), s_star, energy])

    u_star_l = star_state(UL, rho_l, u_l, p_l, s_l)
    u_star_r = star_state(UR, rho_r, u_r, p_r, s_r)

    f_star_l = f_l + s_l * (u_star_l - UL)
    f_star_r = f_r + s_r * (u_star_r - UR)

    return np.select(
        [s_l >= 0.0, s_star >= 0.0, s_r >= 0.0],
        [f_l, f_star_l, f_star_r],
        default=f_r,
    )


FLUXES: dict[str, NumericalFlux] = {
    "lax_friedrichs": lax_friedrichs_flux,
    "rusanov": rusanov_flux,
    "hll": hll_flux,
    "hllc": hllc_flux,
}
#: Convenient aliases matching the notation used in the report.
FLUXES["lf"] = lax_friedrichs_flux
FLUXES["llf"] = rusanov_flux


def get_flux(name: str) -> NumericalFlux:
    try:
        return FLUXES[name]
    except KeyError:
        raise ValueError(
            f"unknown numerical flux {name!r}; available: {sorted(FLUXES)}"
        ) from None


# ---------------------------------------------------------------------------
# Slope limiters
# ---------------------------------------------------------------------------
def minmod(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Minmod limiter — the most diffusive, and the one used in the report."""
    return np.where(a * b <= 0.0, 0.0, np.where(np.abs(a) < np.abs(b), a, b))


def mc_limiter(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Monotonized-central limiter (van Leer's MC)."""
    return np.where(
        a * b <= 0.0,
        0.0,
        np.sign(a) * np.minimum(np.minimum(2.0 * np.abs(a), 2.0 * np.abs(b)),
                                0.5 * np.abs(a + b)),
    )


def superbee(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Roe's superbee limiter — the sharpest of the second-order TVD family."""
    s = np.sign(a)
    return np.where(
        a * b <= 0.0,
        0.0,
        s * np.maximum(
            np.minimum(2.0 * np.abs(a), np.abs(b)),
            np.minimum(np.abs(a), 2.0 * np.abs(b)),
        ),
    )


def van_leer(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Van Leer's smooth, differentiable limiter."""
    denominator = a + b
    safe = np.where(np.abs(denominator) < 1e-300, 1.0, denominator)
    return np.where(a * b <= 0.0, 0.0, 2.0 * a * b / safe)


LIMITERS: dict[str, Callable[[np.ndarray, np.ndarray], np.ndarray]] = {
    "minmod": minmod,
    "mc": mc_limiter,
    "superbee": superbee,
    "van_leer": van_leer,
}


def get_limiter(name: str):
    try:
        return LIMITERS[name]
    except KeyError:
        raise ValueError(
            f"unknown limiter {name!r}; available: {sorted(LIMITERS)}"
        ) from None


def _positivity_fallback(
    U: np.ndarray, slopes: np.ndarray, gamma: float
) -> np.ndarray:
    """Drop to first order in cells whose reconstruction is not admissible.

    A limited linear reconstruction is TVD in each conserved variable, but that
    does not guarantee a positive *pressure* at the cell edges: near-vacuum
    states such as Toro's 123 problem break it.  Where either edge value has a
    non-positive density or pressure the slope is zeroed, which is the standard
    (and locally first-order) safeguard.
    """
    admissible = np.ones(slopes.shape[1], dtype=bool)
    for edge in (U + 0.5 * slopes, U - 0.5 * slopes):
        rho = edge[0]
        safe_rho = np.where(rho > 0.0, rho, 1.0)
        p = (gamma - 1.0) * (edge[2] - 0.5 * edge[1] ** 2 / safe_rho)
        admissible &= (rho > 0.0) & (p > 0.0)
    return np.where(admissible, slopes, 0.0)


def reconstruct(
    U: np.ndarray,
    order: int = 2,
    limiter: str = "minmod",
    gamma: float = GAMMA_AIR,
    positivity_fix: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    r"""Reconstruct interface states from cell averages.

    With ``order=1`` the data are piecewise constant and the interface states
    are simply the neighbouring cell averages.  With ``order=2`` a limited
    piecewise-linear (MUSCL) reconstruction is used,

    .. math::

        \mathbf{U}^{L}_{i+1/2} = \mathbf{U}_i + \tfrac12 \sigma_i , \qquad
        \mathbf{U}^{R}_{i+1/2} = \mathbf{U}_{i+1} - \tfrac12 \sigma_{i+1} ,

    with the slope :math:`\sigma_i` given by a TVD limiter applied to the
    forward and backward differences.

    ``U`` must carry two ghost cells on each side.  Both orders return the same
    ``n - 3`` interfaces, so the two reconstructions are interchangeable inside
    the solver.
    """
    if U.shape[1] < 5:
        raise ValueError("reconstruction needs at least two ghost cells per side")
    if order == 1:
        return U[:, 1:-2], U[:, 2:-1]
    if order != 2:
        raise ValueError("only first- and second-order reconstruction are implemented")

    limit = get_limiter(limiter)
    backward = U[:, 1:-1] - U[:, :-2]
    forward = U[:, 2:] - U[:, 1:-1]
    slopes = limit(backward, forward)  # slopes for cells 1 .. n-2
    if positivity_fix:
        slopes = _positivity_fallback(U[:, 1:-1], slopes, gamma)

    left = U[:, 1:-1] + 0.5 * slopes  # left state of interfaces 1+1/2 ...
    right = U[:, 1:-1] - 0.5 * slopes  # right state of interfaces ...-1/2
    return left[:, :-1], right[:, 1:]


# ---------------------------------------------------------------------------
# Time integrators (SSP Runge-Kutta family)
# ---------------------------------------------------------------------------
def forward_euler(U: np.ndarray, dt: float, rhs: Callable) -> np.ndarray:
    """First-order explicit Euler step."""
    return U + dt * rhs(U)


def heun(U: np.ndarray, dt: float, rhs: Callable) -> np.ndarray:
    r"""Heun's method, i.e. the two-stage SSP Runge-Kutta scheme SSP-RK2.

    .. math::

        \mathbf{U}^{*} = \mathbf{U}^n + \Delta t\, L(\mathbf{U}^n), \qquad
        \mathbf{U}^{n+1} = \tfrac12\bigl(\mathbf{U}^n + \mathbf{U}^{*}
                          + \Delta t\, L(\mathbf{U}^{*})\bigr).
    """
    u1 = U + dt * rhs(U)
    return 0.5 * (U + u1 + dt * rhs(u1))


def ssp_rk3(U: np.ndarray, dt: float, rhs: Callable) -> np.ndarray:
    """Third-order strong-stability-preserving Runge-Kutta (Shu-Osher)."""
    u1 = U + dt * rhs(U)
    u2 = 0.75 * U + 0.25 * (u1 + dt * rhs(u1))
    return (U + 2.0 * (u2 + dt * rhs(u2))) / 3.0


TIME_INTEGRATORS: dict[str, Callable] = {
    "euler": forward_euler,
    "heun": heun,
    "ssprk2": heun,
    "ssprk3": ssp_rk3,
}
