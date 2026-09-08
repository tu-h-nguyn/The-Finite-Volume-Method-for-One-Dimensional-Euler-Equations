r"""Finite-volume driver for the 1-D Euler equations.

The semi-discrete form of the conservation law on a uniform grid is

.. math::

    \frac{\mathrm{d}\mathbf{U}_i}{\mathrm{d}t}
      = -\frac{1}{\Delta x}
        \bigl(\mathbf{F}_{i+1/2} - \mathbf{F}_{i-1/2}\bigr)
      \;=:\; L(\mathbf{U})_i ,

which is then advanced in time by one of the SSP Runge-Kutta integrators of
:mod:`euler1d.schemes`.  Boundaries are handled with two ghost cells per side,
so the interior update is uniform and no special-cased first/last cell is
needed — this is the main structural difference from the MATLAB prototype.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field, replace

import numpy as np

from .schemes import TIME_INTEGRATORS, get_flux, reconstruct
from .thermo import GAMMA_AIR, conservative_to_primitive, max_wave_speed, sound_speed

__all__ = ["SolverConfig", "Solution", "solve", "PRESETS", "preset"]

N_GHOST = 2


@dataclass(frozen=True)
class SolverConfig:
    """Everything that defines one numerical experiment."""

    nx: int = 200
    x_min: float = 0.0
    x_max: float = 1.0
    t_final: float = 0.2
    cfl: float = 0.5
    gamma: float = GAMMA_AIR
    flux: str = "rusanov"
    order_space: int = 1
    limiter: str = "minmod"
    time_integrator: str = "euler"
    boundary: str = "transmissive"
    max_steps: int = 200_000
    label: str = ""

    @property
    def dx(self) -> float:
        return (self.x_max - self.x_min) / self.nx

    @property
    def cell_centers(self) -> np.ndarray:
        return self.x_min + (np.arange(self.nx) + 0.5) * self.dx

    def describe(self) -> str:
        space = "1st order" if self.order_space == 1 else f"MUSCL/{self.limiter}"
        return f"{self.flux} + {space} + {self.time_integrator}"


@dataclass
class Solution:
    """Result of a run, with primitive variables computed on demand."""

    x: np.ndarray
    U: np.ndarray
    t: float
    steps: int
    config: SolverConfig
    history: list[float] = field(default_factory=list)

    @property
    def primitives(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return conservative_to_primitive(self.U, self.config.gamma)

    @property
    def density(self) -> np.ndarray:
        return self.primitives[0]

    @property
    def velocity(self) -> np.ndarray:
        return self.primitives[1]

    @property
    def pressure(self) -> np.ndarray:
        return self.primitives[2]

    @property
    def specific_internal_energy(self) -> np.ndarray:
        rho, _, p = self.primitives
        return p / ((self.config.gamma - 1.0) * rho)

    @property
    def mach_number(self) -> np.ndarray:
        rho, u, p = self.primitives
        return np.abs(u) / sound_speed(rho, p, self.config.gamma)

    def conserved_totals(self) -> np.ndarray:
        r"""Integrals :math:`\int \mathbf{U}\,\mathrm{d}x` over the domain."""
        return self.U.sum(axis=1) * self.config.dx


#: Named configurations reproducing (and extending) the methods of the report.
PRESETS: dict[str, dict] = {
    "lax_friedrichs": dict(
        flux="lax_friedrichs", order_space=1, time_integrator="euler",
        label="Lax-Friedrichs",
    ),
    "local_lax_friedrichs": dict(
        flux="rusanov", order_space=1, time_integrator="euler",
        label="Local Lax-Friedrichs",
    ),
    "second_order_space": dict(
        flux="rusanov", order_space=2, limiter="minmod", time_integrator="euler",
        label="2nd order in space (MUSCL-minmod)",
    ),
    "second_order_time": dict(
        flux="rusanov", order_space=1, time_integrator="heun",
        label="2nd order in time (Heun)",
    ),
    "second_order_combined": dict(
        flux="rusanov", order_space=2, limiter="minmod", time_integrator="heun",
        label="2nd order in space and time",
    ),
    "hllc_muscl": dict(
        flux="hllc", order_space=2, limiter="mc", time_integrator="ssprk3",
        label="HLLC + MUSCL-MC + SSP-RK3",
    ),
}


def preset(name: str, **overrides) -> SolverConfig:
    """Build a :class:`SolverConfig` from a named method of :data:`PRESETS`."""
    try:
        base = PRESETS[name]
    except KeyError:
        raise ValueError(
            f"unknown scheme {name!r}; available: {sorted(PRESETS)}"
        ) from None
    return replace(SolverConfig(**base), **overrides)


def _apply_boundary(U: np.ndarray, boundary: str) -> np.ndarray:
    """Extend the interior solution with two ghost cells on each side."""
    if boundary == "transmissive":
        return np.concatenate(
            [U[:, :1], U[:, :1], U, U[:, -1:], U[:, -1:]], axis=1
        )
    if boundary == "periodic":
        return np.concatenate([U[:, -N_GHOST:], U, U[:, :N_GHOST]], axis=1)
    raise ValueError(f"unknown boundary condition {boundary!r}")


def _check_admissible(U: np.ndarray, gamma: float, t: float) -> None:
    rho, _, p = conservative_to_primitive(U, gamma)
    if not (np.all(np.isfinite(U)) and np.all(rho > 0.0) and np.all(p > 0.0)):
        raise FloatingPointError(
            f"solution left the physically admissible set at t = {t:.6g} "
            "(negative density or pressure); try a smaller CFL number"
        )


def solve(
    config: SolverConfig,
    initial_condition: Callable[[np.ndarray], np.ndarray] | np.ndarray,
    record_totals: bool = False,
) -> Solution:
    """Integrate the Euler equations to ``config.t_final``.

    Parameters
    ----------
    config:
        Discretisation and scheme selection.
    initial_condition:
        Either a ``(3, nx)`` array of cell averages or a callable mapping the
        vector of cell centres to such an array.
    record_totals:
        Store the total mass after each step, which the tests use to verify
        discrete conservation.
    """
    x = config.cell_centers
    U = (
        np.array(initial_condition, dtype=float)
        if not callable(initial_condition)
        else np.asarray(initial_condition(x), dtype=float)
    )
    if U.shape != (3, config.nx):
        raise ValueError(f"initial condition must have shape (3, {config.nx})")

    numerical_flux = get_flux(config.flux)
    try:
        integrate = TIME_INTEGRATORS[config.time_integrator]
    except KeyError:
        raise ValueError(
            f"unknown time integrator {config.time_integrator!r}; "
            f"available: {sorted(TIME_INTEGRATORS)}"
        ) from None

    dx = config.dx
    gamma = config.gamma
    history: list[float] = []

    def make_rhs(dx_over_dt: float) -> Callable[[np.ndarray], np.ndarray]:
        def rhs(state: np.ndarray) -> np.ndarray:
            extended = _apply_boundary(state, config.boundary)
            UL, UR = reconstruct(
                extended, config.order_space, config.limiter, gamma
            )
            F = numerical_flux(UL, UR, gamma, dx_over_dt)
            return -(F[:, 1:] - F[:, :-1]) / dx

        return rhs

    t = 0.0
    step = 0
    while t < config.t_final - 1e-14:
        if step >= config.max_steps:
            raise RuntimeError(
                f"step limit {config.max_steps} reached at t = {t:.6g}; "
                "the CFL condition may be forcing an unreasonably small dt"
            )
        _check_admissible(U, gamma, t)
        dt = config.cfl * dx / max_wave_speed(U, gamma)
        dt = min(dt, config.t_final - t)

        U = integrate(U, dt, make_rhs(dx / dt))
        t += dt
        step += 1
        if record_totals:
            history.append(float(U[0].sum() * dx))

    _check_admissible(U, gamma, t)
    return Solution(x=x, U=U, t=t, steps=step, config=config, history=history)
