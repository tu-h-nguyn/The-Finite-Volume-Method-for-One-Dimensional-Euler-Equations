"""Benchmark problems with known exact solutions.

Two families are provided:

* :class:`RiemannProblem` — shock-tube tests, including Sod's problem used
  throughout the report and four harder cases from Toro's test suite.  These
  contain discontinuities, so they measure *robustness* and *resolution*, not
  order of accuracy.
* :class:`SmoothDensityWave` — a smooth, periodic, exactly-advected density
  profile.  Because the solution stays analytic, this is the test that can
  actually confirm the design order of the schemes.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from .riemann import GasState, exact_solution
from .solver import SolverConfig
from .thermo import GAMMA_AIR, primitive_to_conservative

__all__ = ["RiemannProblem", "SmoothDensityWave", "PROBLEMS", "get_problem"]


@dataclass(frozen=True)
class RiemannProblem:
    """A shock-tube problem together with its exact solution."""

    name: str
    left: GasState
    right: GasState
    t_final: float
    x_min: float = 0.0
    x_max: float = 1.0
    x0: float = 0.5
    gamma: float = GAMMA_AIR
    description: str = ""
    boundary: str = "transmissive"

    def initial_condition(self, x: np.ndarray) -> np.ndarray:
        rho = np.where(x < self.x0, self.left.rho, self.right.rho)
        u = np.where(x < self.x0, self.left.u, self.right.u)
        p = np.where(x < self.x0, self.left.p, self.right.p)
        return primitive_to_conservative(rho, u, p, self.gamma)

    def exact(self, x: np.ndarray, t: float | None = None):
        """Exact primitive solution ``(rho, u, p)`` at time ``t``."""
        t = self.t_final if t is None else t
        return exact_solution(x, t, self.left, self.right, self.gamma, self.x0)

    def config(self, **overrides: Any) -> SolverConfig:
        """A :class:`~euler1d.solver.SolverConfig` matching this problem."""
        return replace(
            SolverConfig(
                x_min=self.x_min,
                x_max=self.x_max,
                t_final=self.t_final,
                gamma=self.gamma,
                boundary=self.boundary,
            ),
            **overrides,
        )


@dataclass(frozen=True)
class SmoothDensityWave:
    """Uniform pressure and velocity, sinusoidal density: a pure contact wave.

    With ``u`` and ``p`` constant the Euler system degenerates to linear
    advection of the density, so

    .. math:: \\rho(x, t) = \\rho_0(x - u t)

    is an exact solution for all times.  It is periodic and infinitely smooth,
    which makes it the right vehicle for a grid-convergence study.
    """

    name: str = "smooth_density_wave"
    amplitude: float = 0.5
    u0: float = 1.0
    p0: float = 1.0
    t_final: float = 0.5
    x_min: float = 0.0
    x_max: float = 1.0
    gamma: float = GAMMA_AIR
    description: str = "smooth advected density profile (periodic)"
    boundary: str = "periodic"

    def _density(self, x: np.ndarray) -> np.ndarray:
        return 1.0 + self.amplitude * np.sin(2.0 * np.pi * x)

    def initial_condition(self, x: np.ndarray) -> np.ndarray:
        rho = self._density(x)
        return primitive_to_conservative(
            rho, np.full_like(rho, self.u0), np.full_like(rho, self.p0), self.gamma
        )

    def exact(self, x: np.ndarray, t: float | None = None):
        t = self.t_final if t is None else t
        period = self.x_max - self.x_min
        shifted = self.x_min + np.mod(x - self.u0 * t - self.x_min, period)
        rho = self._density(shifted)
        return rho, np.full_like(rho, self.u0), np.full_like(rho, self.p0)

    def config(self, **overrides: Any) -> SolverConfig:
        """A :class:`~euler1d.solver.SolverConfig` matching this problem."""
        return replace(
            SolverConfig(
                x_min=self.x_min,
                x_max=self.x_max,
                t_final=self.t_final,
                gamma=self.gamma,
                boundary=self.boundary,
            ),
            **overrides,
        )


PROBLEMS: dict[str, RiemannProblem | SmoothDensityWave] = {
    "sod": RiemannProblem(
        name="sod",
        left=GasState(1.0, 0.0, 1.0),
        right=GasState(0.125, 0.0, 0.1),
        t_final=0.2,
        description="Sod (1978): left rarefaction, contact, right shock",
    ),
    "lax": RiemannProblem(
        name="lax",
        left=GasState(0.445, 0.698, 3.528),
        right=GasState(0.5, 0.0, 0.571),
        t_final=0.14,
        description="Lax problem: strong contact, moderate shock",
    ),
    "toro2": RiemannProblem(
        name="toro2",
        left=GasState(1.0, -2.0, 0.4),
        right=GasState(1.0, 2.0, 0.4),
        t_final=0.15,
        description="123 problem: two strong rarefactions, near-vacuum star state",
    ),
    "toro3": RiemannProblem(
        name="toro3",
        left=GasState(1.0, 0.0, 1000.0),
        right=GasState(1.0, 0.0, 0.01),
        t_final=0.012,
        description="left half of the Woodward-Colella blast wave: very strong shock",
    ),
    "toro4": RiemannProblem(
        name="toro4",
        left=GasState(5.99924, 19.5975, 460.894),
        right=GasState(5.99242, -6.19633, 46.0950),
        t_final=0.035,
        x0=0.4,
        description="collision of two strong shocks",
    ),
    "smooth": SmoothDensityWave(),
}


def get_problem(name: str):
    try:
        return PROBLEMS[name]
    except KeyError:
        raise ValueError(
            f"unknown problem {name!r}; available: {sorted(PROBLEMS)}"
        ) from None
