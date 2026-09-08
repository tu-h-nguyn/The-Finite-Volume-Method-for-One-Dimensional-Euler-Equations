"""A finite-volume solver for the one-dimensional Euler equations.

The package accompanies the report *Phương pháp thể tích hữu hạn cho phương
trình Euler* (Finite volume methods for the Euler equations) and provides a
tested, vectorised implementation of the schemes discussed there:

* the exact Riemann solver (all four wave patterns, vacuum detection),
* Lax-Friedrichs, local Lax-Friedrichs (Rusanov), HLL and HLLC fluxes,
* MUSCL piecewise-linear reconstruction with four TVD limiters,
* forward Euler, Heun (SSP-RK2) and SSP-RK3 time integration.

Example
-------
>>> from euler1d import get_problem, preset, solve
>>> problem = get_problem("sod")
>>> config = problem.config(**{"nx": 200})
>>> solution = solve(config, problem.initial_condition)
>>> round(float(solution.density[0]), 3)
1.0
"""

from __future__ import annotations

from .errors import ConvergenceTable, l1_error, l2_error, linf_error, observed_order
from .problems import PROBLEMS, RiemannProblem, SmoothDensityWave, get_problem
from .riemann import GasState, StarState, exact_solution, solve_star_state
from .schemes import FLUXES, LIMITERS, TIME_INTEGRATORS, reconstruct
from .solver import PRESETS, Solution, SolverConfig, preset, solve
from .thermo import (
    GAMMA_AIR,
    conservative_to_primitive,
    flux,
    max_wave_speed,
    primitive_to_conservative,
    sound_speed,
)

__version__ = "1.0.0"

__all__ = [
    "__version__",
    "GAMMA_AIR",
    "GasState",
    "StarState",
    "SolverConfig",
    "Solution",
    "RiemannProblem",
    "SmoothDensityWave",
    "ConvergenceTable",
    "PROBLEMS",
    "PRESETS",
    "FLUXES",
    "LIMITERS",
    "TIME_INTEGRATORS",
    "solve",
    "preset",
    "get_problem",
    "solve_star_state",
    "exact_solution",
    "reconstruct",
    "flux",
    "sound_speed",
    "max_wave_speed",
    "primitive_to_conservative",
    "conservative_to_primitive",
    "l1_error",
    "l2_error",
    "linf_error",
    "observed_order",
]
