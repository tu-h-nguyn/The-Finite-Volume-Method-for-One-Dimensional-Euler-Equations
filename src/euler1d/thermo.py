r"""Thermodynamics and flux algebra for the 1-D compressible Euler equations.

State vectors are stored as ``(3, N)`` arrays of *conservative* variables

.. math::

    \\mathbf{U} = (\\rho,\; \\rho u,\; E)^\\top ,
    \\qquad
    E = \\frac{p}{\\gamma - 1} + \\tfrac12 \\rho u^2 ,

so that a whole grid can be advanced with vectorised NumPy expressions instead
of the cell-by-cell loops used in the original MATLAB prototype.
"""

from __future__ import annotations

import numpy as np

#: Ratio of specific heats for a diatomic ideal gas (air).
GAMMA_AIR = 1.4

__all__ = [
    "GAMMA_AIR",
    "conservative_to_primitive",
    "primitive_to_conservative",
    "flux",
    "sound_speed",
    "pressure",
    "max_wave_speed",
    "total_energy",
]


def total_energy(rho, u, p, gamma: float = GAMMA_AIR):
    """Total energy per unit volume from primitive variables."""
    return np.asarray(p) / (gamma - 1.0) + 0.5 * np.asarray(rho) * np.asarray(u) ** 2


def primitive_to_conservative(rho, u, p, gamma: float = GAMMA_AIR) -> np.ndarray:
    """Pack primitive variables ``(rho, u, p)`` into conservative form."""
    rho = np.asarray(rho, dtype=float)
    u = np.asarray(u, dtype=float)
    p = np.asarray(p, dtype=float)
    return np.stack([rho, rho * u, total_energy(rho, u, p, gamma)])


def conservative_to_primitive(
    U: np.ndarray, gamma: float = GAMMA_AIR
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Unpack conservative variables into ``(rho, u, p)``."""
    U = np.asarray(U, dtype=float)
    rho = U[0]
    u = U[1] / rho
    p = (gamma - 1.0) * (U[2] - 0.5 * rho * u**2)
    return rho, u, p


def pressure(U: np.ndarray, gamma: float = GAMMA_AIR) -> np.ndarray:
    """Pressure recovered from the ideal-gas equation of state."""
    return conservative_to_primitive(U, gamma)[2]


def sound_speed(rho, p, gamma: float = GAMMA_AIR) -> np.ndarray:
    r"""Speed of sound :math:`a = \sqrt{\gamma p / \rho}`."""
    return np.sqrt(gamma * np.asarray(p) / np.asarray(rho))


def flux(U: np.ndarray, gamma: float = GAMMA_AIR) -> np.ndarray:
    r"""Physical flux :math:`\mathbf{F}(\mathbf{U})` of the Euler system.

    .. math::

        \mathbf{F} = \bigl(\rho u,\; \rho u^2 + p,\; (E + p) u\bigr)^\top .
    """
    rho, u, p = conservative_to_primitive(U, gamma)
    return np.stack([rho * u, rho * u**2 + p, (U[2] + p) * u])


def max_wave_speed(U: np.ndarray, gamma: float = GAMMA_AIR) -> float:
    r"""Largest characteristic speed :math:`\max(|u| + a)` on the grid.

    This is the quantity that sets the explicit CFL time step.
    """
    rho, u, p = conservative_to_primitive(U, gamma)
    return float(np.max(np.abs(u) + sound_speed(rho, p, gamma)))
