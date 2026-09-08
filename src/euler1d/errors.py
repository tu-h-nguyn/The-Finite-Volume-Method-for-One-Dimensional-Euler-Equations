"""Error norms and grid-convergence utilities."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["l1_error", "l2_error", "linf_error", "ConvergenceTable", "observed_order"]


def l1_error(numeric: np.ndarray, exact: np.ndarray, dx: float) -> float:
    r"""Grid-normalised :math:`L^1` error :math:`\Delta x \sum |e_i|`."""
    return float(np.sum(np.abs(numeric - exact)) * dx)


def l2_error(numeric: np.ndarray, exact: np.ndarray, dx: float) -> float:
    return float(np.sqrt(np.sum((numeric - exact) ** 2) * dx))


def linf_error(numeric: np.ndarray, exact: np.ndarray, dx: float = 1.0) -> float:
    return float(np.max(np.abs(numeric - exact)))


def observed_order(errors: list[float], resolutions: list[int]) -> list[float]:
    r"""Observed order of accuracy between successive refinements.

    .. math::

        q = \frac{\log(e_{k-1} / e_k)}{\log(N_k / N_{k-1})}
    """
    orders = [float("nan")]
    for k in range(1, len(errors)):
        if errors[k] <= 0.0 or errors[k - 1] <= 0.0:
            orders.append(float("nan"))
            continue
        orders.append(
            float(
                np.log(errors[k - 1] / errors[k])
                / np.log(resolutions[k] / resolutions[k - 1])
            )
        )
    return orders


@dataclass
class ConvergenceTable:
    """Errors and observed orders for a sequence of refined grids."""

    resolutions: list[int]
    errors: list[float]
    norm: str = "L1"
    label: str = ""

    @property
    def orders(self) -> list[float]:
        return observed_order(self.errors, self.resolutions)

    @property
    def asymptotic_order(self) -> float:
        """Order measured on the two finest grids."""
        return self.orders[-1]

    def to_markdown(self) -> str:
        lines = [
            f"| N | {self.norm} error | observed order |",
            "|---:|---:|---:|",
        ]
        for n, e, q in zip(self.resolutions, self.errors, self.orders):
            order = "—" if np.isnan(q) else f"{q:.2f}"
            lines.append(f"| {n} | {e:.3e} | {order} |")
        return "\n".join(lines)

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        header = f"{'N':>6} {self.norm + ' error':>14} {'order':>8}"
        rows = [header, "-" * len(header)]
        for n, e, q in zip(self.resolutions, self.errors, self.orders):
            order = "     --- " if np.isnan(q) else f"{q:8.2f}"
            rows.append(f"{n:6d} {e:14.4e} {order}")
        return "\n".join(rows)
