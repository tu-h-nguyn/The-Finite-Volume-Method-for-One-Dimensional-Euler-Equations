# MATLAB prototype

These are the scripts the report was originally written against. They are kept for
provenance and are **not** the maintained implementation — see [`../src/euler1d/`](../src/euler1d/)
for the tested Python package.

## Contents

| File | Method |
|---|---|
| `riemann_exact_solution.m` | exact solution of Sod's problem (assumes a left rarefaction and a right shock) |
| `lax_friedrichs_solution.m` | Lax-Friedrichs |
| `local_lax_friedrichs_solution.m` | local Lax-Friedrichs |
| `second_order_spatial_solution.m` | MUSCL piecewise-linear reconstruction with minmod |
| `second_order_temporal_solution.m` | Heun's method |
| `second_order_combined_solution.m` | second order in both space and time |
| `compare_all_methods.m` | overlay of every method against the exact solution |
| `run_all.m` | driver that runs all of the above |
| `compute_flux.m`, `minmod.m` | shared helpers |

## Running

```matlab
cd matlab
run_all
```

The scripts export TikZ figures into `../report/figure/` via
[`matlab2tikz`](https://github.com/matlab2tikz/matlab2tikz), which must be on the MATLAB
path; the report includes those `.tex` files directly.

## Known limitations

Two of these are fixed in the Python package:

* the exact solver hard-codes Sod's wave pattern and fails for data that produce a left
  shock or a right rarefaction;
* boundaries are handled by overwriting the first and last cell after the update rather
  than with ghost cells, so the boundary treatment is not part of the conservative update.
