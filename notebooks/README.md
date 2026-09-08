# Notebooks

| Notebook | Kernel | Contents |
|---|---|---|
| [`demo.ipynb`](demo.ipynb) | Python 3 | A guided tour of the `euler1d` package: the exact Riemann solver against Toro's tabulated values, one run plotted against the exact solution, the accuracy comparison of every scheme, the measured order of accuracy, all five benchmarks, and a conservation check. |
| `fvm-with-theory.ipynb` | MATLAB | Derivations and the finite-volume schemes, worked through step by step while the report was being drafted. |
| `fvm-final.ipynb` | MATLAB | The final runs and figures used in the report. |

## Running them

`demo.ipynb` is **generated and executed** by
[`../scripts/build_demo_notebook.py`](../scripts/build_demo_notebook.py) rather than edited by
hand, so its committed outputs are always what the current code produces — CI re-executes it on
every push. The build is deterministic (numbered cell ids, no execution timestamps), so
rebuilding it when nothing has changed leaves the working tree clean. Regenerate it with:

```bash
pip install -e ".[dev,notebook]"
make notebook
```

The other two use a **MATLAB kernel**: reading them on GitHub works (their outputs are saved),
but re-running them needs MATLAB plus a MATLAB Jupyter kernel. They are kept as a record of how
the work developed. For anything you want to run, use the Python package or the CLI:

```python
from euler1d import get_problem, preset, solve
```

```bash
euler1d run --help
```
