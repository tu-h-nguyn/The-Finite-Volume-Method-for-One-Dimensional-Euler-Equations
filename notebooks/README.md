# Notebooks

Exploratory notebooks written while the report was being drafted.

| Notebook | Contents |
|---|---|
| `fvm-with-theory.ipynb` | derivations and the finite-volume schemes, worked through step by step |
| `fvm-final.ipynb` | the final runs and figures used in the report |

They are a record of how the work developed. For anything you want to *run*, prefer the
package:

```python
from euler1d import get_problem, preset, solve
```

or the CLI (`euler1d run --help`).
