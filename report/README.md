# Report and slides

The written part of the project, in Vietnamese.

| Path | Contents |
|---|---|
| `master.tex` | the report's root document |
| `report.pdf` | pre-built PDF |
| `chapter/` | chapters: gas dynamics, the Riemann problem, the finite-volume schemes, results |
| `figure/` | TikZ figures, most of them exported from the MATLAB scripts |
| `utils/` | front matter, conclusion, bibliography |
| `slide/` | beamer slides (`presentation_slides.pdf`) |

## Building

```bash
make report          # from the repository root
# or
cd report && latexmk -pdf master.tex
```

Requires a TeX distribution with `babel-vietnamese`, `pgfplots` and `tikz`.
Build artifacts (`.aux`, `.log`, `.toc`, …) are ignored by git — do not commit them.
