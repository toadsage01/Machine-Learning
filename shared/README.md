# `shared/` — project-wide utilities

This package is imported by every project in the `Machine-Learning` monorepo
to enforce a single, consistent visual identity across all generated figures.

## What lives here

| File                  | Purpose                                                                 |
|-----------------------|-------------------------------------------------------------------------|
| `plot_style.mplstyle` | Project-wide matplotlib theme (color cycle, fonts, grids, spines, …). |
| `__init__.py`         | Python helper that locates & applies the style from any sub-project.   |

## Usage from any project

```python
# Option A — explicit (preferred inside scripts):
import matplotlib.pyplot as plt
plt.style.use("/abs/path/to/Machine-Learning/shared/plot_style.mplstyle")

# Option B — auto-apply via the helper package (preferred inside packages):
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))  # repo root
from shared import apply_style
apply_style()
```

## Design notes

- **Color cycle**: Okabe-Ito-inspired, colorblind-safe, ordered for high
  contrast on white backgrounds.
- **Typography**: sans-serif (`Inter` → `Helvetica Neue` → `Arial` →
  `Noto Sans SC` → `DejaVu Sans`). The CJK fallback (`Noto Sans SC`) ensures
  Indic/Hindi/CJK labels render correctly for projects like P7 & P14.
- **Layout**: `constrained_layout.use = True` so subplot spacing is computed
  automatically — do **not** combine with `tight_layout()` or `bbox_inches="tight"`.
- **Spines**: top & right spines hidden by default for a clean editorial look.
