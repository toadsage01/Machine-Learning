# Machine-Learning — Production-Grade ML Systems Monorepo

A 16-project suite covering the full ML engineering stack: from EDA
(`P1_autoinsight`) to diffusion models (`P16_diffusion_from_scratch`),
organized as a single Git monorepo with shared utilities, plot styling,
and consistent module conventions.

## Repository layout

```
Machine-Learning/
├── shared/                         # Shared plot styles & utilities
│   ├── plot_style.mplstyle         # Project-wide matplotlib theme
│   ├── __init__.py                 # apply_style() helper
│   └── README.md
│
├── ml-foundations-lab/            # Module A — foundational ML systems
│   ├── P1_autoinsight/            # ✅ Automated EDA + drift report generator
│   ├── P2_iris_production/        # ✅ Production ML pipeline (sklearn + SHAP + ONNX + FastAPI)
│   ├── P3_minigrad/               # ✅ NumPy optimization library (GD, Momentum, RMSProp, Adam)
│   ├── P4_titanic_twoships/      # ✅ Feature engineering across Titanic + Spaceship Titanic
│   └── P5_housing_geospatial/    # ✅ OSMnx geospatial housing + Quantile Regression
│
├── ml-applied-lab/                # Module B — applied ML systems
│   ├── P6_leaf_disease/           # ✅ ResNet50 vs EfficientNetV2 vs ViT + Grad-CAM
│   ├── P7_hinglish_sentiment/      # ✅ TF-IDF vs IndicBERT on code-mixed Hinglish
│   ├── P8_churn_survival/         # ✅ Churn + Kaplan-Meier + Cox PH + Uplift modeling
│   ├── P9_nse_forecasting/        # ✅ LightGBM vs Chronos / TimesFM zero-shot
│   └── P15_experiment_kit/        # ⏳ A/B testing (Frequentist, Bayesian, CUPED, Sequential)
│
├── dl-advanced-lab/               # Module C — advanced deep learning
│   ├── P10_nn_from_scratch/      # ✅ Reverse-mode autograd engine in NumPy
│   ├── P11_face_recognition_realtime/ # ✅ RetinaFace + ArcFace + FAISS pipeline
│   ├── P12_recsys_two_tower/    # ✅ Two-Tower retrieval + LightGBM ranking
│   └── P13_automl_pipeline/    # ✅ Optuna + MLflow AutoML pipeline
├── production-lab/                # Module E — production ML systems
│   └── P15_experiment_kit/        # ✅ A/B testing (Frequentist, Bayesian, CUPED, Sequential)
├── generative-lab/                # Module D — generative models
│   └── P14_indic_lm_from_scratch/ # ✅ NanoGPT-style decoder-only LM (~25-50M params)
├── nn-from-scratch/               # P10 (legacy path — redirects to dl-advanced-lab/)
├── face-recognition-realtime/     # P11 (legacy path — redirects to dl-advanced-lab/)
├── recsys-two-tower/             # P12 (legacy path — redirects to dl-advanced-lab/)
├── automl-pipeline/               # P13 (legacy path — redirects to dl-advanced-lab/)
├── indic-lm-from-scratch/         # P14 (legacy path — redirects to generative-lab/)
└── diffusion-from-scratch/        # P16 — DDPM + DDIM + classifier-free guidance
```

Legend: ✅ completed · ⏳ pending

## Conventions

Every project folder follows the same file layout:

| File              | Purpose                                                          |
|-------------------|------------------------------------------------------------------|
| `dataset.py`      | Data acquisition, ETL, type inference, PyTorch/Dataset loaders     |
| `model.py`        | Model architecture or core algorithm definition                  |
| `train.py`        | CLI entry-point (`argparse`-based, with structured logging)      |
| `requirements.txt`| Pinned dependencies                                              |
| `metadata.json`   | Machine-readable project metadata (id, tier, tags, tech_stack…)   |
| `README.md`       | System architecture, trade-offs, technical documentation         |
| `tests/`          | Smoke / unit tests                                               |
| `assets/`         | Hero images, diagrams, generated figures                         |

## Shared utilities

The `shared/` folder is imported by every project to enforce a single
visual identity across all matplotlib figures. Usage:

```python
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from shared import apply_style
apply_style()  # now every plt.* call uses the project theme
```

See `shared/README.md` for details on the color cycle, typography, and
layout rules.

## Execution model

Projects are built **sequentially, one at a time**, and committed to
`main` after each. The current status is reflected in the layout table
above (✅ / ⏳). To resume work on a pending project, follow the same
flow:

1. Create the folder + `requirements.txt` + `metadata.json` + `.gitignore`.
2. Implement `dataset.py` → `model.py` → `report.py` / `train.py`.
3. Write smoke tests that exercise every code path.
4. Generate the hero image and write the `README.md`.
5. Commit + push, wait for review before starting the next project.
