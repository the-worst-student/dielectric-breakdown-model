# Dielectric Breakdown Model

A Python research project on electrical-tree growth in heterogeneous dielectrics. Numerical simulation, statistical analysis, and machine learning are used to study how material disorder affects discharge initiation, channel morphology, and breakdown probability.

## Technology stack

| Area | Technologies | Application |
|---|---|---|
| Language | **Python** | Simulation engine and research notebooks |
| Numerical computing | **NumPy, SciPy** | Grid operations, correlated random fields, Weibull transforms, Sobol sampling |
| Performance | **Numba, Joblib** | JIT-compiled solver kernels and parallel Monte Carlo runs |
| Data processing | **pandas, JSONL, CSV** | Simulation records, parameter plans, grouped aggregation, train/test tables |
| Statistics | **SciPy, statsmodels** | Bootstrap intervals, permutation and Friedman tests, Wilcoxon comparisons with Holm correction |
| Machine learning | **scikit-learn, CatBoost** | Linear and tree-based classifiers, grouped validation, hyperparameter search, calibration diagnostics |
| Analysis environment | **Matplotlib, Jupyter, tqdm** | Field maps, morphology plots, interactive experiments, progress monitoring |

## Physical model

- **Electrostatics** — finite-difference solution of the Laplace equation on a 2D grid, with Dirichlet conditions on electrodes and the conducting channel; iterative SOR solver accelerated with Numba
- **Material disorder** — spatially correlated Gaussian fields transformed into Weibull-distributed local breakdown thresholds, controlled by `mean_breakdown_field`, `correlation_length`, and `weibull_shape`
- **Channel growth** — stochastic selection of admissible frontier cells from the local electric field and breakdown threshold, with field sensitivity controlled by `eta`

Simulations distinguish three outcomes: `no_start` (no initiation), `arrested` (growth stops inside the sample), and `breakdown` (the channel reaches the target boundary). Separate material and growth seeds make individual realizations reproducible.

## Simulation and morphology

- **Monte Carlo sampling** — Joblib-parallelized runs, Sobol parameter sampling, adaptive refinement near outcome transitions, and incremental JSONL storage
- **Morphology** — box-counting fractal dimension, channel size, branch counts and density, mean free branch length, depth, width, sinuosity, lacunarity, and radial mass profiles
- **Channel capture** — controlled low-strength paths, repeated growth simulations, and logistic estimation of the 50% capture threshold

## Statistical analysis

Parameter studies are evaluated at the material-realization level to avoid treating repeated stochastic growth trajectories as independent observations. The implemented analysis includes bootstrap confidence intervals, permutation tests, Friedman repeated-measures tests, Wilcoxon post-hoc comparisons with Holm correction, and joint parameter-interaction maps.

Detailed experiments and numerical results are available in the notebooks for [disorder realizations](notebooks/07_disorder_seed_analysis.ipynb), [Weibull shape](notebooks/08_weibull_k_analysis.ipynb), [correlation length](notebooks/09_correlation_length_analysis.ipynb), [growth selectivity](notebooks/10_eta_analysis.ipynb), [breakdown threshold](notebooks/11_threshold_analysis.ipynb), and the [joint parameter sweep](notebooks/12_joint_parameter_sweep.ipynb).

## Machine learning

Models estimate the probabilities of `no_start`, `arrested`, and `breakdown` from `eta`, `mean_breakdown_field`, `correlation_length`, and `weibull_shape`. Dataset generation combines space-filling parameter sampling with adaptive refinement near outcome transitions and preserves an independent test split.

### Implemented models

| Model | Implementation and role |
|---|---|
| Dummy classifier | `strategy="prior"`; empirical class probabilities as a reference baseline |
| Logistic regression | `StandardScaler` followed by multinomial logistic regression; linear probabilistic baseline |
| Random Forest | Nonlinear tree ensemble with leaf-size regularization to control overfitting |
| Class-weighted Random Forest | Cost-sensitive forest used to study the rare-class precision–recall trade-off |
| CatBoost | Multiclass gradient boosting selected as the primary probabilistic model |

Model selection uses `StratifiedGroupKFold`, keeping repeated physical parameter combinations within the same fold. The evaluation covers multiclass log loss, balanced accuracy, per-class diagnostics, confusion matrices, permutation importance, reliability diagrams, and one- and two-parameter probability maps. Special one-vs-rest analysis is included for the rare `arrested` outcome.

Model configurations, validation scores, independent-test metrics, calibration plots, and rare-class diagnostics are reported in [`16_ml_baselines.ipynb`](notebooks/16_ml_baselines.ipynb).

## Repository

| Location | Contents |
|---|---|
| `src/` | Configuration, electrostatic solver, disorder generation, and channel growth |
| `notebooks/` | Numerical experiments, statistical studies, dataset generation, and ML evaluation |
| `data/ml_dataset/` | Parameter plans, simulation records, train/test tables, metadata, and images |

Dataset construction is implemented in [`15_dataset_generation.ipynb`](notebooks/15_dataset_generation.ipynb); model comparison and evaluation are in [`16_ml_baselines.ipynb`](notebooks/16_ml_baselines.ipynb).
