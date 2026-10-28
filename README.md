# HMW Prediction Modelling

Configurable, modular regression pipeline for predicting three wheel properties (for example, sigma, damping, and stiffness) from wheel geometry. The repository currently includes `wheel_params.csv` as a geometry example; it has no target columns, so it is not yet a trainable dataset. Add the supplied labeled dataset and update the three target names in the configuration before training.

## Installation

Use Python 3.10 or newer. From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Configure and train

Edit `config/config.yaml`:

```yaml
dataset:
  path: data/dataset.csv
  target_columns: [sigma, damping, stiffness] # exact column names in your file
  feature_columns: null                       # infer all non-target, non-dropped columns
  drop_columns: [case_id]
```

Place the dataset at the configured path, then run the complete workflow:

```bash
python main.py --config config/config.yaml
```

The command validates the data, writes a dataset summary and missing-value report, makes reproducible train/validation/test splits, fits preprocessing on training rows, trains each enabled model, optionally tunes configured models, evaluates candidates on the held-out test set, selects the lowest average RMSE model, and saves reports, plots, and artifacts. Regression is currently supported. Target columns must be numeric and complete. CSV, Parquet, and Excel files are supported.

Run an individual stage with:

```bash
python main.py --stage preprocess --config config/config.yaml
python main.py --stage train --config config/config.yaml
python main.py --stage tune --config config/config.yaml
python main.py --stage evaluate --config config/config.yaml
```

`preprocess` saves the fitted transformer and feature schema. `train` creates the split artifacts and model pipelines. `tune` requires training data to be loadable and uses the configured search spaces. `evaluate` requires completed training artifacts. The full `all` stage is the recommended one-command path.

## Inference

Provide a CSV containing the input feature columns (targets are not needed):

```bash
python predict.py --input data/new_data.csv --output predictions.csv
```

Use `--config` to select another configuration. Predictions are written in the configured target order. The saved feature schema controls column selection, and unknown categorical values are handled by the fitted encoder.

## Configuration guide

`config/config.yaml` contains dataset and target names, inferred or explicit features, problem type, split ratios and seed, preprocessing, model options, tuning spaces, cross-validation folds, neural-network settings, CPU parallelism, and output locations. A normal use case only needs the dataset path and three target names. The example defaults enable Random Forest, Extra Trees, and Gradient Boosting; optional tuning is off by default. Enable it with `tuning.enabled: true` and configure `param_distributions`. Neural network training can be enabled under `neural_network` and listed in `models`.

Artifacts are written under `outputs/`: serialized models and selected model in `models/`, metrics/reports in `metrics/`, comparison plots in `plots/`, execution logs in `logs/`, and split/preprocessing/schema files in `artifacts/`.

## Add a model

For a sklearn-compatible regressor, add its class to `REGISTRY` in `src/model_factory.py`. Add a matching section beneath `models` in `config/config.yaml` with `enabled` and estimator `params`. The training and evaluation code will then pick it up automatically. For an estimator needing special construction, add that small adapter in `create_models`; keep data transforms inside the sklearn pipeline so inference uses the same fitted preprocessing.

## Project modules

- `src/data_loader.py`: load and validate tabular input.
- `src/feature_engineering.py`: resolve configured or inferred input columns.
- `src/preprocessing.py`: build imputing, scaling, and categorical encoding transforms.
- `src/model_factory.py`: estimator registry and construction.
- `src/trainer.py`: deterministic splitting, fitting, and saved artifacts.
- `src/hyperparameter_tuning.py`: optional randomized search.
- `src/evaluation.py`: per-target and aggregate regression metrics and plot.
- `src/explainability.py`: tree feature importance export.
- `src/inference.py`: schema-aware prediction.
- `src/utils.py`: config validation, paths, logging, and seeds.
- `notebooks/experimentation.ipynb`: optional exploration; core functionality remains in `src/`.
