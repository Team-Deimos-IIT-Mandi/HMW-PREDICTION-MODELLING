# Pipeline usage guide

This guide walks through preparing data, configuring, training, evaluating, and using the saved model. Run commands from the repository root so paths such as `data/dataset.csv` and `outputs/` resolve as shown.

## 1. Install the dependencies

Python 3.10 or later is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows, activate the environment with `.venv\\Scripts\\activate`.

## 2. Prepare the labeled dataset

Put the labeled training dataset at the path in `config/config.yaml`. CSV is the simplest supported format; Parquet and Excel files are also accepted. Each row should represent one sample. The table must contain:

- Three complete numeric target columns, one per quantity to predict.
- One or more input feature columns.
- Optional identifier columns, such as `case_id`, that should be excluded from learning.

For example:

| OD_mm | tread_thk_mm | material | sigma | damping | stiffness |
|---:|---:|---|---:|---:|---:|
| 265 | 6.25 | A | 12.4 | 0.18 | 820 |

The example `wheel_params.csv` contains geometry inputs only; it does not include targets and therefore cannot be used to fit or evaluate the model yet.

## 3. Set up `config/config.yaml`

At minimum, set the dataset path and exact target column names:

```yaml
dataset:
  path: data/wheel_training.csv
  target_columns: [sigma, damping, stiffness]
  feature_columns: null
  drop_columns: [case_id]
  problem_type: regression
```

With `feature_columns: null`, every column except targets and `drop_columns` is used as an input. Set an explicit list if only selected columns should be used:

```yaml
  feature_columns: [OD_mm, tread_thk_mm, material]
```

Feature columns can be numeric or categorical. Missing feature values are imputed; target values cannot be missing. The three targets must be numeric because the implemented task is multi-output regression.

All data paths are relative to the project root for the standard `config/config.yaml` location. Output locations are also configured centrally under `outputs`.

## 4. Run the complete workflow

```bash
python main.py --config config/config.yaml
```

This command performs the following sequence:

1. Validates the configuration and dataset columns.
2. Writes dataset summary and missing-value reports.
3. Creates reproducible train, validation, and test partitions.
4. Fits imputers, numeric scaling, and categorical encoding using training rows.
5. Trains each enabled model.
6. Runs configured tuning when `tuning.enabled` is true.
7. Computes per-target and average MAE, RMSE, and R² on the test split.
8. Chooses the model with the lowest average test RMSE and saves it for inference.

The default split is 70% train, 15% validation, and 15% test. Change `split.random_seed` to establish a different repeatable split. For a very small dataset, a three-way split may leave too few rows in a partition; use more samples or adjust the ratios.

## 5. Run stages separately

```bash
python main.py --stage preprocess --config config/config.yaml
python main.py --stage train --config config/config.yaml
python main.py --stage tune --config config/config.yaml
python main.py --stage evaluate --config config/config.yaml
```

- `preprocess` saves a fitted transformer and schema from the training partition.
- `train` loads the dataset, creates all split artifacts, and trains enabled models.
- `tune` loads the dataset and searches enabled models that have a search space. It needs the estimator configuration and a valid dataset. With tuning disabled or no search space, it reports/skips tuning; evaluation also requires models from a prior `train` stage.
- `evaluate` loads the saved test split, schema, and model files. Run `train` first.

Running `all` is the easiest way to keep the artifacts in sync after changing the dataset or configuration. Individual commands report missing files or columns with corrective messages.

## 6. Enable models and tuning

Switch models on or off in the `models` section. For example:

```yaml
models:
  random_forest:
    enabled: true
    params:
      n_estimators: 300
      max_depth: null
  extra_trees:
    enabled: false
    params: {}
```

Available identifiers are `random_forest`, `extra_trees`, `gradient_boosting`, and `neural_network`. Neural network parameters are read from `neural_network` and may also be overridden under its model `params`.

Tuning is off by default. To enable randomized search for the Random Forest, for example:

```yaml
tuning:
  enabled: true
  method: randomized # or grid
  n_iter: 20
  scoring: neg_root_mean_squared_error
  param_distributions:
    random_forest:
      model__n_estimators: [100, 200, 400]
      model__max_depth: [null, 5, 10, 20]
```

Search parameter keys use sklearn pipeline notation: `model__` followed by the estimator parameter. `grid` evaluates each combination; `randomized` samples up to `n_iter` combinations. Cross-validation fold count is controlled by `cross_validation.folds`.

## 7. Predict from new samples

After a successful full run, make a CSV with the same input feature columns used during training. Target columns are not required. Extra columns are ignored.

```bash
python predict.py --input data/new_samples.csv --output outputs/predictions.csv
```

To use another configuration:

```bash
python predict.py --config config/config.yaml --input data/new_samples.csv --output outputs/predictions.csv
```

The saved schema determines feature selection and target output order. Missing required feature columns cause a clear error; unknown categories are accepted by the fitted encoder. The output CSV contains the three predicted target columns.

## 8. Find generated outputs

- `outputs/models/`: each fitted estimator pipeline, `best_model.joblib`, and selected model name.
- `outputs/metrics/`: dataset summary, missing-value report, validation scores, test metrics by model, comparison JSON, and tree feature importance when available.
- `outputs/plots/model_comparison.png`: model comparison by average test RMSE.
- `outputs/artifacts/`: data splits, feature schema, and fitted preprocessor.
- `outputs/logs/pipeline.log`: run log.

Model files include both preprocessing and estimator steps, so prediction applies the same fitted transformations. Keep the configuration, artifacts, and model files together when moving a trained model to another machine.

## 9. Add an estimator

For a compatible scikit-learn regressor:

1. Import the estimator class in `src/model_factory.py`.
2. Add a key-to-class entry in `REGISTRY`.
3. Add that key to `models` in the YAML with `enabled` and `params`.
4. If desired, add a tuning search space under `tuning.param_distributions`.

The estimator must support fitting against a two-dimensional target (three target columns) and return predictions with the same output shape. Keep custom input transformations in the sklearn preprocessing pipeline so training and inference remain consistent.

## Troubleshooting

- **Dataset not found:** check `dataset.path`, its relative location from the project root, and the file extension.
- **Missing target columns:** copy exact header names into `dataset.target_columns`.
- **Missing feature columns at prediction:** include all feature columns saved during training, or retrain with a revised `feature_columns` list.
- **No models enabled:** enable one or more entries in `models`.
- **Evaluation artifacts missing:** run `python main.py --stage train --config config/config.yaml` first.
- **Small partition or cross-validation errors:** increase the dataset size or reduce split/CV requirements.
- **Dependency import error:** activate the intended environment and reinstall `requirements.txt`.
