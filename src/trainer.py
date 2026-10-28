"""Split data, fit candidate models, and persist trained pipelines."""
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.base import clone
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from .feature_engineering import get_feature_columns
from .model_factory import create_models
from .preprocessing import build_preprocessor
from .utils import artifact_path


def split_dataset(frame, config):
    split = config["split"]
    seed = split["random_seed"]
    targets = config["dataset"]["target_columns"]
    features = get_feature_columns(frame, config)
    train, remainder = train_test_split(frame, test_size=1 - split["train"], random_state=seed)
    test_fraction = split["test"] / (split["validation"] + split["test"])
    validation, test = train_test_split(remainder, test_size=test_fraction, random_state=seed)
    return features, targets, train, validation, test


def train_models(frame, config, logger):
    features, targets, train, validation, test = split_dataset(frame, config)
    prep = build_preprocessor(features, train, config.get("preprocessing", {}))
    models = create_models(config)
    model_dir = Path(config["outputs"]["models"])
    model_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    for name, estimator in models.items():
        pipeline = Pipeline([("preprocessor", clone(prep)), ("model", estimator)])
        pipeline.fit(train[features], train[targets])
        joblib.dump(pipeline, model_dir / f"{name}.joblib")
        results[name] = {"validation_rmse": float(np.sqrt(mean_squared_error(
            validation[targets], pipeline.predict(validation[features]))))}
        logger.info("Trained %s; validation RMSE %.6g", name, results[name]["validation_rmse"])
    joblib.dump({"features": features, "targets": targets}, artifact_path(config, "schema.joblib"))
    joblib.dump(pipeline.named_steps["preprocessor"], artifact_path(config, "preprocessor.joblib"))
    (model_dir / "model_configurations.json").write_text(
        json.dumps({name: config["models"].get(name, config.get("neural_network", {})) for name in models}, indent=2),
        encoding="utf-8",
    )
    for name, part in (("train", train), ("validation", validation), ("test", test)):
        part.to_pickle(artifact_path(config, f"{name}.pkl"))
    (Path(config["outputs"]["metrics"]) / "validation_metrics.json").write_text(json.dumps(results, indent=2))
    return results
