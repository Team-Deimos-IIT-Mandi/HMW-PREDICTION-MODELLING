"""Optional configuration-driven search using the validation split."""
from pathlib import Path

import joblib
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline

from .model_factory import create_models
from .preprocessing import build_preprocessor
from .trainer import split_dataset


def tune_models(frame, config, logger):
    settings = config.get("tuning", {})
    if not settings.get("enabled", False):
        logger.info("Hyperparameter tuning disabled; skipping.")
        return {}
    features, targets, train, validation, _ = split_dataset(frame, config)
    distributions = settings.get("param_distributions", {})
    output = {}
    for name, estimator in create_models(config).items():
        params = distributions.get(name)
        if not params:
            logger.info("No tuning search space configured for %s; skipping.", name)
            continue
        pipeline = Pipeline([("preprocessor", build_preprocessor(features, train, config.get("preprocessing", {}))),
                             ("model", estimator)])
        common = dict(estimator=pipeline, param_grid=params,
                      scoring=settings.get("scoring", "neg_root_mean_squared_error"),
                      cv=int(config.get("cross_validation", {}).get("folds", 5)), n_jobs=-1, refit=True)
        if settings.get("method", "randomized") == "grid":
            search = GridSearchCV(**common)
        elif settings.get("method", "randomized") == "randomized":
            common.pop("param_grid")
            search = RandomizedSearchCV(**common, param_distributions=params,
                                        n_iter=int(settings.get("n_iter", 20)),
                                        random_state=config["split"]["random_seed"])
        else:
            raise ValueError("tuning.method must be 'grid' or 'randomized'.")
        search.fit(train[features], train[targets])
        search.best_estimator_.fit(pd.concat([train, validation])[features], pd.concat([train, validation])[targets])
        joblib.dump(search.best_estimator_, Path(config["outputs"]["models"]) / f"{name}.joblib")
        output[name] = {"best_params": search.best_params_, "best_cv_score": float(search.best_score_)}
    return output


import pandas as pd
