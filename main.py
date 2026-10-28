"""Run the modular multi-target regression workflow."""
import argparse
import json
from pathlib import Path

import joblib
import pandas as pd

from src.data_loader import load_dataset
from src.evaluation import calculate_metrics, save_comparison_plot, save_metrics
from src.explainability import feature_importances
from src.hyperparameter_tuning import tune_models
from src.trainer import train_models
from src.utils import artifact_path, load_config, prepare_directories, set_seed, setup_logging


def evaluate(config, logger):
    test_path = artifact_path(config, "test.pkl")
    schema_path = artifact_path(config, "schema.joblib")
    if not test_path.is_file() or not schema_path.is_file():
        raise FileNotFoundError("Evaluation artifacts are missing. Run --stage train first.")
    test = pd.read_pickle(test_path)
    schema = joblib.load(artifact_path(config, "schema.joblib"))
    targets, features = schema["targets"], schema["features"]
    metrics_by_model = {}
    model_dir = Path(config["outputs"]["models"])
    for path in sorted(model_dir.glob("*.joblib")):
        if path.name == "best_model.joblib":
            continue
        model = joblib.load(path)
        metrics = calculate_metrics(test[targets].to_numpy(), model.predict(test[features]), targets)
        metrics_by_model[path.stem] = metrics
        save_metrics(metrics, Path(config["outputs"]["metrics"]) / f"{path.stem}_test_metrics.json")
        importance = feature_importances(model, features)
        if importance:
            (Path(config["outputs"]["metrics"]) / f"{path.stem}_feature_importance.json").write_text(json.dumps(importance, indent=2))
        logger.info("%s average test RMSE %.6g", path.stem, metrics["average"]["rmse"])
    if not metrics_by_model:
        raise RuntimeError("No trained model files found. Run the train stage first.")
    winner = min(metrics_by_model, key=lambda model: metrics_by_model[model]["average"]["rmse"])
    (model_dir / "best_model.txt").write_text(winner, encoding="utf-8")
    joblib.dump(joblib.load(model_dir / f"{winner}.joblib"), model_dir / "best_model.joblib")
    (Path(config["outputs"]["metrics"]) / "model_comparison.json").write_text(json.dumps(metrics_by_model, indent=2))
    save_comparison_plot(metrics_by_model, Path(config["outputs"]["plots"]) / "model_comparison.png")
    logger.info("Selected %s by lowest average test RMSE", winner)
    return winner


def run_stage(stage, config, config_dir):
    prepare_directories(config)
    logger = setup_logging(config)
    set_seed(config["split"]["random_seed"])
    if stage in ("all", "preprocess", "train", "tune"):
        frame = load_dataset(config, config_dir)
        logger.info("Loaded %d rows and %d columns", len(frame), len(frame.columns))
        if stage == "all":
            summary_dir = Path(config["outputs"]["metrics"])
            frame.describe(include="all").transpose().to_csv(summary_dir / "dataset_summary.csv")
            frame.isna().sum().rename("missing_values").to_csv(summary_dir / "missing_values.csv")
            logger.info("Dataset summary and missing-value report saved")
        if stage in ("all", "preprocess"):
            # Preprocessing is fitted as part of training to avoid data leakage.
            if stage == "preprocess":
                from src.trainer import split_dataset
                from src.preprocessing import build_preprocessor
                import joblib
                features, _, train, _, _ = split_dataset(frame, config)
                fitted = build_preprocessor(features, train, config.get("preprocessing", {})).fit(train[features])
                joblib.dump(fitted, artifact_path(config, "preprocessor.joblib"))
                joblib.dump({"features": features, "targets": config["dataset"]["target_columns"]}, artifact_path(config, "schema.joblib"))
                for name, part in (("train", train),):
                    part.to_pickle(artifact_path(config, f"{name}.pkl"))
                logger.info("Preprocessor and schema saved")
                return
        if stage in ("all", "train"):
            train_models(frame, config, logger)
        if stage in ("all", "tune"):
            tuned = tune_models(frame, config, logger)
            if tuned:
                (Path(config["outputs"]["metrics"]) / "tuning_results.json").write_text(json.dumps(tuned, indent=2))
        if stage in ("all", "train", "tune"):
            evaluate(config, logger)
    elif stage == "evaluate":
        evaluate(config, logger)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--stage", choices=("all", "preprocess", "train", "evaluate", "tune"), default="all")
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    config = load_config(config_path)
    # Dataset paths are project-root relative (e.g. data/dataset.csv).
    run_stage(args.stage, config, config_path.parent.parent if config_path.parent.name == "config" else config_path.parent)


if __name__ == "__main__":
    main()
