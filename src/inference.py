"""Load a saved best model and apply its stored feature schema."""
from pathlib import Path

import joblib
import pandas as pd

from .utils import artifact_path


def load_best_model(config):
    model_dir = Path(config["outputs"]["models"])
    best_path = model_dir / "best_model.txt"
    if not best_path.is_file():
        raise FileNotFoundError("No selected model found. Run `python main.py --config config/config.yaml` first.")
    name = best_path.read_text(encoding="utf-8").strip()
    model_path = model_dir / f"{name}.joblib"
    schema_path = artifact_path(config, "schema.joblib")
    if not model_path.is_file() or not schema_path.is_file():
        raise FileNotFoundError("Saved model or feature schema is missing. Run the training pipeline first.")
    return name, joblib.load(model_path), joblib.load(schema_path)


def predict_frame(frame, config):
    name, model, schema = load_best_model(config)
    missing = [col for col in schema["features"] if col not in frame]
    if missing:
        raise ValueError(f"Prediction input is missing required feature columns: {missing}")
    predictions = model.predict(frame[schema["features"]])
    return name, pd.DataFrame(predictions, columns=schema["targets"], index=frame.index)
