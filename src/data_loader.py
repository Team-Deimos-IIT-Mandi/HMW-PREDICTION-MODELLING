"""Load and validate tabular datasets."""
from pathlib import Path

import pandas as pd

from .utils import resolve_path


def load_dataset(config: dict, config_dir: str | Path = ".") -> pd.DataFrame:
    path = resolve_path(config["dataset"]["path"], config_dir)
    if not path.is_file():
        raise FileNotFoundError(f"Dataset not found: {path}. Set dataset.path in config/config.yaml.")
    suffix = path.suffix.lower()
    if suffix == ".csv":
        frame = pd.read_csv(path)
    elif suffix in (".parquet", ".pq"):
        frame = pd.read_parquet(path)
    elif suffix in (".xlsx", ".xls"):
        frame = pd.read_excel(path)
    else:
        raise ValueError(f"Unsupported dataset format '{suffix}'. Use CSV, Parquet, or Excel.")
    validate_dataset(frame, config)
    return frame


def validate_dataset(frame: pd.DataFrame, config: dict) -> None:
    if frame.empty:
        raise ValueError("Dataset contains no rows.")
    if frame.columns.duplicated().any():
        raise ValueError("Dataset contains duplicate column names.")
    targets = config["dataset"]["target_columns"]
    absent = [col for col in targets if col not in frame.columns]
    if absent:
        raise ValueError(f"Missing target columns: {absent}. Set their exact names in dataset.target_columns.")
    features = config["dataset"].get("feature_columns")
    if features is not None:
        absent = [col for col in features if col not in frame.columns]
        if absent:
            raise ValueError(f"Configured feature columns not found: {absent}")
    if frame[targets].isna().any().any():
        bad = frame[targets].columns[frame[targets].isna().any()].tolist()
        raise ValueError(f"Target columns contain missing values: {bad}. Remove or correct those rows.")
    non_numeric = [name for name in targets if not pd.api.types.is_numeric_dtype(frame[name])]
    if non_numeric:
        raise ValueError(f"Regression targets must be numeric: {non_numeric}")
    if len(frame) < 5:
        raise ValueError("At least five rows are required for train/validation/test splitting.")
