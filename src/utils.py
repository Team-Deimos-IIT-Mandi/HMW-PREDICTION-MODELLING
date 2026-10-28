"""Configuration, paths, logging, and reproducibility helpers."""
from __future__ import annotations

import logging
import random
from pathlib import Path

import numpy as np
import yaml


def load_config(path: str | Path) -> dict:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream) or {}
    required = ("dataset", "split", "models", "outputs")
    missing = [key for key in required if key not in config]
    if missing:
        raise ValueError(f"Configuration is missing sections: {', '.join(missing)}")
    ratios = config["split"]
    split_values = [float(ratios[k]) for k in ("train", "validation", "test")]
    if any(value <= 0 or value >= 1 for value in split_values):
        raise ValueError("Train, validation, and test ratios must each be between 0 and 1.")
    if abs(sum(split_values) - 1) > 1e-8:
        raise ValueError("Train, validation, and test ratios must sum to 1.")
    targets = config["dataset"].get("target_columns", [])
    if len(targets) != 3 or len(set(targets)) != 3:
        raise ValueError("dataset.target_columns must contain three unique column names.")
    if config["dataset"].get("problem_type", "regression") != "regression":
        raise ValueError("This pipeline currently supports regression only.")
    if not any(model.get("enabled", False) for model in config["models"].values()):
        raise ValueError("Enable at least one model in the models section.")
    return config


def resolve_path(path: str | Path, base: str | Path | None = None) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return (Path(base) / path).resolve() if base else path.resolve()


def prepare_directories(config: dict) -> None:
    for path in config["outputs"].values():
        Path(path).mkdir(parents=True, exist_ok=True)


def setup_logging(config: dict) -> logging.Logger:
    log_dir = Path(config["outputs"]["logs"])
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("ml_pipeline")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
        file_handler = logging.FileHandler(log_dir / "pipeline.log", encoding="utf-8")
        file_handler.setFormatter(formatter)
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)
    return logger


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def artifact_path(config: dict, name: str) -> Path:
    return Path(config["outputs"]["artifacts"]) / name
