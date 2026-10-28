"""Regression metrics and comparison plots for three-output models."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def calculate_metrics(y_true, y_pred, targets):
    rows = {}
    for index, name in enumerate(targets):
        actual = np.asarray(y_true)[:, index]
        predicted = np.asarray(y_pred)[:, index]
        rows[name] = {
            "mae": float(mean_absolute_error(actual, predicted)),
            "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
            "r2": float(r2_score(actual, predicted)),
        }
    rows["average"] = {metric: float(np.mean([rows[t][metric] for t in targets])) for metric in ("mae", "rmse", "r2")}
    return rows


def save_metrics(metrics, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")


def save_comparison_plot(metrics_by_model, path):
    names = list(metrics_by_model)
    values = [metrics_by_model[name]["average"]["rmse"] for name in names]
    fig, ax = plt.subplots(figsize=(max(6, len(names) * 1.4), 4))
    ax.bar(names, values)
    ax.set_ylabel("Average RMSE")
    ax.set_title("Test set model comparison (lower is better)")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)
