"""Small interface checks for pipeline helpers."""
import pandas as pd

from src.data_loader import validate_dataset
from src.evaluation import calculate_metrics
from src.feature_engineering import get_feature_columns


def test_inferred_features_exclude_targets_and_drop_columns():
    frame = pd.DataFrame({"x": [1], "case_id": ["a"], "a": [2], "b": [3], "c": [4]})
    config = {"dataset": {"target_columns": ["a", "b", "c"], "drop_columns": ["case_id"]}}
    assert get_feature_columns(frame, config) == ["x"]


def test_three_target_metrics_include_average():
    values = pd.DataFrame([[1, 2, 3], [2, 3, 4]]).to_numpy()
    metrics = calculate_metrics(values, values, ["a", "b", "c"])
    assert metrics["average"]["rmse"] == 0
    assert set(metrics) == {"a", "b", "c", "average"}
