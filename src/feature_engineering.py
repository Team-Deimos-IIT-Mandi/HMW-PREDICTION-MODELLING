"""Resolve configured or inferred model feature columns."""


def get_feature_columns(frame, config):
    dataset = config["dataset"]
    targets = dataset["target_columns"]
    if dataset.get("feature_columns") is not None:
        features = list(dataset["feature_columns"])
    else:
        excluded = set(targets) | set(dataset.get("drop_columns", []))
        features = [column for column in frame.columns if column not in excluded]
    if not features:
        raise ValueError("No input features remain. Configure feature_columns or adjust drop_columns.")
    overlap = set(features) & set(targets)
    if overlap:
        raise ValueError(f"Target columns cannot also be input features: {sorted(overlap)}")
    return features
