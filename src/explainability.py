"""Lightweight model explanation helpers."""
import numpy as np


def feature_importances(pipeline, feature_names):
    model = pipeline.named_steps["model"]
    values = getattr(model, "feature_importances_", None)
    if values is None:
        return None
    names = pipeline.named_steps["preprocessor"].get_feature_names_out()
    order = np.argsort(values)[::-1]
    return [{"feature": str(names[i]), "importance": float(values[i])} for i in order]
