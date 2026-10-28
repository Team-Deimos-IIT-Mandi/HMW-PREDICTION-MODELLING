"""Build and fit reusable train-only preprocessing pipelines."""
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_preprocessor(features, frame, options):
    numeric = [name for name in features if frame[name].dtype.kind in "biufc"]
    categorical = [name for name in features if name not in numeric]
    transformers = []
    if numeric:
        steps = [("imputer", SimpleImputer(strategy=options.get("numeric_imputation", "median")))]
        if options.get("scale_numeric", True):
            steps.append(("scaler", StandardScaler()))
        transformers.append(("numeric", Pipeline(steps), numeric))
    if categorical:
        pipe = Pipeline([
            ("imputer", SimpleImputer(strategy=options.get("categorical_imputation", "most_frequent"))),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ])
        transformers.append(("categorical", pipe, categorical))
    return ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=False)
