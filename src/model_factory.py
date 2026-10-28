"""Central registry for constructing enabled regression estimators."""
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.neural_network import MLPRegressor


REGISTRY = {
    "random_forest": RandomForestRegressor,
    "extra_trees": ExtraTreesRegressor,
    "gradient_boosting": GradientBoostingRegressor,
    "neural_network": MLPRegressor,
}


def create_models(config):
    models = {}
    seed = config["split"]["random_seed"]
    jobs = config.get("hardware", {}).get("n_jobs", -1)
    for name, options in config["models"].items():
        if not options.get("enabled", False):
            continue
        if name not in REGISTRY:
            raise ValueError(f"Unknown model '{name}'. Registered models: {sorted(REGISTRY)}")
        params = dict(options.get("params", {}))
        if name == "neural_network":
            nn = config.get("neural_network", {})
            params.setdefault("hidden_layer_sizes", tuple(nn.get("hidden_layers", [128, 64])))
            params.setdefault("max_iter", nn.get("max_iter", 500))
            params.setdefault("alpha", nn.get("alpha", 0.0001))
        if name in ("random_forest", "extra_trees"):
            params.setdefault("n_jobs", jobs)
        if "random_state" in REGISTRY[name]().get_params():
            params.setdefault("random_state", seed)
        models[name] = REGISTRY[name](**params)
    if config.get("neural_network", {}).get("enabled") and "neural_network" not in models:
        nn = config["neural_network"]
        models["neural_network"] = MLPRegressor(hidden_layer_sizes=tuple(nn.get("hidden_layers", [128, 64])),
                                                  max_iter=nn.get("max_iter", 500), alpha=nn.get("alpha", .0001),
                                                  random_state=seed)
    return models
