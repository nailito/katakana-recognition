"""Run a selected model series: python -m src.run_experiments --help."""

import argparse
import json

from .experiments import DEFAULT_TRAINING, compare_models, prepare_data, train_model
from .load_metadata import PROJECT_ROOT
from .models import MODELS


def run_series(model_keys, prefix="dropout_study", seed=12):
    """Train missing runs and evaluate the series on validation only."""
    if not prefix or any(char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for char in prefix):
        raise ValueError("Use letters, numbers, underscores or hyphens for prefix.")
    settings = {**DEFAULT_TRAINING, "seed": seed}
    frames, characters = prepare_data(seed=seed)
    artifacts = {}
    for key in model_keys:
        run_name = f"{prefix}_{key}_seed{seed}"
        path = PROJECT_ROOT / "models" / f"{run_name}.keras"
        config_path = path.with_name(path.stem + "_config.json")
        if path.exists():
            if not config_path.exists():
                raise ValueError(f"Cannot resume a run without configuration: {run_name}")
            saved = json.loads(config_path.read_text())
            if saved["training"] != settings or saved["model_key"] != key or saved["architecture"] != MODELS[key].CONFIG:
                raise ValueError(f"Existing run settings differ: {run_name}")
            print(f"Reusing saved run: {run_name}", flush=True)
        else:
            print(f"Training {run_name}", flush=True)
            train_model(key, frames, characters, run_name, settings=settings)
        artifacts[MODELS[key].NAME] = path
        comparison = compare_models(artifacts, frames["validation"], characters,
                                     batch_size=settings["batch_size"])
        reports = PROJECT_ROOT / "reports"
        reports.mkdir(exist_ok=True)
        comparison.to_csv(reports / f"{prefix}_comparison.csv")
        print(comparison.to_string(), flush=True)
    return comparison


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=MODELS,
                        default=["baseline", "dropout20", "dropout40"])
    parser.add_argument("--prefix", default="dropout_study")
    parser.add_argument("--seed", type=int, default=12)
    args = parser.parse_args()
    run_series(args.models, prefix=args.prefix, seed=args.seed)
