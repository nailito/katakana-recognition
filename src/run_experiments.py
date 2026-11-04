"""Run a selected model series: python -m src.run_experiments --help."""

import argparse
import json

import pandas as pd

from .experiments import (
    compare_models, normalize_training, prepare_data, train_model,
)
from .load_metadata import PROJECT_ROOT
from .models import MODELS

STUDY_COLUMNS = [
    "Run", "Model key", "Model", "Split seed", "Training seed", "Parameters",
    "Validation accuracy", "Validation loss", "Validation errors",
    "Class mapping", "Split provenance",
]


def run_series(model_keys, prefix="dropout_study", seed=12):
    """Train missing runs and evaluate the series on validation only."""
    if not prefix or any(char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for char in prefix):
        raise ValueError("Use letters, numbers, underscores or hyphens for prefix.")
    settings = normalize_training({"seed": seed})
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
            if normalize_training(saved["training"]) != settings or saved["model_key"] != key or saved["architecture"] != MODELS[key].CONFIG:
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


def seed_study_plan(model_keys=("baseline", "dropout20", "dropout40"),
                    training_seeds=(12, 23, 34), split_seed=12,
                    prefix="dropout_multiseed"):
    """List independent training runs sharing one fixed writer split."""
    if not model_keys or not training_seeds:
        raise ValueError("Choose at least one model and training seed.")
    if len(set(model_keys)) != len(model_keys) or len(set(training_seeds)) != len(training_seeds):
        raise ValueError("Model keys and training seeds must be unique.")
    if not prefix or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in prefix):
        raise ValueError("Use letters, numbers, underscores or hyphens for prefix.")
    rows = []
    for training_seed in training_seeds:
        for key in model_keys:
            if key not in MODELS:
                raise ValueError(f"Unknown model: {key}")
            name = f"{prefix}_{key}_split{split_seed}_train{training_seed}"
            rows.append({
                "Run": name, "Model key": key, "Model": MODELS[key].NAME,
                "Split seed": split_seed, "Training seed": training_seed,
                "Artifact": PROJECT_ROOT / "models" / f"{name}.keras",
            })
    return pd.DataFrame(rows)


def _check_study_run(row, settings):
    """Reject incompatible or incomplete saved runs rather than silently reuse."""
    path = row["Artifact"]
    for suffix in ("_config.json", "_classes.json", "_history.json"):
        if not path.with_name(path.stem + suffix).is_file():
            raise ValueError(f"Incomplete saved run: {row['Run']}")
    saved = json.loads(path.with_name(path.stem + "_config.json").read_text())
    if (normalize_training(saved["training"]) != settings
            or saved["model_key"] != row["Model key"]
            or saved["architecture"] != MODELS[row["Model key"]].CONFIG):
        raise ValueError(f"Existing run settings differ: {row['Run']}")


def _study_row(row, validation, characters, batch_size):
    values = compare_models({row["Run"]: row["Artifact"]}, validation,
                            characters, batch_size=batch_size).reset_index().iloc[0].to_dict()
    values.update({name: row[name] for name in
                   ("Model key", "Model", "Split seed", "Training seed")})
    return values


def summarize_seed_study(results, plan):
    """Mean and sample standard deviation; explicitly identify unfinished groups."""
    rows = []
    for key in plan["Model key"].unique():
        observed = results.loc[results["Model key"].eq(key)] if not results.empty else pd.DataFrame()
        count = len(observed)
        expected = int(plan["Model key"].eq(key).sum())
        row = {"Model": MODELS[key].NAME, "Runs": count, "Expected runs": expected,
               "Complete": count == expected}
        for metric in ("Validation accuracy", "Validation loss"):
            row[metric + " mean"] = observed[metric].mean() if count else float("nan")
            row[metric + " std"] = observed[metric].std(ddof=1) if count > 1 else float("nan")
        rows.append(row)
    return pd.DataFrame(rows).set_index("Model")


def _save_study_tables(results, plan, prefix):
    reports = PROJECT_ROOT / "reports"
    reports.mkdir(exist_ok=True)
    summary = summarize_seed_study(results, plan)
    results.to_csv(reports / f"{prefix}_runs.csv", index=False)
    summary.to_csv(reports / f"{prefix}_summary.csv")
    return summary


def evaluate_seed_study(model_keys=("baseline", "dropout20", "dropout40"),
                        training_seeds=(12, 23, 34), split_seed=12,
                        prefix="dropout_multiseed"):
    """Evaluate available saved runs only; never start training."""
    plan = seed_study_plan(model_keys, training_seeds, split_seed, prefix)
    frames, characters = prepare_data(split_seed=split_seed)
    rows = []
    for _, row in plan.iterrows():
        if not row["Artifact"].is_file():
            print(f"Pending: {row['Run']}")
            continue
        settings = normalize_training({"split_seed": split_seed,
                                       "training_seed": int(row["Training seed"])})
        _check_study_run(row, settings)
        rows.append(_study_row(row, frames["validation"], characters, settings["batch_size"]))
    results = pd.DataFrame(rows, columns=STUDY_COLUMNS)
    summary = _save_study_tables(results, plan, prefix)
    return results, summary


def run_seed_study(model_keys=("baseline", "dropout20", "dropout40"),
                   training_seeds=(12, 23, 34), split_seed=12,
                   prefix="dropout_multiseed"):
    """Train missing study runs, keeping writer membership fixed for every run."""
    plan = seed_study_plan(model_keys, training_seeds, split_seed, prefix)
    frames, characters = prepare_data(split_seed=split_seed)
    rows = []
    for _, row in plan.iterrows():
        settings = normalize_training({"split_seed": split_seed,
                                       "training_seed": int(row["Training seed"])})
        if row["Artifact"].is_file():
            _check_study_run(row, settings)
            print(f"Reusing saved run: {row['Run']}", flush=True)
        else:
            print(f"Training {row['Run']}", flush=True)
            train_model(row["Model key"], frames, characters, row["Run"], settings=settings)
        rows.append(_study_row(row, frames["validation"], characters, settings["batch_size"]))
        results = pd.DataFrame(rows, columns=STUDY_COLUMNS)
        summary = _save_study_tables(results, plan, prefix)
        print(summary.to_string(), flush=True)
    return results, summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=MODELS,
                        default=["baseline", "dropout20", "dropout40"])
    parser.add_argument("--prefix")
    parser.add_argument("--seed", type=int,
                        help="Legacy single-series seed controlling split and training.")
    parser.add_argument("--training-seeds", type=int, nargs="+",
                        help="Enable a fixed-split study with these training seeds.")
    parser.add_argument("--split-seed", type=int)
    parser.add_argument("--evaluate-only", action="store_true",
                        help="Evaluate saved fixed-split study runs without training.")
    args = parser.parse_args()
    if args.training_seeds:
        if args.seed is not None:
            parser.error("Use --split-seed with --training-seeds, not legacy --seed.")
        runner = evaluate_seed_study if args.evaluate_only else run_seed_study
        runner(args.models, training_seeds=args.training_seeds,
               split_seed=12 if args.split_seed is None else args.split_seed,
               prefix=args.prefix or "dropout_multiseed")
    else:
        if args.evaluate_only:
            parser.error("--evaluate-only requires --training-seeds")
        if args.split_seed is not None:
            parser.error("--split-seed requires --training-seeds")
        run_series(args.models, prefix=args.prefix or "dropout_study",
                   seed=12 if args.seed is None else args.seed)
