"""Evaluate the frozen model once on the untouched ETL1 test split.

Run from the repository root:
    python -m src.final_test

The report is written once under reports/. Re-running is refused unless
--force is passed, because the test split must not be reused for model
selection.
"""

import argparse
import json
import math
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from .experiments import prepare_data, validation_fingerprint
from .input_pipeline import build_dataset
from .load_metadata import PROJECT_ROOT

FROZEN_RUN = "dropout_multiseed_dropout40_split12_train12"
SPLIT_SEED = 12
REPORT_STEM = "final_test_evaluation"


def wilson_interval(correct, total, z=1.96):
    """Return the Wilson score interval for a binomial proportion."""
    if total <= 0:
        raise ValueError("total must be positive")
    proportion = correct / total
    denominator = 1 + z**2 / total
    centre = (proportion + z**2 / (2 * total)) / denominator
    margin = (
        z
        * math.sqrt(proportion * (1 - proportion) / total + z**2 / (4 * total**2))
        / denominator
    )
    return max(0.0, centre - margin), min(1.0, centre + margin)


def load_frozen_model(run=FROZEN_RUN, split_seed=SPLIT_SEED):
    """Load the frozen artifact and verify its sidecars against the split."""
    path = PROJECT_ROOT / "models" / f"{run}.keras"
    if not path.is_file():
        raise FileNotFoundError(f"Frozen model artifact not found: {path}")
    frames, characters = prepare_data(split_seed=split_seed)
    config = json.loads(path.with_name(path.stem + "_config.json").read_text())
    classes = json.loads(path.with_name(path.stem + "_classes.json").read_text())
    if classes != characters:
        raise ValueError("Saved class order differs from the prepared split.")
    if config["validation_fingerprint"] != validation_fingerprint(
        frames["validation"]
    ):
        raise ValueError("Saved validation fingerprint differs from the prepared split.")
    model = tf.keras.models.load_model(path, compile=False)
    if model.output_shape[-1] != len(characters):
        raise ValueError("Saved model does not match the number of characters.")
    return model, frames, characters, config


def _per_class_table(labels, predictions, characters):
    """Return recall and error counts for every character."""
    total = np.bincount(labels, minlength=len(characters))
    correct = np.bincount(
        labels[predictions == labels], minlength=len(characters)
    )
    table = pd.DataFrame(
        {
            "Character": characters,
            "Images": total,
            "Correct": correct,
        }
    )
    table["Errors"] = table["Images"] - table["Correct"]
    table["Recall"] = table["Correct"] / table["Images"]
    return table.sort_values(["Recall", "Character"], kind="stable").reset_index(drop=True)


def _confusion_table(labels, predictions, characters):
    """Return non-zero misclassification pairs, most frequent first."""
    confusion = pd.crosstab(
        pd.Series([characters[index] for index in labels], name="Actual"),
        pd.Series([characters[index] for index in predictions], name="Predicted"),
    )
    pairs = (
        confusion.rename_axis(index="Actual", columns="Predicted")
        .stack()
        .rename("Count")
        .reset_index()
    )
    pairs = pairs.loc[(pairs["Actual"] != pairs["Predicted"]) & (pairs["Count"] > 0)]
    return pairs.sort_values(
        ["Count", "Actual"], ascending=[False, True]
    ).reset_index(drop=True)


def _write_report(path, run, config, source_path, frames, metrics,
                  interval, per_class, confusions, validation):
    """Write the single-use test report as Markdown."""
    test = frames["test"]
    train = frames["train"]
    validation_frame = frames["validation"]
    test_fingerprint = validation_fingerprint(test)
    top_confusions = confusions.head(12)
    weakest = per_class.head(8)
    architecture = config["architecture"]
    training = config["training"]
    date = datetime.now().astimezone().strftime("%Y-%m-%d")

    confusion_lines = "\n".join(
        f"| {row.Actual} → {row.Predicted} | {int(row.Count)} |"
        for row in top_confusions.itertuples()
    )
    weakest_lines = "\n".join(
        f"| {row.Character} | {row.Images} | {row.Errors} | {row.Recall:.1%} |"
        for row in weakest.itertuples()
    )

    path.write_text(
        f"""# Final test evaluation

Frozen configuration: **{run}** (`model_key={config["model_key"]}`). Test images were evaluated once, after model selection on validation, and were not used for tuning.

## Provenance

| Field | Value |
|---|---|
| Artifact | `{source_path.name}` ({source_path.stat().st_size / 1e6:.1f} MB, weights kept local) |
| Architecture | {architecture["filters"]} filters, kernel {architecture["kernel_size"]}, Dense {architecture["dense_units"]}, dropout {architecture["dropout"]} |
| Training | split seed {training["split_seed"]}, training seed {training["training_seed"]}, batch {training["batch_size"]}, Adam {training["learning_rate"]}, {training["max_epochs"]} max epochs, patience {training["patience"]} |
| Selected epoch | {config["best_epoch"]} of {len(json.loads(source_path.with_name(source_path.stem + "_history.json").read_text())["loss"])} |
| TensorFlow | {config["tensorflow_version"]} |
| Validation fingerprint | `{config["validation_fingerprint"]}` |
| Test fingerprint | `{test_fingerprint}` |
| Split | writer seed {config["training"]["split_seed"]}; {len(train):,} train / {len(validation_frame):,} validation / {len(test):,} test images |

## Test metrics

| Metric | Value |
|---|---:|
| Test accuracy | **{metrics["accuracy"]:.4%}** |
| 95% Wilson interval | {interval[0]:.2%} – {interval[1]:.2%} |
| Test loss | **{metrics["loss"]:.4f}** |
| Errors | {metrics["errors"]:,} / {metrics["total"]:,} |

The same checkpoint scored {validation["accuracy"]:.2%} accuracy and {validation["loss"]:.4f} loss on the {len(validation_frame):,} validation images used for early stopping. Validation remains the selection metric; this test result is reported once and does not justify further dropout tuning. Seed 12 is the checkpoint used in the main comparison tables; the other two 0.4 seeds were deliberately not evaluated on test, keeping this a single-shot evaluation.

## Weakest characters

| Character | Images | Errors | Recall |
|---|---:|---:|---:|
{weakest_lines}

## Most frequent confusions

| Actual → predicted | Images |
|---|---:|
{confusion_lines}

The complete per-class table is in `final_test_per_class.csv`; all non-zero confusion pairs are in `final_test_confusions.csv`. These tables describe the single frozen evaluation and are not used for further model selection.

_Generated on {date} by `python -m src.final_test`._
""",
        encoding="utf-8",
    )


def evaluate_frozen_test(run=FROZEN_RUN, split_seed=SPLIT_SEED, force=False):
    """Evaluate the frozen artifact once and write the reports."""
    reports = PROJECT_ROOT / "reports"
    report_path = reports / f"{REPORT_STEM}.md"
    if report_path.exists() and not force:
        raise FileExistsError(
            f"{report_path} already exists. The test split is single-use; "
            "pass --force only to regenerate the report."
        )
    model, frames, characters, config = load_frozen_model(run, split_seed)
    test = frames["test"]
    dataset = build_dataset(test, batch_size=config["training"]["batch_size"])
    probabilities = model.predict(dataset, verbose=0)
    labels = test["label"].to_numpy()
    if len(probabilities) != len(labels):
        raise ValueError("Prediction count does not match the test frame.")

    predictions = probabilities.argmax(axis=1)
    correct = int((predictions == labels).sum())
    total = len(labels)
    accuracy = correct / total
    loss = float(
        -np.log(
            np.clip(probabilities[np.arange(total), labels], 1e-12, 1.0)
        ).mean()
    )

    history_path = PROJECT_ROOT / "models" / f"{run}_history.json"
    history = json.loads(history_path.read_text())
    best = config["best_epoch"] - 1
    validation = {
        "accuracy": float(history["val_accuracy"][best]),
        "loss": float(history["val_loss"][best]),
    }

    per_class = _per_class_table(labels, predictions, characters)
    confusions = _confusion_table(labels, predictions, characters)
    interval = wilson_interval(correct, total)

    reports.mkdir(exist_ok=True)
    source_path = PROJECT_ROOT / "models" / f"{run}.keras"
    metrics = {
        "run": run,
        "accuracy": accuracy,
        "loss": loss,
        "errors": total - correct,
        "total": total,
        "wilson_95_interval": list(interval),
        "validation_accuracy": validation["accuracy"],
        "validation_loss": validation["loss"],
        "test_fingerprint": validation_fingerprint(test),
        "validation_fingerprint": config["validation_fingerprint"],
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    }
    (reports / f"{REPORT_STEM}_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    per_class.to_csv(reports / f"{REPORT_STEM}_per_class.csv", index=False)
    confusions.to_csv(reports / f"{REPORT_STEM}_confusions.csv", index=False)
    _write_report(
        report_path, run, config, source_path, frames, metrics,
        interval, per_class, confusions, validation,
    )
    return report_path, metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate the report even if it already exists (single-use exception).",
    )
    args = parser.parse_args()
    report_path, metrics = evaluate_frozen_test(force=args.force)
    print(f"Wrote {report_path}")
    print(f"Test accuracy: {metrics['accuracy']:.4%}")
    print(f"Test loss: {metrics['loss']:.4f}")
    print(f"Errors: {metrics['errors']} / {metrics['total']}")


if __name__ == "__main__":
    main()
