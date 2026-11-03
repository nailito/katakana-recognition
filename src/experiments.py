"""Shared preparation, training, and validation for model comparisons."""

import hashlib
import json
from pathlib import Path

import pandas as pd
import tensorflow as tf

from .input_pipeline import build_dataset
from .load_metadata import PROJECT_ROOT, load_metadata
from .models import MODELS
from .split_dataset import split

DEFAULT_TRAINING = {
    "seed": 12,
    "batch_size": 32,
    "learning_rate": 0.001,
    "max_epochs": 10,
    "patience": 3,
}


def prepare_data(seed=12):
    """Return independently usable splits and a shared ordered character list."""
    metadata = load_metadata()
    frames = dict(zip(("train", "validation", "test"), split(metadata, seed=seed)))
    characters = sorted(metadata["char"].unique())
    mapping = {char: index for index, char in enumerate(characters)}
    for frame in frames.values():
        frame["label"] = frame["char"].map(mapping)
    return frames, characters


def validation_fingerprint(frame):
    """Identify the ordered validation records independently of the root path."""
    records = [
        [path.parent.name, path.name, int(writer), int(label)]
        for path, writer, label in zip(frame.image_path, frame.writer_id, frame.label)
    ]
    return hashlib.sha256(json.dumps(records).encode()).hexdigest()


def train_model(model_key, frames, characters, run_name, settings=None,
                output_dir=PROJECT_ROOT / "models"):
    """Train a fresh model and save a separate run without overwriting artifacts."""
    config = {**DEFAULT_TRAINING, **(settings or {})}
    if not run_name or Path(run_name).name != run_name or run_name in {".", ".."}:
        raise ValueError("Use a nonempty filename stem for run_name.")
    output_dir = Path(output_dir)
    paths = {suffix: output_dir / f"{run_name}{suffix}" for suffix in (
        ".keras", "_history.json", "_classes.json", "_config.json",
    )}
    if any(path.exists() for path in paths.values()):
        raise FileExistsError("This run already exists. Choose a new run_name.")
    expected, expected_characters = prepare_data(config["seed"])
    if list(characters) != expected_characters or any(
        not frames[name].equals(expected[name]) for name in expected
    ):
        raise ValueError("Prepared data does not match the configured split seed.")

    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(config["seed"])
    definition = MODELS[model_key]
    model = definition.build_model(num_classes=len(characters))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config["learning_rate"]),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=["accuracy"],
    )
    train_ds = build_dataset(frames["train"], batch_size=config["batch_size"],
                             training=True, seed=config["seed"])
    val_ds = build_dataset(frames["validation"], batch_size=config["batch_size"])
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=config["patience"], restore_best_weights=True,
    )
    history = model.fit(train_ds, validation_data=val_ds, epochs=config["max_epochs"],
                        callbacks=[early_stop], shuffle=False, verbose=2)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save(paths[".keras"])
    metadata = {
        "model_key": model_key, "architecture": definition.CONFIG,
        "training": config, "tensorflow_version": tf.__version__,
        "validation_fingerprint": validation_fingerprint(frames["validation"]),
        "split_fractions": [0.7, 0.15, 0.15],
        "best_epoch": min(range(len(history.history["val_loss"])),
                          key=history.history["val_loss"].__getitem__) + 1,
    }
    for suffix, value in [("_history.json", history.history),
                          ("_classes.json", list(characters)), ("_config.json", metadata)]:
        paths[suffix].write_text(json.dumps(value, ensure_ascii=False, indent=2),
                                 encoding="utf-8")
    return model, history


def compare_models(artifacts, validation, characters, batch_size=32):
    """Evaluate saved models globally; never train or use the test split."""
    rows = []
    ds = build_dataset(validation, batch_size=batch_size, training=False)
    for run_name, artifact in artifacts.items():
        path = Path(artifact)
        classes_path = path.with_name(path.stem + "_classes.json")
        config_path = path.with_name(path.stem + "_config.json")
        if classes_path.exists():
            if json.loads(classes_path.read_text()) != list(characters):
                raise ValueError(f"Character mapping mismatch: {run_name}")
        if config_path.exists():
            config = json.loads(config_path.read_text())
            if config["validation_fingerprint"] != validation_fingerprint(validation):
                raise ValueError(f"Validation split mismatch: {run_name}")
        model = tf.keras.models.load_model(path, compile=False)
        if model.output_shape[-1] != len(characters):
            raise ValueError(f"Output class count mismatch: {run_name}")
        model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                       metrics=["accuracy"])
        metrics = model.evaluate(ds, verbose=0, return_dict=True)
        rows.append({
            "Run": run_name, "Parameters": model.count_params(),
            "Validation accuracy": metrics["accuracy"],
            "Validation loss": metrics["loss"],
            "Validation errors": round((1 - metrics["accuracy"]) * len(validation)),
            "Class mapping": "Saved" if classes_path.exists() else "Historical assumption",
            "Split provenance": "Recorded" if config_path.exists() else "Historical assumption",
        })
    return pd.DataFrame(rows).set_index("Run")
