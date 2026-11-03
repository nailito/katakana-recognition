# Model experiments

[Back to the project overview](../README.md)

## Active workflow

Use `notebooks/04_model_comparison.ipynb` as the central workspace. Run common data preparation, inspect available architectures, optionally train one selected model, and compare explicitly selected saved artifacts. Training is disabled by default. Existing runs cannot be overwritten; choose a new run name for each experiment. Loading and evaluating saved models never requires rerunning training.

Use `notebooks/05_error_analysis.ipynb` separately for recall, confusion tables, image grids, and orientation annotations. Select its saved model explicitly. Diagnostic findings generate hypotheses; global validation accuracy and loss remain the main comparison criteria. Parameter count describes the size trade-off, not inference speed measured on hardware.

Earlier notebooks 04 and 05 are preserved in `notebooks/archive/` with their historical code and outputs. They are reference records rather than executable entry points in the active workflow.

## Architecture scripts

| Key | Script | Difference |
|---|---|---|
| `baseline` | `src/models/baseline.py` | Two convolutions, one pooling layer, Dense 128 |
| `dense64` | `src/models/dense64.py` | Same architecture, Dense 64 |

Each script exposes `NAME`, `ARTIFACT`, `CONFIG`, and `build_model(input_shape, num_classes)`. These scripts define the network only. Shared data preparation and training are implemented once in `src/experiments.py`.

To add a model, create an architecture script, register it in `src/models/__init__.py`, and select its key in the central notebook. Keep other settings fixed when testing one architectural factor. Additional dropout or augmentation variants have not yet been implemented.

## New training runs

The notebook uses an explicit `MODEL_KEY`, unique `RUN_NAME`, and `RUN_TRAINING` flag. Common defaults are seed 12, batch size 32, Adam learning rate 0.001, at most 10 epochs, and EarlyStopping on validation loss with patience 3 and restored best weights.

Each run saves four files under `models/`:

- `<run_name>.keras`: model and weights selected by EarlyStopping.
- `<run_name>_history.json`: all recorded training and validation metrics.
- `<run_name>_classes.json`: ordered character labels.
- `<run_name>_config.json`: architecture, training settings, split fractions, TensorFlow version, selected epoch, and ordered validation fingerprint.

Set the split seed before preparing data. The runner checks that the supplied data matches that seed. It creates a fresh shuffled dataset immediately before fitting. The historical notebooks consumed a shuffled batch during a pretraining check; the shared runner does not. Consequently identical seeds do not guarantee identical historical batch orders. Compare future variants under this common runner, and check promising settings across multiple seeds.

## Validation comparison

The `artifacts` dictionary selects saved runs to compare. The notebook loads their saved weights, evaluates the same validation images with `model.evaluate`, and exports `reports/model_comparison.csv`. It displays accuracy, cross-entropy loss, total errors, parameter count, and provenance availability. Do not select a model using its last training epoch or a different epoch chosen separately for accuracy.

Saved class order and validation fingerprints are checked when present. Historical artifacts lack some sidecars; this is reported as a historical assumption, not silently treated as verified provenance. Their association with the seed-12 split and sorted character labels comes from the archived notebooks. The original model files remain unchanged.

The comparison does not train missing models, use the test set, or infer absent historical training curves. The current 128-unit baseline has no saved history sidecar, so its missing curve is reported explicitly. Preserve the test set for final evaluation after model selection.
