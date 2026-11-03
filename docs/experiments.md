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
| `dropout20` | `src/models/dropout20.py` | Dense 128, then dropout with rate 0.2 |
| `dropout40` | `src/models/dropout40.py` | Dense 128, then dropout with rate 0.4 |

Each script exposes `NAME`, `ARTIFACT`, `CONFIG`, and `build_model(input_shape, num_classes)`. These scripts define the network only. Shared data preparation and training are implemented once in `src/experiments.py`.

To add a model, create an architecture script, register it in `src/models/__init__.py`, and select its key in the central notebook. Keep other settings fixed when testing one architectural factor. L2 and augmentation variants have not yet been implemented.

## Dropout study

The first regularization study compares rates 0, 0.2, and 0.4 after the hidden Dense-128 activation. Rate 0 is the unchanged baseline. All three models have 6,677,040 trainable parameters. With seed 12, their initial trainable weights were checked to be identical. Dropout uses a fixed internal random seed of 12; the training seed still controls initialization and dataset shuffling. Dropout is active during training and disabled for validation and inference.

The retained rates compare no regularization with two increasing strengths; they do not exhaust possible rates. Learning rate, batch size, architecture, writer split, epoch budget, and stopping rule are fixed. A new baseline is trained under the common runner, since the older baseline used a preliminary dataset iteration. Rank configurations using restored-model validation accuracy and loss, then inspect learning curves. Do not infer improvement from lower training accuracy or a smaller train/validation gap alone. The test set remains unused.

To execute or resume this series from the repository root:

```bash
python -m src.run_experiments --models baseline dropout20 dropout40 --prefix dropout_study --seed 12
```

The series runner reuses completed runs only when their recorded settings and architecture match, and validates their labels and validation fingerprint during evaluation. It exports an incremental `reports/dropout_study_comparison.csv`. Use a different prefix for a fresh series; existing artifacts are never overwritten. The central notebook also exposes the three saved runs. See [the study report](../reports/dropout_study.md) for measured results and limitations.

All three initial runs are complete. Dropout 0.4 leads this single-seed comparison at 94.95% validation accuracy and loss 0.1878. Next, separate split and training seeds before repeating the candidate range across training seeds. The current `--seed` changes the writer split as well as training randomness; it cannot yet perform that fixed-split confirmation by itself. Future training remains user-run.

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
