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
| `dropout30` | `src/models/dropout30.py` | Dense 128, then dropout with rate 0.3 |
| `dropout40` | `src/models/dropout40.py` | Dense 128, then dropout with rate 0.4 |
| `dropout50` | `src/models/dropout50.py` | Dense 128, then dropout with rate 0.5 |

Each script exposes `NAME`, `ARTIFACT`, `CONFIG`, and `build_model(input_shape, num_classes, seed)`. The builder's seed controls dropout randomness where present; the runner sets the global training seed before building the network. These scripts define the network only. Shared data preparation and training are implemented once in `src/experiments.py`.

To add a model, create an architecture script, register it in `src/models/__init__.py`, and select its key in the central notebook. Keep other settings fixed when testing one architectural factor. L2 and augmentation variants have not yet been implemented.

## Dropout study

The first regularization study compares rates 0, 0.2, and 0.4 after the hidden Dense-128 activation. Rate 0 is the unchanged baseline. All three models have 6,677,040 trainable parameters. With seed 12, their initial trainable weights were checked to be identical. This initial study used dropout seed 12; new runs pass the configurable training seed to dropout. Dropout is active during training and disabled for validation and inference.

The retained rates compare no regularization with two increasing strengths; they do not exhaust possible rates. Learning rate, batch size, architecture, writer split, epoch budget, and stopping rule are fixed. A new baseline is trained under the common runner, since the older baseline used a preliminary dataset iteration. Rank configurations using restored-model validation accuracy and loss, then inspect learning curves. Do not infer improvement from lower training accuracy or a smaller train/validation gap alone. The test set remains unused.

To execute or resume this series from the repository root:

```bash
python -m src.run_experiments --models baseline dropout20 dropout40 --prefix dropout_study --seed 12
```

The series runner reuses completed runs only when their recorded settings and architecture match, and validates their labels and validation fingerprint during evaluation. It exports an incremental `reports/dropout_study_comparison.csv`. Use a different prefix for a fresh series; existing artifacts are never overwritten. The central notebook also exposes the three saved runs. See [the study report](../reports/dropout_study.md) for measured results and limitations.

All three initial runs and the nine-run confirmation below are complete. Dropout 0.4 leads the initial single-seed comparison at 94.95% validation accuracy and loss 0.1878, and the three-seed study at 95.04% mean accuracy and 0.1808 mean loss. Future training remains user-run.

## Fixed-split confirmation across training seeds

The central notebook contains a nine-run plan: three architectures (`baseline`, `dropout20`, `dropout40`) crossed with training seeds **12, 23, 34**, while the writer-split seed remains **12**. The training seeds are arbitrary predefined reproducibility choices. They are not selected based on their performance. Initialization, training shuffle, and dropout randomness vary; writer membership, labels, and all other settings remain fixed.

In notebook 04, run imports and common preparation, inspect the confirmation plan, then set **`RUN_MULTISEED_STUDY = True`**. Leave `RUN_TRAINING` and `RUN_DROPOUT_SERIES` false. This launches nine distinct runs under the prefix `dropout_multiseed`; it does not overwrite the first study. Interrupted unsaved runs restart from scratch and completed compatible runs are reused.

The same user-run experiment is available from the repository root:

```bash
python -m src.run_experiments --models baseline dropout20 dropout40 --prefix dropout_multiseed --split-seed 12 --training-seeds 12 23 34
```

To evaluate saved runs without training, use the notebook's confirmation-evaluation cell, or:

```bash
python -m src.run_experiments --models baseline dropout20 dropout40 --prefix dropout_multiseed --split-seed 12 --training-seeds 12 23 34 --evaluate-only
```

New filenames include both seeds: `<prefix>_<model>_split12_train23.keras`, with the usual sidecars. `reports/dropout_multiseed_runs.csv` contains individual results and `reports/dropout_multiseed_summary.csv` contains per-model means and sample standard deviations (`ddof=1`). Missing runs are listed explicitly; aggregate groups are marked incomplete until all three planned seeds exist. Standard deviation is unavailable for fewer than two runs and is not a confidence interval. The notebook plots individual scores and mean/std for available runs.

Do not rank configurations using incomplete groups or choose only the best seed. Confirm that accuracy/loss gains are consistent before finer dropout tuning or other interventions. The test set remains untouched. The legacy `run_series(..., seed=...)` and CLI `--seed` retain their original coupled behavior for historical single-seed runs; use the explicit confirmation API for fixed-split repetitions.

## New training runs

### Prepared dropout refinement

The completed three-seed study favored 0.4 over 0.2 and no dropout for each training seed. The next prepared comparison retains rates **0.3, 0.4, 0.5** and seeds **12, 23, 34**, with writer-split seed **12**. The three saved 0.4 runs are reused; only six new runs are required. The original epoch budget and stopping rule remain fixed.

In notebook 04, run common preparation, then the section **Refine dropout: 0.3, 0.4, and 0.5**. Enable only `RUN_DROPOUT_REFINEMENT`; all other training flags should remain false. Its evaluation cell never trains models and marks incomplete groups explicitly.

Artifact prefix `dropout_multiseed` keeps existing 0.4 paths unchanged. A separate `report_prefix="dropout_refinement"` exports `dropout_refinement_runs.csv` and `dropout_refinement_summary.csv`, preserving the original study's CSVs. No model is copied or overwritten. The code checks settings and provenance before accepting saved runs.

Equivalent user-run command:

```bash
python -m src.run_experiments --models dropout30 dropout40 dropout50 --prefix dropout_multiseed --report-prefix dropout_refinement --split-seed 12 --training-seeds 12 23 34
```

Append `--evaluate-only` to evaluate saved runs without training. The finer study is complete: 0.5 has the highest mean accuracy (95.15%), while 0.4 has the lowest mean loss (0.1808). Retain 0.4 as a practical working configuration and end dropout tuning given the small differences and user-reported 50-minute cost. See [the conclusion](../reports/dropout_refinement.md). Future experiments start with one exploratory run and an agreed compute budget; repeated seeds are reserved for worthwhile gains. Retain the test set for final evaluation.

### Run artifacts

The notebook uses an explicit `MODEL_KEY`, unique `RUN_NAME`, and `RUN_TRAINING` flag. Common defaults are split seed 12, training seed 12, batch size 32, Adam learning rate 0.001, at most 10 epochs, and EarlyStopping on validation loss with patience 3 and restored best weights.

Each run saves four files under `models/`:

- `<run_name>.keras`: model and weights selected by EarlyStopping.
- `<run_name>_history.json`: all recorded training and validation metrics.
- `<run_name>_classes.json`: ordered character labels.
- `<run_name>_config.json`: architecture, training settings, split fractions, TensorFlow version, selected epoch, and ordered validation fingerprint.

Set `split_seed` before preparing data and vary `training_seed` independently. The runner checks that the supplied data matches the split seed. It creates a fresh shuffled dataset immediately before fitting. The historical notebooks consumed a shuffled batch during a pretraining check; the shared runner does not. Consequently identical seeds do not guarantee identical historical batch orders. Historical configurations containing only `seed` are interpreted as using it for both stages. Their original files are unchanged.

## Validation comparison

The `artifacts` dictionary selects saved runs to compare. The notebook loads their saved weights, evaluates the same validation images with `model.evaluate`, and exports `reports/model_comparison.csv`. It displays accuracy, cross-entropy loss, total errors, parameter count, and provenance availability. Do not select a model using its last training epoch or a different epoch chosen separately for accuracy.

Saved class order and validation fingerprints are checked when present. Historical artifacts lack some sidecars; this is reported as a historical assumption, not silently treated as verified provenance. Their association with the seed-12 split and sorted character labels comes from the archived notebooks. The original model files remain unchanged.

The comparison does not train missing models, use the test set, or infer absent historical training curves. The current 128-unit baseline has no saved history sidecar, so its missing curve is reported explicitly. Preserve the test set for final evaluation after model selection.
