# Prepared dropout refinement

## Motivation

The completed 0/0.2/0.4 study used three training seeds on one fixed writer split. Mean validation accuracy was 93.67%, 94.68%, and 95.04%, respectively; mean validation loss was 0.2305, 0.1990, and 0.1808. Rate 0.4 led both measures for every seed. This motivates comparing neighboring rates, not assuming that 0.4 is optimal.

## Protocol

- Rates: 0.3, 0.4, 0.5, applied after the hidden Dense-128 activation.
- Training seeds: 12, 23, 34; writer-split seed: 12.
- Fixed architecture: two convolutions, one pooling layer, Dense 128, 48 softmax outputs, 6,677,040 parameters.
- Fixed training: batch size 32, Adam learning rate 0.001, at most 10 epochs, EarlyStopping on validation loss with patience 3 and restored best weights.
- Reuse the three completed 0.4 runs without selecting only their best seed. Train three 0.3 and three 0.5 runs, for six new trainings.
- Preserve existing artifact names under `dropout_multiseed`; export new tables under `dropout_refinement` so the first comparison remains intact.

The code is prepared and training is disabled by default. No refinement training was launched during preparation. The 0.3 and 0.5 results are pending.

## Evaluation and interpretation

Use the saved restored model from each run. Report every run and each rate's mean and sample standard deviation of validation accuracy and loss. Do not rank incomplete groups. Treat standard deviation as initialization/shuffle variability, not a confidence interval or evidence across different writer splits.

If rankings differ between accuracy and loss or between seeds, document that trade-off. Small differences may not support a firm choice with only three seeds. Finer tuning on the same validation set remains model selection; reserve the untouched test set for the final frozen configuration. Character-level confusions remain exploratory diagnostics.
