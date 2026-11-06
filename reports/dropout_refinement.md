# Dropout refinement and conclusion

## Motivation

The completed 0/0.2/0.4 study used three training seeds on one fixed writer split. Mean validation accuracy was 93.67%, 94.68%, and 95.04%, respectively; mean validation loss was 0.2305, 0.1990, and 0.1808. Rate 0.4 led both measures for every seed. This motivates comparing neighboring rates, not assuming that 0.4 is optimal.

## Protocol

- Rates: 0.3, 0.4, 0.5, applied after the hidden Dense-128 activation.
- Training seeds: 12, 23, 34; writer-split seed: 12.
- Fixed architecture: two convolutions, one pooling layer, Dense 128, 48 softmax outputs, 6,677,040 parameters.
- Fixed training: batch size 32, Adam learning rate 0.001, at most 10 epochs, EarlyStopping on validation loss with patience 3 and restored best weights.
- Reuse the three completed 0.4 runs without selecting only their best seed. Train three 0.3 and three 0.5 runs, for six new trainings.
- Preserve existing artifact names under `dropout_multiseed`; export new tables under `dropout_refinement` so the first comparison remains intact.

The user completed the six new trainings. The existing three 0.4 runs were reused. The user reported approximately 50 minutes for this refinement.

## Evaluation and interpretation

Use the saved restored model from each run. Report every run and each rate's mean and sample standard deviation of validation accuracy and loss. Do not rank incomplete groups. Treat standard deviation as initialization/shuffle variability, not a confidence interval or evidence across different writer splits.

If rankings differ between accuracy and loss or between seeds, document that trade-off. Small differences may not support a firm choice with only three seeds. Finer tuning on the same validation set remains model selection; the untouched test set was reserved for the final frozen configuration. Character-level confusions remain exploratory diagnostics.

## Results

| Rate | Mean validation accuracy | Accuracy sample std (points) | Mean validation loss |
|---|---:|---:|---:|
| 0.3 | 95.10% | 0.05 | 0.1907 |
| 0.4 | 95.04% | 0.12 | 0.1808 |
| 0.5 | 95.15% | 0.19 | 0.1842 |

The source tables are `dropout_refinement_runs.csv` and `dropout_refinement_summary.csv`. Rate 0.5 gives the highest mean accuracy, whereas 0.4 gives the lowest mean loss. The 0.5 versus 0.4 accuracy difference is 0.11 percentage points, equivalent to about 11 validation images on average, and is not a consistent win for every seed. No statistical significance claim is made.

## Conclusion and retained configuration

Dropout provides a useful improvement for this architecture: the initial three-seed comparison increased mean accuracy from 93.67% without dropout to 95.04% at 0.4, while reducing mean loss from 0.2305 to 0.1808. The refinement adds only a small accuracy trade-off and no configuration dominates both criteria.

Retain **dropout 0.4** as the working configuration because it preserves the demonstrated improvement and has the best mean validation loss among the tested rates. This is a practical choice, not evidence that 0.4 maximizes accuracy or is universally optimal. End dropout tuning here; the limited marginal gain did not justify further compute for this learning project. Model selection was then frozen and the test split was evaluated once: **94.29%** accuracy and **0.2335** loss, with 584 errors on 10,223 images. See the [final test report](final_test_evaluation.md).

For future work, agree on a compute budget before training. Start with one exploratory run per motivated change and repeat only when the gain appears useful enough to justify confirmation. Prefer changes with a clear benefit in recognition or computational cost over finer rate searches. The user launches training; no further experiment is started by this conclusion.
