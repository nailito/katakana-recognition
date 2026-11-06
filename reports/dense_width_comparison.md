# Dense-width experiment: initial validation comparison

Only the hidden Dense layer changes from 128 to 64 units. Both saved models are evaluated on the same 10,128 validation images using the writer split with seed 12. Evaluate the restored models with Keras `evaluate` and sparse categorical cross-entropy. The test set was not used for this comparison.

| Model | Parameters | Validation accuracy | Validation loss | Errors |
|---|---:|---:|---:|---:|
| Dense 128 | 6,677,040 | 93.51% | 0.2439 | 657 |
| Dense 64 | 3,347,952 | 93.09% | 0.2517 | 700 |

The smaller model reduces parameter count by about 49.9%, at the cost of 43 additional errors and a 0.42 percentage-point decrease in accuracy. Validation loss also increases. The 128-unit model remains the accuracy reference for this run; the smaller model is an efficiency trade-off rather than a validation improvement. This single-seed result does not establish a universal ranking of Dense widths.

The Dense-64 training history reaches its minimum validation loss at epoch 4, followed by three epochs without improvement. Training loss continues decreasing, so reducing width alone has not eliminated the observed generalization gap.

## Proposed next investigation

Compare ways of limiting overfitting: dropout, L2 regularization, or label-preserving image augmentation. Retain the 128-unit baseline and vary one approach at a time. Dropout directly tests regularization while retaining the hidden layer width, so it is a reasonable next experiment. Before training, define a small candidate range and compare all candidates under the same split, training budget, and stopping rule. Evaluate global validation accuracy and loss first, parameter cost second; use character confusions only to interpret results. Check promising settings across multiple seeds before drawing a stronger conclusion. No such experiment has been implemented or run here.
