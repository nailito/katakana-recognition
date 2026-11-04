# Dropout study: Dense-128 CNN

## Question and protocol

Can dropout improve validation recognition relative to the unregularized Dense-128 CNN? Compare rates 0, 0.2, and 0.4 after the hidden Dense activation. These are a limited candidate range, not established optimal values.

All variants retain the original two convolutions, one pooling layer, Dense 128, and 48 softmax outputs: 6,677,040 trainable parameters. Initial trainable weights are identical with seed 12. The dropout layers use a fixed internal seed of 12. All models use the same writer split, batch size 32, Adam learning rate 0.001, at most 10 epochs, and EarlyStopping on validation loss with patience 3 and restored best weights. A fresh baseline is trained with the common runner; historical models are not the controlled baseline for this study.

The training set contains 47,375 images and validation 10,128. No final-test predictions are computed. The study uses one training seed and one writer split, so results are exploratory. Any promising rate should be checked across additional training seeds while keeping the evaluation protocol explicit.

## Results

All three runs are complete and saved. The comparison uses the restored models evaluated on the same validation images, as recorded in `dropout_study_comparison.csv`. Configuration sidecars confirm matching validation fingerprints and training settings.

| Dropout rate | Validation accuracy | Validation loss | Errors | Selected epoch | Epochs trained |
|---|---:|---:|---:|---:|---:|
| 0 | 93.35% | 0.2458 | 674 | 3 | 6 |
| 0.2 | 94.59% | 0.2126 | 548 | 6 | 9 |
| 0.4 | 94.95% | 0.1878 | 511 | 5 | 8 |

## Interpretation

Dropout 0.4 is the strongest candidate in this run: it gains 1.61 percentage points of accuracy and makes 163 fewer errors than the common-protocol baseline, a 24.2% relative error reduction. It also improves validation cross-entropy. Dropout 0.2 improves both measures over the baseline, but ranks below 0.4. All models retain the same parameter count; dropout does not make this architecture smaller at inference.

The learning histories support exploring regularization: the lowest validation loss occurs later with dropout, although validation loss still worsens after each selected epoch. EarlyStopping remains useful. Training metrics with active dropout are not directly comparable to validation metrics with dropout disabled; a small or inverted accuracy gap alone does not demonstrate better generalization.

Use the selected epoch for each model. For example, dropout 0.2 reaches 94.94% accuracy at its last epoch, but the model restored by minimum validation loss has 94.59% accuracy. Comparing the last-epoch score would misrepresent the saved model.

These are exploratory results from one initialization/shuffling seed and one writer split. They support dropout 0.4 as the current candidate, not as a universally optimal rate. Historical Dense-128 results are contextual; the new baseline is the controlled comparator.

## Next steps

1. Preparation complete: the shared runner now separates the fixed writer-split seed from training randomness, including configurable dropout randomness. Existing single-seed artifacts remain unchanged.
2. Confirmation complete: all nine experiments for rates 0/0.2/0.4 and training seeds 12/23/34 use split seed 12. Mean accuracy is 93.67%, 94.68%, and 95.04%; sample standard deviation is 0.41, 0.11, and 0.12 percentage points. Mean validation loss is 0.2305, 0.1990, and 0.1808. Rate 0.4 leads both measures for every seed. These results support a consistent improvement across the tested initializations on this fixed writer split.
3. The next prepared comparison refines rates to 0.3/0.4/0.5 using the same seeds. It reuses the three 0.4 runs and requires six new trainings. See `dropout_refinement.md`; this finer study has not yet been trained. L2 and label-preserving augmentation remain separate future interventions. Keep global metrics primary; use character errors to generate or interpret hypotheses.
4. Freeze model selection before evaluating the untouched test set. Report final test results separately from validation results.

No additional training is launched by this analysis update. The user runs future training experiments locally before results are reviewed.
