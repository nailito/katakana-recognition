# Upper-stroke orientation in training and validation samples

## Question and method

Are approximately horizontal upper strokes underrepresented in training examples of チ? Sample 60 images independently from each split using pandas sampling with seed 12. The training split contains 987 images of チ; validation contains 211. The assistant visually annotated the samples as Horizontal, Inclined, or Ambiguous without using model predictions. These subjective categories have no measured angle threshold and are provisional. The test set was not inspected and the model was not retrained.

## Observed sample composition

| Upper-stroke category | Training sample | Validation sample |
|---|---:|---:|
| Horizontal | 5/60 (8.3%) | 6/60 (10.0%) |
| Inclined | 52/60 (86.7%) | 47/60 (78.3%) |
| Ambiguous | 3/60 (5.0%) | 7/60 (11.7%) |

The horizontal form is uncommon in both samples. The difference of one horizontal example does not establish a meaningful train/validation imbalance. Sampling uncertainty and subjective annotation prevent a stronger conclusion or extrapolation to exact full-split counts.

In the previously annotated validation sample, the saved baseline confused チ with テ for 3/6 horizontal examples, 1/47 inclined examples, and 1/7 ambiguous examples. This supports investigating robustness to handwriting variation, but does not establish a causal role for orientation or prove a shortage of training examples caused the errors.

## Implication

There is no clear evidence here of a split-specific shortage of horizontal forms. Do not rebalance data or change the split based on these counts alone. A controlled model comparison, such as reducing dense-layer capacity while keeping the split and other settings fixed, can be evaluated using global validation metrics and the existing error subgroups. The small, already inspected subgroups remain diagnostic rather than independent evidence of generalization.

The two annotation CSVs preserve sample numbers, relative image paths, writer identifiers, and categories. They can be reviewed and revised without changing training labels.
