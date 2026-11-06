# Final test evaluation

Frozen configuration: **dropout_multiseed_dropout40_split12_train12** (`model_key=dropout40`). Test images were evaluated once, after model selection on validation, and were not used for tuning.

## Provenance

| Field | Value |
|---|---|
| Artifact | `dropout_multiseed_dropout40_split12_train12.keras` (26.7 MB, weights kept local) |
| Architecture | [32, 64] filters, kernel 3, Dense 128, dropout 0.4 |
| Training | split seed 12, training seed 12, batch 32, Adam 0.001, 10 max epochs, patience 3 |
| Selected epoch | 5 of 8 |
| TensorFlow | 2.21.0 |
| Validation fingerprint | `2a14e042543cd60c58f41ba59e2297f7f66f39e8f2a33f4cc76ac0f259314897` |
| Test fingerprint | `03bce354e2c7b063f1ff4b2a8f0a4e51bf0193491c6d669bdafad5e629a8c003` |
| Split | writer seed 12; 47,375 train / 10,128 validation / 10,223 test images |

## Test metrics

| Metric | Value |
|---|---:|
| Test accuracy | **94.2874%** |
| 95% Wilson interval | 93.82% – 94.72% |
| Test loss | **0.2335** |
| Errors | 584 / 10,223 |

The same checkpoint scored 94.95% accuracy and 0.1878 loss on the 10,128 validation images used for early stopping. Validation remains the selection metric; this test result is reported once and does not justify further dropout tuning. Seed 12 is the checkpoint used in the main comparison tables; the other two 0.4 seeds were deliberately not evaluated on test, keeping this a single-shot evaluation.

## Weakest characters

| Character | Images | Errors | Recall |
|---|---:|---:|---:|
| チ | 213 | 33 | 84.5% |
| ヱ | 213 | 31 | 85.4% |
| ク | 213 | 25 | 88.3% |
| エ | 213 | 22 | 89.7% |
| ン | 213 | 22 | 89.7% |
| シ | 213 | 21 | 90.1% |
| ツ | 213 | 17 | 92.0% |
| セ | 213 | 16 | 92.5% |

## Most frequent confusions

| Actual → predicted | Images |
|---|---:|
| ヱ → エ | 21 |
| ン → シ | 12 |
| エ → ヱ | 9 |
| ク → フ | 9 |
| コ → ユ | 8 |
| ソ → ツ | 7 |
| チ → テ | 7 |
| ヘ → ハ | 7 |
| シ → ン | 6 |
| チ → ナ | 6 |
| ツ → ソ | 6 |
| テ → ラ | 6 |

The complete per-class table is in `final_test_per_class.csv`; all non-zero confusion pairs are in `final_test_confusions.csv`. These tables describe the single frozen evaluation and are not used for further model selection.

_Generated on 2026-11-06 by `python -m src.final_test`._
