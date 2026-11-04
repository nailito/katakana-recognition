# Machine learning methodology

[Back to the project overview](../README.md)

## Workflow and current status

1. **Explored the extracted data.** Inspected image dimensions, grayscale format, character labels, and sample counts. Displayed representative images with character labels and counts, and checked that all 46 basic modern Katakana are present. 
Loaded the seven Katakana metadata CSVs, associated each record with its PNG path and source file, and checked for duplicate paths and missing images.
2. **Recovered writer identifiers.** Read `Serial Sheet Number` from the binary source records and matched it to each metadata row. Verified 1,411 distinct writer identifiers with no missing values.
3. **Split the dataset by writer.** Sorted unique writer identifiers and shuffled them with NumPy's `default_rng`, using seed `12`. Assigned 70% of writers to training and 15% to validation, rounding both counts down; the remaining writers belong to the test set. Verified that no writers are shared between sets.

   | Set | Writers | Images |
   |---|---:|---:|
   | Training | 987 | 47,375 |
   | Validation | 211 | 10,128 |
   | Test | 213 | 10,223 |

4. **Encoded character labels.** Sorted the 48 unique characters and used the same character-to-integer mapping for all three sets, with labels from 0 to 47.
5. **Built the TensorFlow input pipeline.** Paired image paths with labels, decoded PNGs into one grayscale channel, and converted pixels to `float32` with `tf.image.convert_image_dtype`, dividing values by 255. Images retain their original shape of `(63, 64, 1)`. The pipeline batches examples and prefetches subsequent batches; shuffling is enabled only for training. Inspected a training batch to verify shapes, types, and pixel values. No additional rescaling is needed in the model.
6. **Trained and compared two architectures.** Evaluated saved Dense-128 and Dense-64 CNNs on the same validation split. Dense-128 reached 93.51% accuracy and loss 0.2439; Dense-64 reached 93.09% and loss 0.2517, with about half as many parameters. These initial single-seed results retain Dense-128 as the accuracy reference. See the [comparison report](../reports/dense_width_comparison.md).
7. **Separated comparisons from diagnostics.** Architecture definitions live in `src/models/`, with shared preparation, training, and evaluation in `src/experiments.py`. Notebook 04 compares global validation metrics and notebook 05 examines errors. Specific confusions, including チ/テ, are exploratory evidence rather than model-selection criteria. Historical notebooks remain archived. New training runs save their configuration, labels, history, and validation fingerprint.
8. **Compared dropout strengths.** Completed rates 0, 0.2, and 0.4 on the Dense-128 architecture with the common protocol and seed 12. Validation accuracy improved from 93.35% to 94.59% and 94.95%, respectively; validation loss decreased from 0.2458 to 0.2126 and 0.1878. Dropout 0.4 is the current candidate, with multi-seed confirmation still pending. See the [dropout study](../reports/dropout_study.md) for the analysis and ordered next steps.
9. **Prepared fixed-split repetitions.** Separated split and training seeds, made dropout randomness configurable, and added a nine-run plan to notebook 04. Evaluation and aggregation report available runs, means, sample standard deviations, and incomplete groups. The confirmation training has not yet been run.

### Why split by writer?

Each writer contributes multiple character images with a shared handwriting style. Splitting images independently could place samples from the same person in both training and evaluation sets. The resulting scores could be overly optimistic, because the model would already have encountered that person's writing style during training.

Keeping all images from a given `writer_id` in one set allows validation and test results to measure recognition of handwriting from people not seen during training. Writer identifiers organize the split; they are not input features for character recognition. The validation and test sets must also contain different writers from each other.

## Future investigations

- **Confirm regularization gains.** The runner now separates writer-split seed from training randomness. The prepared plan repeats rates 0, 0.2, and 0.4 across training seeds 12, 23, and 34 with split seed 12, keeping writers and labels fixed. Compare mean validation accuracy and loss and their variability. These repeated trials have not yet been run.

- **Context-assisted recognition.** Explore combining character prediction probabilities with a Japanese word lexicon or language model to resolve ambiguous Katakana within a word. For example, if four characters of a five-character word are recognized confidently, compare candidate readings for the remaining character using both visual evidence and lexical plausibility. Keep multiple character candidates rather than committing immediately to the highest-scoring prediction, and allow words absent from the lexicon, such as names and loanwords. This investigation has not yet been carried out.

- **Further hyperparameter variation.** Initial Dense-width and dropout-rate comparisons have been completed. Broaden comparisons to learning rates, batch sizes, convolutional filter counts, kernel sizes, dense-layer widths, L2 regularization, and label-preserving augmentation. Define the candidate ranges and comparison criteria before running experiments. Keep the writer split fixed, record each configuration and its validation results, and assess promising configurations across multiple training seeds. Reserve the test set for the final evaluation.
