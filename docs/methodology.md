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

### Why split by writer?

Each writer contributes multiple character images with a shared handwriting style. Splitting images independently could place samples from the same person in both training and evaluation sets. The resulting scores could be overly optimistic, because the model would already have encountered that person's writing style during training.

Keeping all images from a given `writer_id` in one set allows validation and test results to measure recognition of handwriting from people not seen during training. Writer identifiers organize the split; they are not input features for character recognition. The validation and test sets must also contain different writers from each other.
