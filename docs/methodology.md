# Machine learning methodology

[Back to the project overview](../README.md)

## Workflow and current status

1. **Explored the extracted data.** Inspected image dimensions, grayscale format, character labels, and sample counts. Displayed representative images with character labels and counts, and checked that all 46 basic modern Katakana are present. 
Loaded the seven Katakana metadata CSVs, associated each record with its PNG path and source file, and checked for duplicate paths and missing images.
2. **Recovered writer identifiers.** Read `Serial Sheet Number` from the binary source records and matched it to each metadata row. Verified 1,411 distinct writer identifiers with no missing values.

### Why split by writer?

Each writer contributes multiple character images with a shared handwriting style. Splitting images independently could place samples from the same person in both training and evaluation sets. The resulting scores could be overly optimistic, because the model would already have encountered that person's writing style during training.

Keeping all images from a given `writer_id` in one set allows validation and test results to measure recognition of handwriting from people not seen during training. Writer identifiers organize the split; they are not input features for character recognition. The validation and test sets must also contain different writers from each other.
