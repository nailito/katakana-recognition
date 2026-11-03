# Katakana recognition

A learning project in computer vision: exploring handwritten Katakana from ETL1 and building a character classifier step by step.

## Dataset overview

The current selection contains **67,726 grayscale images**, **48 characters**, and **1,411 writers**. Images are 64 pixels wide by 63 pixels high. One acquisition group is retained for each repeated character; historical `ヰ` and `ヱ` remain included.

## Current progress

- Image exploration and class-count inspection.
- Metadata loading and acquisition-group selection.
- Writer identifiers recovered from the original binary records.
- Reproducible training, validation, and test split with no shared writers.
- Shared integer labels for the 48 characters.
- TensorFlow input pipeline: grayscale decoding, normalization, batching, and prefetching. Only training data is shuffled.
- Two trained CNN architectures: the 128-unit Dense baseline and a 64-unit variant.
- A central notebook for controlled model experiments and global validation comparisons.
- A separate notebook for exploratory character confusions and visual error analysis.

Initial saved models reach **93.51%** validation accuracy for Dense 128 and **93.09%** for Dense 64 on the same 10,128 validation images. These are single-seed validation results, not final test scores. The test set remains unused.

## Documentation

- [Machine learning methodology](docs/methodology.md): completed steps and the reasoning behind dataset preparation.
- [Model experiments](docs/experiments.md): architecture scripts, central notebook, saved runs, and comparison protocol.
- [Setup guide](docs/setup.md): Python environment, notebook kernel, and Japanese fonts.
- [Data guide](data/README.md): ETL1 source, terms of use, extraction, and file structure.

## Project structure

```text
data/        Data documentation; generated data stays local
docs/        Methodology and setup guides
notebooks/   Exploration, input pipeline, central model comparison, error analysis
notebooks/archive/ Historical training notebooks and their saved outputs
models/      Saved model artifacts and run metadata
reports/     Figures and analysis outputs
src/         Reusable Python functions
src/models/  One architecture definition per Python script
```
