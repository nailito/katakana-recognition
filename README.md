# Katakana recognition

A learning project in computer vision: exploring handwritten Katakana from ETL1 and building a character classifier step by step.

## Dataset overview

The current selection contains **67,726 grayscale images**, **48 characters**, and **1,411 writers**. Images are 64 pixels wide by 63 pixels high. One acquisition group is retained for each repeated character; historical `ヰ` and `ヱ` remain included.

## Current progress

- Image exploration and class-count inspection.
- Metadata loading and acquisition-group selection.
- Writer identifiers recovered from the original binary records.
- Next: create separate training, validation, and test sets by writer.

Model training and evaluation have not yet been implemented in this repository. No performance results are reported at this stage.

## Documentation

- [Machine learning methodology](docs/methodology.md): reasoning, dataset choices, and planned evaluation.
- [Setup guide](docs/setup.md): Python environment, notebook kernel, and Japanese fonts.
- [Data guide](data/README.md): ETL1 source, terms of use, extraction, and file structure.

## Project structure

```text
data/        Data documentation; generated data stays local
docs/        Methodology and setup guides
notebooks/   Exploration and dataset preparation
reports/     Figures and analysis outputs
src/         Reusable Python functions
```
