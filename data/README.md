# Data: ETL1 Character Database

This project starts from the ETL1 Character Database, a collection of isolated handwritten characters. This document describes the source data and the files produced by the official extraction utility.

## Source and terms of use

Download ETL1 from the [official ETL Character Database website](https://etlcdb.db.aist.go.jp/download2/) after accepting the provider's terms of use.

The database is free to use, but redistribution is restricted. Raw ETL1 files and extracted images are kept locally and are not distributed through this repository. Any license applied to the project code is separate from the database's terms.

The provider requests the following reference:

> Electrotechnical Laboratory, Japanese Technical Committee for Optical Character Recognition, ETL Character Database, 1973-1984.

## What ETL1 contains

According to the [official dataset description](https://etlcdb.db.aist.go.jp/the-etl-character-database/), ETL1 contains:

- 141,319 handwritten character images from 1,445 writers.
- 99 categories: 10 digits, 26 uppercase Latin letters, 12 symbols, and 51 Katakana categories.
- Images measuring 64 pixels wide by 63 pixels high.
- Grayscale source images encoded with four bits per pixel.

The Katakana records are stored in files `ETL1C_07` through `ETL1C_13`. Together, these files contain 71,959 images from 1,411 writers.

The 51 Katakana acquisition categories do not correspond to 51 distinct modern characters: some characters occur in more than one category, and historical characters such as `ヰ` and `ヱ` are included. These source categories are preserved during extraction.

## Raw file format

The downloaded archive contains binary data files and an information file named `ETL1INFO`.

Each ETL1 image belongs to a fixed-length M-type record of 2,052 bytes. A record contains:

- Character identification codes.
- A serial sheet number associated with the writer's sheet.
- Quality indicators and other acquisition metadata.
- The image data, packed into 2,016 bytes.

The files are not ordinary image files and cannot be opened directly as PNGs. The provider supplies an extraction utility that converts the records into images and a metadata table.

See the [official M-type specification](https://etlcdb.db.aist.go.jp/etlcdb/etln/form_m.htm) for the complete record layout.

## Extracting the data locally

1. Download and extract ETL1 into a local directory named `ETL1/`.
2. Download `unpack_etlcdb.zip` from the official ETL download page and extract its contents into `unpack_etlcdb/`.
3. Install the utility's dependencies in your Python environment:

   ```bash
   python -m pip install -r unpack_etlcdb/requirements.txt
   ```

4. Run the utility from its own directory so it can find its supporting file, `euc_co59.dat`:

   ```bash
   cd unpack_etlcdb
   python unpack.py ../ETL1/ETL1C_07
   cd ..
   ```

Repeat the extraction for files `ETL1C_08` through `ETL1C_13` to obtain all Katakana records.

Do not pass `ETL1INFO` to the extraction utility: it describes the archive and does not contain image records.

## What the extracted data looks like

Each source file produces a corresponding directory:

```text
ETL1/
├── ETL1C_07
├── ETL1C_07_unpack/
│   ├── 00000.png
│   ├── 00001.png
│   ├── ...
│   └── meta.csv
├── ...
├── ETL1C_13
└── ETL1C_13_unpack/
    ├── 00000.png
    ├── ...
    └── meta.csv
```

### PNG images

Each PNG contains one handwritten character on a dark background, with lighter strokes.

- Width: 64 pixels.
- Height: 63 pixels.
- One grayscale channel.
- Numbered filename corresponding to its record index.

Filenames restart in each extraction directory. For example, `ETL1C_07_unpack/00000.png` and `ETL1C_08_unpack/00000.png` are different images. Use the directory and filename together to identify a sample.

The utility exports the source grayscale levels as multiples of 16 in an 8-bit PNG. The theoretical exported range is therefore 0 to 240; a particular image may use a smaller range.

### Metadata CSV

By default, `meta.csv` contains a record index and two fields:

| Field | Meaning | Example |
|---|---|---|
| Unnamed index column | Record index used in the image filename. | `0` corresponds to `00000.png`. |
| `char` | Decoded character label. | `ア` |
| `unicode` | Unicode code point written in hexadecimal. | `0x30a2` |

For example:

```csv
,char,unicode
0,ア,0x30a2
1,ア,0x30a2
```

The numeric filename is an image identifier, not a character label. Labels must be read from the corresponding metadata row.

Additional source metadata, including `Serial Sheet Number` and quality indicators, can be exported using the utility's `--fields` option. They are not included in the default two-field CSV.

At this stage, extraction preserves the source records. No class selection, train/validation/test assignment, or image transformation beyond the utility's decoding and PNG export has been applied.

## Acquisition groups retained for this project

To avoid giving `イ`, `エ`, and `ウ` twice as many examples as other characters, `src/load_metadata.py` keeps their records from `ETL1C_07` and excludes the additional groups: `イ` and `エ` from `ETL1C_11`, and `ウ` from `ETL1C_12`. All other records are retained, including `ヰ`, `ヱ`, and the slightly smaller classes `ナ` and `リ`.

The loaded dataset therefore contains **67,726 images across 48 distinct characters**: 1,411 examples per character, except `ナ` and `リ`, which have 1,410 each. This selection only filters the loaded metadata; the source CSVs and PNGs are unchanged.
