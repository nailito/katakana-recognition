"""Load ETL1 Katakana metadata, excluding three repeated acquisition groups.

Run from the repository root:
    python src/load_metadata.py

Use --etl1-dir to read an extraction stored elsewhere. Import load_metadata
to reuse the combined DataFrame in another script or notebook.
"""

import argparse
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ETL1_DIR = PROJECT_ROOT / "ETL1"
SOURCE_FILES = tuple(f"ETL1C_{number:02d}" for number in range(7, 14))
RECORD_SIZE = 2052
EXCLUDED_GROUPS = {
    "ETL1C_11": {"イ", "エ"},
    "ETL1C_12": {"ウ"},
}


def _read_writer_ids(source_path: Path) -> list[int]:
    """Read Serial Sheet Number (bytes 5-6, big-endian) from M-type records."""
    if not source_path.is_file():
        raise FileNotFoundError(
            f"Raw ETL1 file required to recover writer IDs: {source_path}"
        )
    if source_path.stat().st_size % RECORD_SIZE:
        raise ValueError(f"Incomplete M-type record in {source_path}")

    writer_ids = []
    with source_path.open("rb") as source:
        while record := source.read(RECORD_SIZE):
            if len(record) != RECORD_SIZE:
                raise ValueError(f"Incomplete M-type record in {source_path}")
            writer_ids.append(int.from_bytes(record[4:6], byteorder="big"))
    return writer_ids


def load_metadata(
    etl1_dir: str | Path = DEFAULT_ETL1_DIR,
) -> pd.DataFrame:
    """Combine the seven metadata CSVs and associate each row with its PNG.

    Keep I, E, and U from ETL1C_07 only. Historical characters remain included.
    writer_id is recovered from the corresponding raw binary record using
    source_file and record_index; default extraction CSVs omit this field.
    """
    etl1_dir = Path(etl1_dir).expanduser().absolute()
    tables = []

    for source_file in SOURCE_FILES:
        extracted_dir = etl1_dir / f"{source_file}_unpack"
        metadata_path = extracted_dir / "meta.csv"

        if not metadata_path.is_file():
            raise FileNotFoundError(f"Extracted metadata not found: {metadata_path}")

        metadata = pd.read_csv(
            metadata_path,
            index_col=0,
            keep_default_na=False,
            dtype={"char": "string", "unicode": "string"},
        )
        if not {"char", "unicode"}.issubset(metadata.columns):
            raise ValueError(f"Expected char and unicode columns in {metadata_path}")

        metadata = metadata.rename_axis("record_index").reset_index()
        if metadata["record_index"].duplicated().any():
            raise ValueError(f"Duplicate record indices in {metadata_path}")

        writer_ids = _read_writer_ids(etl1_dir / source_file)
        if len(metadata) != len(writer_ids):
            raise ValueError(
                f"Record count mismatch for {source_file}: "
                f"{len(metadata)} CSV rows, {len(writer_ids)} binary records."
            )
        # Map by the original index, not by the current CSV row order.
        metadata["writer_id"] = metadata["record_index"].map(
            pd.Series(writer_ids, dtype="int64")
        )
        if metadata["writer_id"].isna().any():
            raise ValueError(f"Invalid record indices in {metadata_path}")
        metadata["writer_id"] = metadata["writer_id"].astype("int64")

        metadata["source_file"] = source_file
        metadata["image_path"] = metadata["record_index"].map(
            lambda index: extracted_dir / f"{int(index):05d}.png"
        )
        tables.append(metadata)

    combined = pd.concat(tables, ignore_index=True)

    excluded = pd.Series(False, index=combined.index)
    for source_file, characters in EXCLUDED_GROUPS.items():
        excluded |= (
            combined["source_file"].eq(source_file)
            & combined["char"].isin(characters)
        )
    combined = combined.loc[~excluded].reset_index(drop=True)

    # Filenames repeat across source files; the full path identifies an image.
    if combined["image_path"].duplicated().any():
        raise ValueError("The combined metadata contains duplicate image paths.")

    missing_images = combined.loc[
        ~combined["image_path"].map(Path.is_file), "image_path"
    ]
    if not missing_images.empty:
        raise FileNotFoundError(
            f"{len(missing_images)} referenced PNGs are missing. "
            f"First missing file: {missing_images.iloc[0]}"
        )

    return combined


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--etl1-dir",
        type=Path,
        default=DEFAULT_ETL1_DIR,
        help="Directory containing ETL1C_07_unpack through ETL1C_13_unpack.",
    )
    args = parser.parse_args()

    metadata = load_metadata(args.etl1_dir)
    print(f"Loaded {len(metadata):,} records from {len(SOURCE_FILES)} source files.")
    print(f"Unique writers: {metadata['writer_id'].nunique():,}")
    print(metadata.head().to_string(index=False))


if __name__ == "__main__":
    main()
