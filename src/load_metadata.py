"""Load extracted ETL1 Katakana metadata without filtering source categories.

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


def load_metadata(etl1_dir: str | Path = DEFAULT_ETL1_DIR) -> pd.DataFrame:
    """Combine the seven metadata CSVs and associate each row with its PNG.

    All acquisition categories are retained, including historical characters
    and repeated categories. Default extraction CSVs do not contain writer IDs.
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

        metadata["source_file"] = source_file
        metadata["image_path"] = metadata["record_index"].map(
            lambda index: extracted_dir / f"{int(index):05d}.png"
        )
        tables.append(metadata)

    combined = pd.concat(tables, ignore_index=True)

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
    print(metadata.head().to_string(index=False))


if __name__ == "__main__":
    main()
