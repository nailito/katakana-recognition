# Setup guide

[Back to the project overview](../README.md)

## Python environment

Use Python 3.11. Run the following commands from the repository root.

On macOS or Linux:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

On Windows, using PowerShell and the Python launcher:

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
```

Once the environment is activated:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The current dependencies cover data exploration and notebook execution. TensorFlow is not installed at this stage. The virtual environment is excluded from Git.

## Notebook kernel

Open the repository folder in VS Code with Python and Jupyter notebook support installed. Open a notebook, choose **Select Kernel**, and select the Python interpreter inside the project's `.venv`.

To check the selected interpreter, run this in a notebook cell:

```python
import sys
print(sys.executable)
```

The path should point to `.venv/bin/python` on macOS/Linux or `.venv\Scripts\python.exe` on Windows.

Start with `notebooks/01_data_exploration.ipynb`, then continue with `notebooks/02_dataset_preparation.ipynb`.

## Local data

Follow the [data acquisition and extraction instructions](../data/README.md). The notebooks and loader expect an `ETL1/` directory at the repository root, containing the original binary files and the extracted `ETL1C_07_unpack/` through `ETL1C_13_unpack/` directories.

After extraction, check the loader from the repository root:

```bash
python src/load_metadata.py
```

It should report 67,726 records and 1,411 unique writers.

## Japanese characters in plots

To display Katakana labels in plots, install a Japanese font such as Noto Sans JP. Both notebooks use `find_japanese_font` from `src/fonts.py` to detect installed fonts and check that they contain the characters in the current metadata. Each notebook initializes its own font configuration. If none is compatible, images are displayed without character titles; sample counts remain visible where provided.

For Japanese titles, download [Noto Sans JP](https://fonts.google.com/noto/specimen/Noto+Sans+JP) and extract the font files:

- **Windows:** right-click a `.ttf` or `.otf` file and choose **Install**.
- **macOS:** open a font file in Font Book and choose **Install**.
- **Linux:** install through your desktop font manager, or place the files in `~/.local/share/fonts/` and refresh the cache with `fc-cache -f` when available.

Restart the notebook kernel after installation. Alternatively, set `JP_FONT_PATH` in the grid cell to a local `.ttf`, `.otf`, or `.ttc` file. This also supports downloaded fonts that have not been installed system-wide.

Existing compatible fonts such as Hiragino can be detected automatically. The notebook does not download or install fonts. Font selection only changes plot titles, not the image data.
