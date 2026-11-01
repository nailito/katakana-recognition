# katakana-recognition

Exploration of the ETL1 handwritten Katakana dataset. See [data/README.md](data/README.md) for the data source and extraction instructions.

## Japanese characters in plots

To display Katakana labels in plots, install a Japanese font such as Noto Sans JP. Both notebooks use `find_japanese_font` from `src/fonts.py` to detect installed fonts and check that they contain the characters in the current metadata. Each notebook initializes its own font configuration. If none is compatible, images are displayed without character titles; sample counts remain visible where provided.

For Japanese titles, download [Noto Sans JP](https://fonts.google.com/noto/specimen/Noto+Sans+JP) and extract the font files:

- **Windows:** right-click a `.ttf` or `.otf` file and choose **Install**.
- **macOS:** open a font file in Font Book and choose **Install**.
- **Linux:** install through your desktop font manager, or place the files in `~/.local/share/fonts/` and refresh the cache with `fc-cache -f` when available.

Restart the notebook kernel after installation. Alternatively, set `JP_FONT_PATH` in the grid cell to a local `.ttf`, `.otf`, or `.ttc` file. This also supports downloaded fonts that have not been installed system-wide.

Existing compatible fonts such as Hiragino can be detected automatically. The notebook does not download or install fonts. Font selection only changes plot titles, not the image data.
