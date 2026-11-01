"""Find an installed font that can display the requested characters."""

from collections.abc import Iterable
from pathlib import Path

from matplotlib import font_manager, ft2font


def find_japanese_font(
    characters: Iterable[str],
    font_path: str | Path | None = None,
) -> font_manager.FontProperties | None:
    """Return a compatible font, or None when no candidate supports the text.

    An explicit font path takes precedence over automatic detection. No font
    is installed or downloaded. Callers decide how to display a missing font.
    """
    required = {ord(char) for text in characters for char in text}
    preferred_names = (
        "noto sans jp", "noto sans cjk jp", "hiragino",
        "yu gothic", "meiryo", "ipaexgothic", "applegothic",
    )

    if font_path is not None:
        candidates = [str(Path(font_path).expanduser())]
    else:
        fonts = sorted(
            font_manager.fontManager.ttflist,
            key=lambda font: (
                not any(name in font.name.lower() for name in preferred_names),
                font.name,
            ),
        )
        candidates = list(dict.fromkeys(font.fname for font in fonts))

    for candidate in candidates:
        try:
            supported = set(ft2font.FT2Font(candidate).get_charmap())
        except (OSError, RuntimeError):
            continue
        if required.issubset(supported):
            return font_manager.FontProperties(fname=candidate)

    return None
