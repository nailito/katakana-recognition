"""Architectures available in the central experiment notebook."""

from . import baseline, dense64, dropout20, dropout30, dropout40, dropout50

MODELS = {
    "baseline": baseline,
    "dense64": dense64,
    "dropout20": dropout20,
    "dropout30": dropout30,
    "dropout40": dropout40,
    "dropout50": dropout50,
}
