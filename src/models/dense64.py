"""Same convolutional architecture as baseline, with 64 hidden Dense units."""

from tensorflow import keras

NAME = "Dense 64"
ARTIFACT = "katakana_dense64"
CONFIG = {"filters": [32, 64], "kernel_size": 3, "dense_units": 64}


def build_model(input_shape=(63, 64, 1), num_classes=48, seed=12):
    layers = keras.layers
    return keras.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv2D(32, 3, activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, activation="relu"),
        layers.Flatten(),
        layers.Dense(64, activation="relu"),
        layers.Dense(num_classes, activation="softmax"),
    ], name="dense64")
