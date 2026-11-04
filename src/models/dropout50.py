"""Dense-128 baseline with 50% dropout before classification."""

from tensorflow import keras

NAME = "Dense 128 + dropout 0.5"
ARTIFACT = "katakana_dropout50"
CONFIG = {"filters": [32, 64], "kernel_size": 3, "dense_units": 128, "dropout": 0.5}


def build_model(input_shape=(63, 64, 1), num_classes=48, seed=12):
    layers = keras.layers
    return keras.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv2D(32, 3, activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, activation="relu"),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.5, seed=seed),
        layers.Dense(num_classes, activation="softmax"),
    ], name="dropout50")
