# This module defines the CNN architecture for handwritten character
# recognition. It implements a Convolutional Neural Network using
# TensorFlow/Keras and exposes a build_model() factory function that
# constructs, compiles, prints a summary, and returns the model.
# build_model() accepts a num_classes parameter (default 10 for MNIST;
# pass 26 for EMNIST letters) so the same architecture serves both tasks.

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dense,
    Dropout,
)


def build_model(num_classes=10):
    """Build, compile, and return the CNN model for handwritten character recognition.

    Architecture:
        Input  → Conv2D(32, 3x3, relu) → MaxPool(2x2)
               → Conv2D(64, 3x3, relu) → MaxPool(2x2)
               → Flatten
               → Dense(128, relu) → Dropout(0.5)
               → Dense(num_classes, softmax)

    Args:
        num_classes (int): Number of output classes.
                           10  for MNIST digits  (default, backwards-compatible).
                           26  for EMNIST letters.

    Compiled with:
        optimizer : Adam
        loss      : categorical_crossentropy
        metrics   : accuracy

    Returns:
        model (tf.keras.Model): Compiled Keras model ready for training.
    """
    model_name = f"cnn_{num_classes}class"

    model = Sequential(
        [
            # ── Block 1 ───────────────────────────────────────────────────
            Conv2D(32, kernel_size=(3, 3), activation="relu",
                   input_shape=(28, 28, 1), name="conv2d_1"),
            MaxPooling2D(pool_size=(2, 2), name="maxpool_1"),

            # ── Block 2 ───────────────────────────────────────────────────
            Conv2D(64, kernel_size=(3, 3), activation="relu", name="conv2d_2"),
            MaxPooling2D(pool_size=(2, 2), name="maxpool_2"),

            # ── Classifier head ───────────────────────────────────────────
            Flatten(name="flatten"),
            Dense(128, activation="relu", name="dense_1"),
            Dropout(0.5, name="dropout"),
            Dense(num_classes, activation="softmax", name="output"),
        ],
        name=model_name,
    )

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    # Print architecture summary before returning
    model.summary()

    return model


def build_model_augmented(num_classes=10):
    """Build, compile, and return the CNN model for handwritten character recognition.
    This is identical to build_model(), intended to be trained with data augmentation.

    Args:
        num_classes (int): Number of output classes.
                           10  for MNIST digits.
                           26  for EMNIST letters.

    Returns:
        model (tf.keras.Model): Compiled Keras model ready for training.
    """
    model_name = f"cnn_{num_classes}class_aug"

    model = Sequential(
        [
            # ── Block 1 ───────────────────────────────────────────────────
            Conv2D(32, kernel_size=(3, 3), activation="relu",
                   input_shape=(28, 28, 1), name="conv2d_1"),
            MaxPooling2D(pool_size=(2, 2), name="maxpool_1"),

            # ── Block 2 ───────────────────────────────────────────────────
            Conv2D(64, kernel_size=(3, 3), activation="relu", name="conv2d_2"),
            MaxPooling2D(pool_size=(2, 2), name="maxpool_2"),

            # ── Classifier head ───────────────────────────────────────────
            Flatten(name="flatten"),
            Dense(128, activation="relu", name="dense_1"),
            Dropout(0.5, name="dropout"),
            Dense(num_classes, activation="softmax", name="output"),
        ],
        name=model_name,
    )

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    model.summary()

    return model
