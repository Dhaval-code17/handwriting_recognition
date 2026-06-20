# This module handles loading, preprocessing, and augmenting the MNIST
# and EMNIST datasets for training and evaluation. It exposes helper
# functions to normalise pixel values, reshape inputs for CNN consumption,
# and one-hot encode labels.

import gzip
import os
import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical


def load_mnist():
    """Load and preprocess the MNIST dataset.

    Steps performed:
      1. Download / load MNIST via Keras built-in helper.
      2. Normalise pixel values from [0, 255] to [0.0, 1.0].
      3. Reshape images to (28, 28, 1) for CNN input (channels-last).
      4. One-hot encode labels into 10-class vectors.

    Returns:
        x_train (np.ndarray): Training images, shape (60000, 28, 28, 1), float32.
        y_train (np.ndarray): Training labels, shape (60000, 10), float32.
        x_test  (np.ndarray): Test images,     shape (10000, 28, 28, 1), float32.
        y_test  (np.ndarray): Test labels,     shape (10000, 10),  float32.
    """
    # ── 1. Load raw data ────────────────────────────────────────────────────
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # ── 2. Normalise pixel values to [0, 1] ─────────────────────────────────
    x_train = x_train.astype("float32") / 255.0
    x_test  = x_test.astype("float32")  / 255.0

    # ── 3. Reshape to (samples, 28, 28, 1) for CNN input ────────────────────
    x_train = x_train.reshape(-1, 28, 28, 1)
    x_test  = x_test.reshape(-1, 28, 28, 1)

    # ── 4. One-hot encode labels (10 classes: digits 0-9) ───────────────────
    y_train = to_categorical(y_train, num_classes=10)
    y_test  = to_categorical(y_test,  num_classes=10)

    return x_train, y_train, x_test, y_test


# ── IDX file reader (works with gzip-compressed idx files) ──────────────────
def _read_idx(filepath):
    """Read a gzip-compressed IDX binary file and return a numpy array.

    IDX byte-type codes (offset 2 in magic number):
        0x08 → uint8   (images and labels both use this)
    Dimensions encoded starting at byte 4, each as a big-endian uint32.

    Args:
        filepath (str): Absolute path to the gzip-compressed idx file.

    Returns:
        np.ndarray: Decoded array with the correct shape.
    """
    with gzip.open(filepath, 'rb') as f:
        data = f.read()

    # Byte 2 of the magic number encodes the number of dimensions:
    #   2 → 1-D (labels), 3 → 3-D (images)
    ndim = data[3]          # number of dimensions
    if ndim == 3:           # images: (num, rows, cols)
        num  = int.from_bytes(data[4:8],   'big')
        rows = int.from_bytes(data[8:12],  'big')
        cols = int.from_bytes(data[12:16], 'big')
        return np.frombuffer(data[16:], dtype=np.uint8).reshape(num, rows, cols)
    else:                   # labels: (num,)
        num  = int.from_bytes(data[4:8], 'big')
        return np.frombuffer(data[8:], dtype=np.uint8)


def load_emnist():
    """Load and preprocess the EMNIST/letters dataset from local gzip idx files.

    Reads pre-downloaded files from:
        C:\\Users\\Dhaval\\.cache\\emnist\\gzip\\

    Steps performed:
      1. Parse the four gzip-compressed IDX files directly (no network needed).
      2. Normalise pixel values from [0, 255] to [0.0, 1.0].
      3. Reshape images to (28, 28, 1) for CNN input (channels-last).
      4. Shift labels from 1-26 (a=1 … z=26) to 0-25 (a=0 … z=25).
      5. One-hot encode labels into 26-class vectors.
      6. Print shapes for confirmation.

    Returns:
        x_train (np.ndarray): Training images, shape (N_train, 28, 28, 1), float32.
        y_train (np.ndarray): Training labels, shape (N_train, 26), float32.
        x_test  (np.ndarray): Test images,     shape (N_test,  28, 28, 1), float32.
        y_test  (np.ndarray): Test labels,     shape (N_test,  26),  float32.
    """
    base_path = r"C:\Users\Dhaval\.cache\emnist\gzip"

    print("[INFO] Loading EMNIST letters dataset from local gzip files ...")
    print(f"       Source: {base_path}")

    # ── 1. Read the four IDX files ───────────────────────────────────────────
    x_train = _read_idx(os.path.join(base_path, "emnist-letters-train-images-idx3-ubyte.gz"))
    y_train = _read_idx(os.path.join(base_path, "emnist-letters-train-labels-idx1-ubyte.gz"))
    x_test  = _read_idx(os.path.join(base_path, "emnist-letters-test-images-idx3-ubyte.gz"))
    y_test  = _read_idx(os.path.join(base_path, "emnist-letters-test-labels-idx1-ubyte.gz"))

    # ── 2. Normalise pixel values to [0, 1] ─────────────────────────────────
    x_train = x_train.astype("float32") / 255.0
    x_test  = x_test.astype("float32")  / 255.0

    # ── 3. Reshape to (samples, 28, 28, 1) ──────────────────────────────────
    x_train = x_train.reshape(-1, 28, 28, 1)
    x_test  = x_test.reshape(-1, 28, 28, 1)

    # ── 4. Shift labels: EMNIST uses 1-26; convert to 0-25 ──────────────────
    y_train = y_train - 1
    y_test  = y_test  - 1

    # ── 5. One-hot encode labels (26 classes: letters a-z) ───────────────────
    y_train = to_categorical(y_train, num_classes=26)
    y_test  = to_categorical(y_test,  num_classes=26)

    # -- 6. Print shapes for confirmation ------------------------------------
    print("\n-- EMNIST Shapes ---------------------------------------------------")
    print(f"  x_train : {x_train.shape}")
    print(f"  y_train : {y_train.shape}")
    print(f"  x_test  : {x_test.shape}")
    print(f"  y_test  : {y_test.shape}")
    print("--------------------------------------------------------------------\n")

    return x_train, y_train, x_test, y_test
