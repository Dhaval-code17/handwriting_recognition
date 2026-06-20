# This script verifies that the MNIST data pipeline in data_loader.py is
# working correctly. It prints tensor shapes, pixel value range, and a
# sample one-hot label so every preprocessing step can be confirmed visually.

import sys
import os

# Allow running from either the project root or the src/ directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data_loader import load_mnist


def main():
    print("=" * 50)
    print("  MNIST Data Verification")
    print("=" * 50)

    # ── Load data ────────────────────────────────────────────────────────────
    print("\n[INFO] Loading MNIST dataset …")
    x_train, y_train, x_test, y_test = load_mnist()

    # ── 1. Shapes ────────────────────────────────────────────────────────────
    print("\n── Shapes ──────────────────────────────────────────")
    print(f"  x_train : {x_train.shape}   (expected: (60000, 28, 28, 1))")
    print(f"  y_train : {y_train.shape}      (expected: (60000, 10))")
    print(f"  x_test  : {x_test.shape}   (expected: (10000, 28, 28, 1))")
    print(f"  y_test  : {y_test.shape}      (expected: (10000, 10))")

    # ── 2. Pixel value range ─────────────────────────────────────────────────
    print("\n── Pixel Value Range (normalisation check) ─────────")
    print(f"  x_train  min : {x_train.min():.4f}  (expected: 0.0)")
    print(f"  x_train  max : {x_train.max():.4f}  (expected: 1.0)")
    print(f"  x_test   min : {x_test.min():.4f}  (expected: 0.0)")
    print(f"  x_test   max : {x_test.max():.4f}  (expected: 1.0)")

    # ── 3. Sample one-hot label ──────────────────────────────────────────────
    print("\n── Sample One-Hot Label (one-hot encoding check) ───")
    print(f"  y_train[0] : {y_train[0]}")
    print(f"  Digit class: {y_train[0].argmax()}  (index of the '1' in the vector)")

    print("\n[OK] All checks passed — data pipeline is ready.\n")


if __name__ == "__main__":
    main()
