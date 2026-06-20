# This module orchestrates EMNIST letters model training. It loads the
# EMNIST/letters dataset via data_loader, builds a 26-class CNN via model.py,
# trains for 15 epochs, saves the model to models/emnist_cnn.h5, and prints
# final training and validation accuracy.

import os
import sys

# Allow running directly as `python src/train_emnist.py` from the project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data_loader import load_emnist
from src.model import build_model_augmented
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def train_emnist():
    """Load EMNIST letters data, train a 26-class CNN, save, and report accuracy.

    Training configuration:
        epochs           : 15
        batch_size       : 128
        validation_split : 0.1  (10% of training data used for validation)

    Saved artefact:
        models/emnist_cnn.h5

    Returns:
        history (tf.keras.callbacks.History): Keras History object containing
            per-epoch loss and accuracy values for training and validation.
    """
    # ── 1. Load & preprocess EMNIST letters ─────────────────────────────────
    print("[INFO] Loading EMNIST letters dataset …")
    x_train, y_train, x_test, y_test = load_emnist()
    print(f"       x_train: {x_train.shape}  |  x_test: {x_test.shape}\n")

    # ── 2. Build 26-class CNN ────────────────────────────────────────────────
    print("[INFO] Building 26-class augmented CNN model …")
    model = build_model_augmented(num_classes=26)
    print()

    # ── 3. Train ─────────────────────────────────────────────────────────────
    print("[INFO] Training with Data Augmentation for 15 epochs …")
    
    split = int(len(x_train) * 0.9)
    x_val, y_val = x_train[split:], y_train[split:]
    x_train, y_train = x_train[:split], y_train[:split]
    
    datagen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        shear_range=0.1,
        fill_mode='nearest'
    )

    history = model.fit(
        datagen.flow(x_train, y_train, batch_size=128),
        epochs=15,
        validation_data=(x_val, y_val),
        verbose=1
    )

    # ── 4. Save model ────────────────────────────────────────────────────────
    models_dir = os.path.join(os.path.dirname(__file__), "..", "models")
    os.makedirs(models_dir, exist_ok=True)
    save_path = os.path.join(models_dir, "emnist_cnn_aug.h5")
    model.save(save_path)
    print(f"\n[INFO] Model saved -> {os.path.abspath(save_path)}")

    # -- 5. Print final accuracy metrics ------------------------------------
    final_train_acc = history.history["accuracy"][-1]
    final_val_acc   = history.history["val_accuracy"][-1]
    print("\n-- Final Metrics (EMNIST Letters) ----------------------------------")
    print(f"  Training   accuracy : {final_train_acc * 100:.2f}%")
    print(f"  Validation accuracy : {final_val_acc   * 100:.2f}%")
    print("--------------------------------------------------------------------\n")

    return history


if __name__ == "__main__":
    train_emnist()
