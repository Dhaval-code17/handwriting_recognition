# This module evaluates a trained model on the MNIST test split. It loads
# a saved model from models/mnist_cnn.h5, runs model.evaluate() for loss and
# accuracy, generates full predictions, and reports a classification report,
# confusion matrix, and 5 sample predictions via sklearn and numpy.

import os
import sys
import numpy as np

# Allow running directly as `python src/evaluate.py` from the project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

from src.data_loader import load_mnist


def evaluate_model():
    """Load the saved CNN, evaluate on the MNIST test set, and report metrics.

    Steps performed:
        1. Load trained model from  models/mnist_cnn.h5
        2. Load MNIST test data via load_mnist()
        3. model.evaluate()  → test loss & accuracy
        4. model.predict()   → predicted probabilities
        5. argmax() on one-hot y_test and predictions → integer labels
        6. classification_report (precision / recall / f1 per digit)
        7. confusion_matrix
        8. 5 sample predictions with True / Predicted / Correct annotation
    """

    # ── 1. Load saved model ──────────────────────────────────────────────────
    model_path = os.path.join(os.path.dirname(__file__), "..", "models", "mnist_cnn.h5")
    model_path = os.path.abspath(model_path)

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Trained model not found at: {model_path}\n"
            "Please run  python src/train.py  first."
        )

    print(f"[INFO] Loading model from: {model_path}")
    model = tf.keras.models.load_model(model_path, compile=False)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # ── 2. Load test data ────────────────────────────────────────────────────
    print("[INFO] Loading MNIST test data …")
    _, _, x_test, y_test = load_mnist()          # only test split needed

    # ── 3. model.evaluate() → loss & accuracy ───────────────────────────────
    print("\n── Test Set Evaluation ─────────────────────────────")
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"  Test Loss     : {test_loss:.4f}")
    print(f"  Test Accuracy : {test_acc * 100:.2f}%")

    # ── 4. Predict all test samples ──────────────────────────────────────────
    print("\n[INFO] Running predictions on 10 000 test samples …")
    y_pred_probs = model.predict(x_test, verbose=0)   # shape: (10000, 10)

    # ── 5. Convert one-hot / probability arrays → integer digit labels ───────
    y_true = np.argmax(y_test, axis=1)       # (10000,)  e.g. [7, 2, 1, …]
    y_pred = np.argmax(y_pred_probs, axis=1) # (10000,)  e.g. [7, 2, 1, …]

    # ── 6. Classification report (precision / recall / f1 per digit) ─────────
    print("\n── Classification Report ───────────────────────────")
    target_names = [f"Digit {i}" for i in range(10)]
    print(classification_report(y_true, y_pred, target_names=target_names))

    # ── 7. Confusion matrix ──────────────────────────────────────────────────
    print("── Confusion Matrix ────────────────────────────────")
    cm = confusion_matrix(y_true, y_pred)
    # Print with digit headers for readability
    header = "      " + "  ".join(f"{i:3}" for i in range(10))
    print(header)
    print("     " + "─" * (len(header) - 5))
    for i, row in enumerate(cm):
        row_str = "  ".join(f"{v:3}" for v in row)
        print(f"  {i}  | {row_str}")
    print()

    # ── 8. 5 sample predictions ──────────────────────────────────────────────
    print("── 5 Sample Predictions ────────────────────────────")
    for i in range(5):
        true_label      = y_true[i]
        predicted_label = y_pred[i]
        correct         = "YES" if true_label == predicted_label else "NO"
        print(f"  Sample {i + 1}: True={true_label}, Predicted={predicted_label}, Correct={correct}")

    print("\n[OK] Evaluation complete.\n")


if __name__ == "__main__":
    evaluate_model()
