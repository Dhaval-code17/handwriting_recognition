# This module exposes inference functions for handwritten character recognition:
#   - predict_image(image_path)          : loads a PNG/JPG, returns predicted digit.
#   - predict_from_array(img_array)      : numpy array -> predicted digit + confidence.
#   - predict_letter(image_path)         : loads a PNG/JPG, returns predicted letter.
#   - predict_letter_from_array(img_arr) : numpy array -> predicted letter + confidence.
# A test routine (test_predict) pulls 3 MNIST samples, saves them as PNGs,
# and verifies end-to-end prediction correctness.

import os
import sys
import numpy as np
import cv2

# Allow running directly as `python src/predict.py` from the project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import tensorflow as tf
from src.data_loader import load_mnist


# -- Model paths --------------------------------------------------------------
_MODEL_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "models", "mnist_cnn_aug.h5")
)
_EMNIST_MODEL_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "models", "emnist_cnn_aug.h5")
)
# Public aliases used inside the predict_*_from_array() functions
MNIST_MODEL_PATH  = _MODEL_PATH
EMNIST_MODEL_PATH = _EMNIST_MODEL_PATH


def _load_model():
    """Load and return the MNIST digit model. Raises FileNotFoundError if absent."""
    if not os.path.exists(_MODEL_PATH):
        raise FileNotFoundError(
            f"Trained model not found at: {_MODEL_PATH}\n"
            "Please run  python src/train.py  first."
        )
    return tf.keras.models.load_model(_MODEL_PATH)


def _load_emnist_model():
    """Load and return the EMNIST letter model. Raises FileNotFoundError if absent."""
    if not os.path.exists(_EMNIST_MODEL_PATH):
        raise FileNotFoundError(
            f"EMNIST model not found at: {_EMNIST_MODEL_PATH}\n"
            "Please run  python src/train_emnist.py  first."
        )
    return tf.keras.models.load_model(_EMNIST_MODEL_PATH)


# ── Helper: preprocess a (H, W) or (H, W, C) numpy array → (1, 28, 28, 1) ──
def _preprocess_array(img):
    """Convert any raw image array to the 4-D tensor the model expects.

    Steps:
        1. Grayscale conversion if the image has 3 or 4 channels.
        2. Resize to 28×28.
        3. Normalise pixels to [0, 1].
        4. Reshape to (1, 28, 28, 1).

    Args:
        img (np.ndarray): Raw image array, any number of channels.

    Returns:
        np.ndarray: Float32 array of shape (1, 28, 28, 1).
    """
    # Step 1 – grayscale
    if img.ndim == 3 and img.shape[2] in (3, 4):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Step 2 – resize
    img = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)

    # Step 3 – normalise
    img = img.astype("float32") / 255.0

    # Step 4 – reshape to (1, 28, 28, 1)
    img = img.reshape(1, 28, 28, 1)

    return img


# ── Function 1: predict from file path ──────────────────────────────────────
def predict_image(image_path):
    """Load an image from disk, preprocess it, and predict the handwritten digit.

    Args:
        image_path (str): Absolute or relative path to a PNG/JPG image file.

    Returns:
        int: Predicted digit (0–9).
    """
    # Load model
    model = _load_model()

    # Load image with OpenCV
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Image not found or unreadable: {image_path}")

    # Preprocess
    img_tensor = _preprocess_array(img)

    # Predict
    predictions  = model.predict(img_tensor, verbose=0)   # shape: (1, 10)
    predicted    = int(np.argmax(predictions, axis=1)[0])
    confidence   = float(predictions[0][predicted]) * 100

    print(f"  Predicted Digit : {predicted}")
    print(f"  Confidence      : {confidence:.2f}%")

    return predicted


# -- Function 2: predict digit from numpy array ------------------------------
def predict_from_array(img_array):
    """Run MNIST digit inference on a numpy image array from the canvas.

    Preprocessing pipeline:
      1. Grayscale conversion (if needed).
      2. Binary threshold (>50 -> 255).
      3. Bounding-box crop to the drawn digit.
      4. Uniform padding (25% of max side) on all sides.
      5. Resize to 28x28.
      6. Normalise to [0, 1] and reshape to (1, 28, 28, 1).
      7. Predict with the MNIST digit model.

    Args:
        img_array (np.ndarray): Raw image array, shape (H, W) or (H, W, C), uint8.

    Returns:
        tuple[int, float]: (predicted_digit, confidence_percentage)
    """
    # 1. Grayscale
    if len(img_array.shape) == 3:
        img = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
    else:
        img = img_array.copy()

    # 2. Threshold to binary
    _, img = cv2.threshold(img, 50, 255, cv2.THRESH_BINARY)

    # 2.5 Stroke thinning
    kernel = np.ones((3, 3), np.uint8)
    img = cv2.erode(img, kernel, iterations=2)

    # 3. Bounding-box crop
    coords = cv2.findNonZero(img)
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        img = img[y:y + h, x:x + w]

    # 4. Padding
    pad = max(img.shape[0], img.shape[1]) // 4
    img = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)

    # 5. Resize to 28x28
    img = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)

    # 6. Normalise and reshape
    img = img.astype("float32") / 255.0
    img = img.reshape(1, 28, 28, 1)

    # 7. Predict
    model = tf.keras.models.load_model(MNIST_MODEL_PATH)
    pred            = model.predict(img, verbose=0)
    predicted_digit = int(np.argmax(pred))
    confidence      = float(np.max(pred)) * 100

    print(f"  Predicted Digit : {predicted_digit}")
    print(f"  Confidence      : {confidence:.2f}%")

    return predicted_digit, confidence


# -- Letter prediction from file path ----------------------------------------
def predict_letter(image_path):
    """Load an image from disk, preprocess it, and predict the handwritten letter.

    Args:
        image_path (str): Absolute or relative path to a PNG/JPG image file.

    Returns:
        tuple[str, float]: (predicted_letter, confidence_percentage)
            predicted_letter is a lowercase letter a-z.
    """
    model = _load_emnist_model()

    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Image not found or unreadable: {image_path}")

    img_tensor  = _preprocess_array(img)

    # EMNIST images are stored transposed/rotated relative to normal drawing
    # orientation.  Fix: rotate 90 degrees clockwise, then flip horizontally.
    _img28      = img_tensor.reshape(28, 28)
    _img28      = np.rot90(_img28, k=3)    # 90 degrees clockwise
    _img28      = np.fliplr(_img28)        # flip left-right
    img_tensor  = _img28.reshape(1, 28, 28, 1)

    predictions = model.predict(img_tensor, verbose=0)   # shape: (1, 26)
    predicted   = int(np.argmax(predictions, axis=1)[0])
    confidence  = float(predictions[0][predicted]) * 100
    letter      = chr(predicted + ord('a'))

    print(f"  Predicted Letter : {letter}")
    print(f"  Confidence       : {confidence:.2f}%")

    return letter, confidence


def normalize_stroke_thickness(img, target_ratio=0.06):
    ys, xs = np.where(img > 0)
    if len(xs) == 0:
        return img
    bbox_w = xs.max() - xs.min() + 1
    bbox_h = ys.max() - ys.min() + 1
    max_dim = max(bbox_w, bbox_h)
    target_thickness = max(1, int(round(max_dim * target_ratio)))
    kernel = np.ones((target_thickness, target_thickness), np.uint8)
    return cv2.dilate(img, kernel, iterations=1)



# -- Letter prediction from numpy array ---------------------------------------
def predict_letter_from_array(img_array):
    """Run EMNIST letter inference on a numpy image array.

    Preprocessing pipeline that matches EMNIST training orientation:
      1. Grayscale conversion (if needed).
      2. Binary threshold (>50 -> 255).
      3. Bounding-box crop to the drawn character.
      4. 20%-padding around the crop.
      5. Resize to 28x28.
      6. EMNIST orientation fix: rot90(k=3) + fliplr.
      7. Normalise to [0, 1] and reshape to (1, 28, 28, 1).
      8. Predict with the EMNIST letter model.

    Args:
        img_array (np.ndarray): Raw image array, shape (H, W) or (H, W, C), uint8.

    Returns:
        tuple[str, float]: (predicted_letter, confidence_percentage)
            predicted_letter is a lowercase letter a-z.
    """
    # 0. Save debug image for letter mode (shape > 100 ensures we only save the full canvas)
    if img_array.shape[0] > 100:
        import os
        os.makedirs('data', exist_ok=True)
        cv2.imwrite('data/letter_char.png', img_array)

    # 1. Convert to grayscale if needed
    if len(img_array.shape) == 3:
        img = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
    else:
        img = img_array.copy()

    # 2. Threshold to binary
    _, img = cv2.threshold(img, 50, 255, cv2.THRESH_BINARY)

    # 2.5 Normalize stroke thickness dynamically based on bounding box
    img = normalize_stroke_thickness(img, target_ratio=0.06)

    # 3. Find bounding box of the drawn character and crop to it
    coords = cv2.findNonZero(img)
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        img = img[y:y + h, x:x + w]

    # 4. Add padding around the cropped character (25% of max side)
    pad = max(img.shape[0], img.shape[1]) // 4
    img = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)

    # 5. Resize to 28x28
    img = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)

    # 6. Apply EMNIST orientation fix
    img = np.rot90(img, k=3)   # 90 degrees clockwise
    img = np.fliplr(img)       # flip left-right

    # 7. Normalize and reshape
    img = img.astype("float32") / 255.0
    img = img.reshape(1, 28, 28, 1)

    # 8. Predict
    model = tf.keras.models.load_model(EMNIST_MODEL_PATH)
    pred = model.predict(img, verbose=0)
    predicted_idx = int(np.argmax(pred))
    confidence    = float(np.max(pred)) * 100
    predicted_letter = chr(predicted_idx + ord('a'))

    print(f"  Predicted Letter : {predicted_letter}")
    print(f"  Confidence       : {confidence:.2f}%")

    return predicted_letter, confidence


# -- Letter prediction from sentence crop ---------------------------------------
def predict_letter_from_crop(img_array):
    """Run EMNIST letter inference on a character crop from sentence mode.
    
    The input is expected to be a grayscale binary crop (white character on black background).
    This function avoids the rot90 and fliplr steps because sentence crops are already correctly oriented.
    """
    img = img_array.copy()

    # Find bounding box with cv2.findNonZero + cv2.boundingRect
    coords = cv2.findNonZero(img)
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        img = img[y:y + h, x:x + w]

    # Add padding: pad = max(w,h) // 3
    pad = max(img.shape[0], img.shape[1]) // 3
    img = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)

    # Resize to 28x28
    img = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)

    # Erode to thin strokes: kernel = np.ones((2,2)); cv2.erode iterations=1
    kernel = np.ones((2, 2), np.uint8)
    img = cv2.erode(img, kernel, iterations=1)

    # Normalize to 0-1
    img = img.astype("float32") / 255.0

    # Reshape to (1,28,28,1)
    img = img.reshape(1, 28, 28, 1)

    # Predict with EMNIST model
    model = tf.keras.models.load_model(EMNIST_MODEL_PATH)
    pred = model.predict(img, verbose=0)
    predicted_idx = int(np.argmax(pred))
    confidence    = float(np.max(pred)) * 100
    predicted_letter = chr(predicted_idx + ord('a'))

    print(f"  Predicted Letter : {predicted_letter}")
    print(f"  Confidence       : {confidence:.2f}%")

    return predicted_letter, confidence


# -- Test routine -------------------------------------------------------------
def test_predict():
    """Pull 3 random MNIST test samples, save as PNGs, and test predict_image().

    Saved files:
        data/sample_0.png
        data/sample_1.png
        data/sample_2.png

    For each sample, prints:
        True label vs Predicted label and whether they match.
    """
    print("=" * 52)
    print("  predict_image() — End-to-End Test")
    print("=" * 52)

    # Load MNIST to get real test images
    print("\n[INFO] Loading MNIST test data …")
    _, _, x_test, y_test = load_mnist()

    # Directory for sample PNGs
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
    os.makedirs(data_dir, exist_ok=True)

    # Pick 3 fixed indices (0, 1, 2) for reproducibility
    indices = [0, 1, 2]

    saved_paths  = []
    true_labels  = []

    # ── Save PNGs ────────────────────────────────────────────────────────────
    print("[INFO] Saving 3 MNIST samples as PNG …")
    for i, idx in enumerate(indices):
        # x_test is (10000, 28, 28, 1) float32 [0,1] → scale to uint8
        img_uint8  = (x_test[idx].reshape(28, 28) * 255).astype(np.uint8)
        save_path  = os.path.join(data_dir, f"sample_{i}.png")
        cv2.imwrite(save_path, img_uint8)
        saved_paths.append(save_path)
        true_labels.append(int(np.argmax(y_test[idx])))
        print(f"  Saved: {save_path}  (true digit: {true_labels[-1]})")

    # ── Predict each saved PNG ────────────────────────────────────────────────
    print("\n[INFO] Running predict_image() on each saved PNG …\n")
    for i, (path, true_label) in enumerate(zip(saved_paths, true_labels)):
        print(f"── Sample {i} ─────────────────────────────────────")
        predicted = predict_image(path)
        match     = "✓  CORRECT" if predicted == true_label else "✗  WRONG"
        print(f"  True Label      : {true_label}")
        print(f"  Result          : {match}\n")

    print("[OK] Test complete.\n")


if __name__ == "__main__":
    test_predict()
