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
from PIL import Image

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
    """
    img = img_array.copy()

    # Find bounding box with cv2.findNonZero + cv2.boundingRect
    coords = cv2.findNonZero(img)
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        img = img[y:y + h, x:x + w]

    # Add padding: pad = max(w,h) // 4
    pad = max(img.shape[0], img.shape[1]) // 4
    img = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)

    # Resize to 28x28
    img = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)

    # EMNIST orientation fix (model was trained on transposed data)
    img = np.rot90(img, k=3)
    img = np.fliplr(img)

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


# -- Sentence prediction using Bounding Box Segmentation (Upgraded) ---------
def should_merge(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    
    area1 = w1 * h1
    area2 = w2 * h2
    
    smaller_area = min(area1, area2)
    larger_area = max(area1, area2)
    
    if smaller_area < larger_area * 0.15:  # one is much smaller (dot-like)
        x_overlap = min(x1+w1, x2+w2) - max(x1, x2)
        min_width = min(w1, w2)
        if x_overlap > min_width * 0.3:  # significant horizontal overlap
            return True
    
    return False

def predict_sentence_bbox(img_array):
    """Run Bounding Box segmentation and batch predict each letter.
    
    Args:
        img_array: Grayscale or RGBA numpy array from the canvas.
    """
    # 1. Preprocess: Get binary image (white text on black background)
    if len(img_array.shape) == 3 and img_array.shape[2] == 4:
        alpha = img_array[:, :, 3]
        gray = 255 - alpha
    elif len(img_array.shape) == 3:
        gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
    else:
        gray = img_array.copy()
        
    _, binary = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)
    
    # 2. Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 3. Get bounding boxes, filter noise
    boxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w * h > 100 and h > 10:
            boxes.append((x, y, w, h))
            
    if not boxes:
        return "", 0
        
    # 4. Sort left to right
    boxes.sort(key=lambda b: b[0])
    
    # 4.5 Compute median height to detect dot-like boxes
    heights = [h for x, y, w, h in boxes]
    median_h = sorted(heights)[len(heights) // 2]
    
    # 5. Merge boxes that are vertically aligned diacritics (i-dot, j-dot, accents)
    merged = []
    for box in boxes:
        if merged and should_merge(merged[-1], box):
            px, py, pw, ph = merged[-1]
            x, y, w, h = box
            new_x = min(px, x)
            new_y = min(py, y)
            new_x2 = max(px + pw, x + w)
            new_y2 = max(py + ph, y + h)
            merged[-1] = (new_x, new_y, new_x2 - new_x, new_y2 - new_y)
        else:
            merged.append(box)
            
    # 5.5 Clean up implausibly narrow stray strokes
    normal_widths = [w for (x, y, w, h) in merged if h >= median_h * 0.5]
    if not normal_widths:
        normal_widths = [w for (x, y, w, h) in merged]
    base_w = sorted(normal_widths)[len(normal_widths) // 2]
    
    cleaned_boxes = []
    for box in merged:
        x, y, w, h = box
        if h >= median_h * 0.5 and w < base_w * 0.25:
            if cleaned_boxes:
                px, py, pw, ph = cleaned_boxes[-1]
                if x - (px + pw) < 10:  # Very close
                    new_x = min(px, x)
                    new_y = min(py, y)
                    new_x2 = max(px + pw, x + w)
                    new_y2 = max(py + ph, y + h)
                    cleaned_boxes[-1] = (new_x, new_y, new_x2 - new_x, new_y2 - new_y)
                    continue
            cleaned_boxes.append(box)
        else:
            cleaned_boxes.append(box)
            
    final_boxes = cleaned_boxes
    
    # 6. Calculate median character width for space detection
    widths = [w for x, y, w, h in final_boxes]
    median_w = sorted(widths)[len(widths) // 2]
    
    # 7. Batch prediction
    model = tf.keras.models.load_model(EMNIST_MODEL_PATH)
    batch = []
    valid_indices = []
    
    for i, (x, y, w, h) in enumerate(final_boxes):
        char_img = binary[y:y+h, x:x+w]
        if char_img.size == 0 or np.count_nonzero(char_img) < 5:
            continue
            
        # 1. Stroke thickness normalization
        c = normalize_stroke_thickness(char_img, target_ratio=0.06)
            
        # 2. Make square to preserve aspect ratio
        h_crop, w_crop = c.shape
        diff = abs(h_crop - w_crop)
        
        pad_top, pad_bottom, pad_left, pad_right = 0, 0, 0, 0
        if h_crop > w_crop:
            pad_left = diff // 2
            pad_right = diff - pad_left
        else:
            pad_top = diff // 2
            pad_bottom = diff - pad_top
            
        c = cv2.copyMakeBorder(c, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=0)
        
        # 3. Add 15% uniform padding so it doesn't touch the edges
        pad = max(c.shape[0], c.shape[1]) * 15 // 100
        c = cv2.copyMakeBorder(c, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
        
        # 4. Resize and apply EMNIST orientation fix
        c = cv2.resize(c, (28, 28), interpolation=cv2.INTER_AREA)
        
        c = np.rot90(c, k=3)
        c = np.fliplr(c)
        c = c.astype("float32") / 255.0
        c = c.reshape(28, 28, 1)
        
        batch.append(c)
        valid_indices.append(i)
        
    if not batch:
        return "", 0
        
    batch_arr = np.array(batch)
    preds = model.predict(batch_arr, verbose=0)
    
    final_text = ""
    pred_idx = 0
    for i in range(len(final_boxes)):
        # Check for space
        if i > 0:
            prev_x, prev_y, prev_w, prev_h = final_boxes[i-1]
            x, _, _, _ = final_boxes[i]
            gap = x - (prev_x + prev_w)
            if gap > median_w * 0.6:
                final_text += " "
                
        if i in valid_indices:
            predicted_idx = int(np.argmax(preds[pred_idx]))
            letter = chr(predicted_idx + ord('a'))
            final_text += letter
            pred_idx += 1
            
    print(f"  Predicted Sentence : {final_text.upper()}")
    return final_text.upper(), len(valid_indices)


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
