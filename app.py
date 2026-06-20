"""
Flask web application for handwritten character recognition.

Routes:
    GET  /         -> serves templates/index.html (drawing canvas UI)
    POST /predict  -> accepts base64 image + mode, returns prediction

Supported modes (passed as JSON field "mode"):
    "digit"    -> predict a single handwritten digit 0-9
    "letter"   -> predict a single handwritten letter a-z
    "sentence" -> segment the image into characters, predict each as a
                  letter, and return the full joined string
"""

import os
import sys
import io
import base64
import numpy as np
import cv2

# Ensure project root is on the path so `src` imports work correctly
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flask import Flask, request, jsonify, render_template
from PIL import Image
from src.predict import predict_from_array, predict_letter_from_array

app = Flask(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _decode_image(data_url):
    """Decode a base64 data-URL into a grayscale numpy array (H, W) uint8."""
    image_data = data_url
    if "," in image_data:
        image_data = image_data.split(",", 1)[1]        # strip data-URL prefix
    img_bytes = base64.b64decode(image_data)
    pil_img   = Image.open(io.BytesIO(img_bytes)).convert("L")  # grayscale
    return np.array(pil_img, dtype=np.uint8)                    # (H, W) uint8


def should_merge(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    
    # Calculate areas
    area1 = w1 * h1
    area2 = w2 * h2
    
    # Only merge if one box is MUCH smaller (likely a dot) 
    # AND it's positioned above the other box (small y range overlap)
    smaller_area = min(area1, area2)
    larger_area = max(area1, area2)
    
    if smaller_area < larger_area * 0.15:  # one is much smaller (dot-like)
        # Check horizontal overlap - dot should be roughly above the letter
        x_overlap = min(x1+w1, x2+w2) - max(x1, x2)
        min_width = min(w1, w2)
        if x_overlap > min_width * 0.3:  # significant horizontal overlap
            return True
    
    return False


def predict_sentence(gray_img):
    import cv2
    import numpy as np
    from src.predict import predict_letter_from_array

    # 1. Threshold - image is now white background, black text
    # So invert first to get white text on black (easier for contours)
    _, binary = cv2.threshold(gray_img, 180, 255, cv2.THRESH_BINARY_INV)
    
    # 2. Find contours of each character blob
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 3. Get bounding boxes, filter noise
    boxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w * h > 100 and h > 10:  # filter tiny noise
            boxes.append((x, y, w, h))
    
    if not boxes:
        return "", 0
    
    # 4. Sort left to right
    boxes.sort(key=lambda b: b[0])
    
    # 4.5 Compute median height to detect dot-like boxes
    heights = [h for x, y, w, h in boxes]
    median_h = sorted(heights)[len(heights) // 2]
    
    print(f"DEBUG: median_h = {median_h}")
    for x, y, w, h in boxes:
        is_dot = h < median_h * 0.5
        print(f"DEBUG Box: x={x}, y={y}, w={w}, h={h}, is_dot={is_dot}")
    
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
    
    # 5.5 Over-segmentation for touching characters
    normal_widths = [w for (x, y, w, h) in merged if h >= median_h * 0.5]
    if not normal_widths:
        normal_widths = [w for (x, y, w, h) in merged]
    base_w = sorted(normal_widths)[len(normal_widths) // 2]
    
    final_boxes = []
    for (x, y, w, h) in merged:
        is_dot = h < median_h * 0.5
        n_chars = int(round(w / base_w))
        if n_chars > 1 and not is_dot:
            # Column-wise sum of nonzero pixels
            char_img = binary[y:y+h, x:x+w]
            col_sums = np.sum(char_img > 0, axis=0)
            
            margin = int(w * 0.15)
            if w - 2 * margin > n_chars:
                cut_points = []
                for j in range(1, n_chars):
                    expected_cut = int(w * j / n_chars)
                    search_range = max(1, int(base_w * 0.3))
                    start_c = max(margin, expected_cut - search_range)
                    end_c = min(w - margin, expected_cut + search_range)
                    
                    if start_c >= end_c:
                        pass
                    else:
                        local_min_idx = start_c + np.argmin(col_sums[start_c:end_c])
                        cut_points.append(local_min_idx)
                
                valid_cuts = [c for c in cut_points if col_sums[c] <= h * 0.1]
                print(f"DEBUG Split: box_x={x}, orig_w={w}, computed n_chars={n_chars}, candidate_valleys={cut_points}, valid={valid_cuts}")
                
                if valid_cuts:
                    prev_cut = 0
                    import os
                    os.makedirs('data', exist_ok=True)
                    for idx, cut in enumerate(valid_cuts + [w]):
                        sub_w = cut - prev_cut
                        sub_x = x + prev_cut
                        final_boxes.append((sub_x, y, sub_w, h))
                        sub_crop = binary[y:y+h, sub_x:sub_x+sub_w]
                        cv2.imwrite(f'data/split_char_{idx}.png', sub_crop)
                        prev_cut = cut
                else:
                    final_boxes.append((x, y, w, h))
            else:
                final_boxes.append((x, y, w, h))
        else:
            final_boxes.append((x, y, w, h))
            
    final_boxes.sort(key=lambda b: b[0])
    
    # 5.6 Clean up implausibly narrow stray strokes (e.g. disconnected cursive tails)
    cleaned_boxes = []
    for box in final_boxes:
        x, y, w, h = box
        # Only merge if it's very narrow AND extremely close to the previous box
        if h >= median_h * 0.5 and w < base_w * 0.25:
            if cleaned_boxes:
                px, py, pw, ph = cleaned_boxes[-1]
                if x - (px + pw) < 10:  # Very close
                    new_x = min(px, x)
                    new_y = min(py, y)
                    new_x2 = max(px + pw, x + w)
                    new_y2 = max(py + ph, y + h)
                    cleaned_boxes[-1] = (new_x, new_y, new_x2 - new_x, new_y2 - new_y)
                    print(f"DEBUG: Merged stray narrow box x={x}, w={w} into previous box")
                    continue
            cleaned_boxes.append(box)
        else:
            cleaned_boxes.append(box)
    
    final_boxes = cleaned_boxes
    
    # 6. Calculate median character width for space detection
    widths = [w for x, y, w, h in final_boxes]
    median_w = sorted(widths)[len(widths) // 2]
    
    # 7. Predict each character and detect spaces
    result = ""
    for i, (x, y, w, h) in enumerate(final_boxes):
        # Check for space before this character
        if i > 0:
            prev_x, prev_y, prev_w, prev_h = final_boxes[i-1]
            gap = x - (prev_x + prev_w)
            # Space if gap > 60% of median character width
            if gap > median_w * 0.6:
                result += " "
        
        # Crop character from binary image
        char_img = binary[y:y+h, x:x+w]
        
        if i == 0:
            import os
            os.makedirs('data', exist_ok=True)
            cv2.imwrite('data/sent_char.png', char_img)
        
        if char_img.size == 0:
            continue
            
        # Predict using EMNIST letter model
        letter, confidence = predict_letter_from_array(char_img)
        result += letter
    
    return result.upper(), len(merged)

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/", methods=["GET"])
def index():
    """Serve the drawing canvas UI."""
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    """Receive a base64-encoded canvas image and return a prediction.

    Request JSON:
        {
          "image": "<base64 data-url string>",
          "mode":  "digit" | "letter" | "sentence"   (default: "digit")
        }

    Response JSON (digit mode):
        { "mode": "digit",    "digit":      <int>,  "confidence": <float> }

    Response JSON (letter mode):
        { "mode": "letter",   "letter":     <str>,  "confidence": <float> }

    Response JSON (sentence mode):
        { "mode": "sentence", "sentence":   <str>,  "char_count": <int> }
    """
    data = request.get_json()
    if not data or "image" not in data:
        return jsonify({"error": "No image provided"}), 400

    mode      = data.get("mode", "digit").lower()
    img_array = _decode_image(data["image"])   # (H, W) uint8 grayscale

    # ── Digit mode ────────────────────────────────────────────────────────────
    if mode == "digit":
        digit, confidence = predict_from_array(img_array)
        return jsonify({
            "mode":       "digit",
            "digit":      digit,
            "confidence": round(confidence, 2),
        })

    # ── Letter mode ───────────────────────────────────────────────────────────
    if mode == "letter":
        letter, confidence = predict_letter_from_array(img_array)
        return jsonify({
            "mode":       "letter",
            "letter":     letter,
            "confidence": round(confidence, 2),
        })

    # ── Sentence mode ─────────────────────────────────────────────────────────
    if mode == "sentence":
        predicted_text, char_count = predict_sentence(img_array)
        print(f"Detected {char_count} characters.")
        
        return jsonify({
            "mode": "sentence",
            "text": predicted_text,
            "count": char_count
        })

    return jsonify({"error": f"Unknown mode: {mode}"}), 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
