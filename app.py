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
        from src.predict import predict_sentence_bbox
        predicted_text, char_count = predict_sentence_bbox(img_array)
        print(f"Detected {char_count} characters.")
        
        return jsonify({
            "mode": "sentence",
            "text": predicted_text,
            "count": char_count
        })

    return jsonify({"error": f"Unknown mode: {mode}"}), 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
