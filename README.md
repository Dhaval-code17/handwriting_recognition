# Handwriting Recognition

Handwritten Character & Sentence Recognition built with custom Convolutional Neural Networks (CNN) on MNIST and EMNIST datasets.

## Overview
This is a Flask web application that provides an interactive drawing canvas to recognize handwritten digits, individual letters, and full cursive/print sentences. 

Instead of relying on heavy third-party sequence models, this application uses a highly optimized, fully custom pipeline:
1. **Custom AI Models**: Trained entirely from scratch on the MNIST (digits) and EMNIST (letters) datasets using TensorFlow/Keras.
2. **Advanced OpenCV Segmentation**: Uses `cv2.findContours` to separate overlapping letters mathematically without splitting 'i' dots or 'j' dots.
3. **Aspect Ratio & Stroke Normalization**: Automatically thickens pen strokes and adds square padding to preserve the aspect ratio, perfectly mimicking the data the models were trained on.
4. **Lightning-Fast Batch Prediction**: Processes full sentences by grouping all separated character crops into a single tensor array, running inference in under ~50 milliseconds.

## Setup and Run Instructions

Follow these steps to run the project locally on your machine.

### 1. Clone the repository
```bash
git clone https://github.com/Dhaval-code17/handwriting_recognition.git
cd handwriting_recognition
```

### 2. Create and activate a virtual environment
**For Windows:**
```cmd
python -m venv venv
venv\Scripts\activate
```

**For macOS and Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies
Install the required packages using `requirements.txt`:
```bash
pip install -r requirements.txt
```

### 4. Run the application
Start the Flask development server:
```bash
python app.py
```

### 5. Open in Browser
Once the server is running, open your web browser and navigate to:
[http://127.0.0.1:5000](http://127.0.0.1:5000)

## How to Use
- Select the mode: **Digit**, **Letter**, or **Sentence**.
- Draw your character(s) or sentence on the canvas. 
  - *Tip for Sentence Mode: Write with slight physical gaps between letters (print handwriting) for near 100% accuracy!*
- Click **Predict** to see the results dynamically appear on the screen!
