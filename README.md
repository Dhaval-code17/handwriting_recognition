# Handwriting Recognition

Handwritten Character Recognition using CNN on MNIST and EMNIST datasets.

## Overview
This is a Flask web application that provides a drawing canvas UI to recognize handwritten digits, letters, and sentences. It uses OpenCV for character segmentation and deep learning models for prediction.

## Prerequisites
- Python 3.8+
- pip (Python package installer)

## Setup and Run Instructions

Follow these steps to run the project locally on your machine.

### 1. Clone the repository
```bash
git clone <your-github-repo-url>
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

## How to use
- Select the mode: **Digit**, **Letter**, or **Sentence**.
- Draw your character(s) on the canvas.
- Click "Predict" to see the results.
