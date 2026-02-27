import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify, render_template
from keras.models import load_model
import requests
from io import BytesIO
from PIL import Image
import sys

app = Flask(__name__)

# --- MODEL LOADING ---
try:
    # Ensure 'cat_dog_model.h5' is in the same folder as this script
    classifymodel = load_model('cat_dog_model.h5')
    # Automatically detect the input size the model was trained on
    # Most Keras models have input_shape like (None, 256, 256, 3)
    target_h = classifymodel.input_shape[1]
    target_w = classifymodel.input_shape[2]
    print(f"SUCCESS: Model loaded. Expected input size: {target_w}x{target_h}")
except Exception as e:
    print(f"CRITICAL: Could not load model file. Error: {e}")
    classifymodel = None
    target_h, target_w = 256, 256 # Fallback default

@app.route('/')
def home():
    return render_template('pet.html')

def preprocess_image(img_data, is_url=False):
    """Helper to process image from bytes or stream into model-ready array."""
    if is_url:
        img = Image.open(BytesIO(img_data)).convert('RGB')
    else:
        img = Image.open(img_data).convert('RGB')
    
    # RESIZE: Using the target dimensions detected from the model
    img = img.resize((target_w, target_h)) 
    img_array = np.array(img) / 255.0
    # Expand dims adds the batch dimension: (1, height, width, 3)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.route('/predict', methods=['POST'])
def predict():
    if classifymodel is None:
        return jsonify({"error": "Model not loaded"}), 500

    if 'file' not in request.files:
        return jsonify({"error": "Missing file", "message": "No file part in the request"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    try:
        img_array = preprocess_image(file.stream)
        prediction = classifymodel.predict(img_array)
        
        # Result logic: 1 is Dog, 0 is Cat (standard binary classification)
        result = "Dog" if prediction[0][0] > 0.5 else "Cat"
        confidence = float(prediction[0][0]) if result == "Dog" else float(1 - prediction[0][0])

        return jsonify({
            'prediction': result,
            'confidence': round(confidence, 4),
            'input_shape_used': f"{target_w}x{target_h}"
        })

    except Exception as e:
        return jsonify({"error": "Processing Error", "message": str(e)}), 500

@app.route('/predict-url', methods=['POST'])
def predict_url():
    if classifymodel is None:
        return jsonify({"error": "Model not loaded"}), 500

    try:
        data = request.get_json()
        if not data or 'url' not in data:
            return jsonify({"error": "Missing URL"}), 400
        
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(data['url'], headers=headers, timeout=10)
        
        img_array = preprocess_image(response.content, is_url=True)
        prediction = classifymodel.predict(img_array)
        
        result = "Dog" if prediction[0][0] > 0.5 else "Cat"
        confidence = float(prediction[0][0]) if result == "Dog" else float(1 - prediction[0][0])
        
        return jsonify({
            'prediction': result,
            'confidence': round(confidence, 4),
            'status': 'success'
        })

    except Exception as e:
        return jsonify({"error": "Processing Error", "message": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)