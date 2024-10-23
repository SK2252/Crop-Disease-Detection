from flask import Flask, request, jsonify
import os
import torch
from ultralytics import YOLO
from PIL import Image
import io
import tempfile
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Global variable to store image path
image_path = None

# Create a temporary cache directory
cache_dir = tempfile.mkdtemp()

# Load YOLO model
leaf_model = YOLO('leaf_detection_100.pt')
#print(leaf_model)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

leaf_model.to(device)

@app.route('/save-image', methods=['POST'])
def save_image():
    global image_path
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400

        # Get the image from the request
        image_file = request.files['image']
        # Save the image to the cache directory
        image_path = os.path.join(cache_dir, image_file.filename)
        image_file.save(image_path)
        
        return jsonify({'message': 'Image saved successfully'}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predict', methods=['POST'])
def predict():
    global image_path
    try:
        if not image_path or not os.path.exists(image_path):
            return jsonify({'error': 'No image found for prediction'}), 400

        # Load the saved image
        image = Image.open(image_path)

        # Run prediction using YOLO model
        results = leaf_model(image)

        # Check if any result has confidence > 60%
        for result in results:
            for conf in result.boxes.conf:
                if conf > 0.60:
                    return jsonify({'result': True}), 200

        return jsonify({'result': False}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)