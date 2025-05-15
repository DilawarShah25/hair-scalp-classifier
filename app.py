from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import gdown
import os
from PIL import Image
import io

app = Flask(__name__)

# Google Drive URLs for .tflite files (replace with your shareable links)
HYBRID_TFLITE_URL = 'https://drive.google.com/file/d/1FH0_EpHgkFixk-r0vK3yx2159k-GRyOl/view?usp=drive_link'
INVALID_TFLITE_URL = 'https://drive.google.com/file/d/1qfYlBFOqJ0rnOhW2lDe24RnmrgsqrQA_/view?usp=drive_link'
INVALID_TFLITE_PATH = '/home/dilawarshah/mysite/invalid.tflite'
HYBRID_TFLITE_PATH = '/home/dilawarshah/mysite/hybrid.tflite'

# Download models from Google Drive if not already present
os.makedirs(os.path.dirname(INVALID_TFLITE_PATH), exist_ok=True)
if not os.path.exists(INVALID_TFLITE_PATH):
    gdown.download(INVALID_TFLITE_URL, INVALID_TFLITE_PATH, quiet=False)
if not os.path.exists(HYBRID_TFLITE_PATH):
    gdown.download(HYBRID_TFLITE_URL, HYBRID_TFLITE_PATH, quiet=False)

# Load TFLite models
invalid_interpreter = tf.lite.Interpreter(model_path=INVALID_TFLITE_PATH)
invalid_interpreter.allocate_tensors()
invalid_input_details = invalid_interpreter.get_input_details()
invalid_output_details = invalid_interpreter.get_output_details()

hybrid_interpreter = tf.lite.Interpreter(model_path=HYBRID_TFLITE_PATH)
hybrid_interpreter.allocate_tensors()
hybrid_input_details = hybrid_interpreter.get_input_details()
hybrid_output_details = hybrid_interpreter.get_output_details()

# Labels
INVALID_LABELS = ['invalid', 'valid']
HYBRID_LABELS = ['Alopecia_Areata', 'Androgenetic_Alopecia', 'Normal', 'Stage 1', 'Stage 2', 'Stage 3']

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400
        file = request.files['image']
        
        img = Image.open(io.BytesIO(file.read())).resize((224, 224))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0).astype(np.float32)

        invalid_interpreter.set_tensor(invalid_input_details[0]['index'], img_array)
        invalid_interpreter.invoke()
        invalid_pred = invalid_interpreter.get_tensor(invalid_output_details[0]['index'])
        invalid_class = INVALID_LABELS[np.argmax(invalid_pred)]
        invalid_probs = invalid_pred[0].tolist()

        hybrid_class = None
        hybrid_probs = None
        if invalid_class == 'valid':
            hybrid_interpreter.set_tensor(hybrid_input_details[0]['index'], img_array)
            hybrid_interpreter.invoke()
            hybrid_pred = hybrid_interpreter.get_tensor(hybrid_output_details[0]['index'])
            hybrid_class = HYBRID_LABELS[np.argmax(hybrid_pred)]
            hybrid_probs = hybrid_pred[0].tolist()

        return jsonify({
            'invalid_class': invalid_class,
            'hybrid_class': hybrid_class,
            'invalid_probs': invalid_probs,
            'hybrid_probs': hybrid_probs
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)