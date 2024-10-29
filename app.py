from flask import Flask, jsonify, request, render_template, send_file
from keras.models import load_model
import numpy as np
from PIL import Image
import logging

# Initialize the Flask app
app = Flask(__name__)

# Load the trained model
model = load_model('model.keras')

# Configure logging
logging.basicConfig(level=logging.INFO)

@app.route("/")
def home():
    return render_template('index.html')

@app.route("/AIDoctorLogo.png")
def get_logo():
    return send_file('templates/AIDoctorLogo.png')

@app.route('/predict', methods=['POST'])
def predict():
    # Check if the file part is in the request
    if 'file' not in request.files:
        app.logger.error('No file provided')
        return jsonify({'error': 'No file provided'}), 400

    image_file = request.files['file']
    app.logger.info('File received: %s', image_file.filename)
    
    try:
        # Preprocess the image
        image = preprocess_image(image_file)
        # Make predictions
        prediction = model.predict(image)
        label = np.argmax(prediction, axis=1)[0]
        return jsonify({'label': int(label)})
    except Exception as e:
        app.logger.error('Error processing file: %s', str(e))
        return jsonify({'error': str(e)}), 500

def preprocess_image(image_file):
    image = Image.open(image_file)
    image = image.resize((150, 150))  # Adjust the size based on your model's requirement
    image = np.array(image) / 255.0   # Normalize the image
    image = np.expand_dims(image, axis=0)  # Expand dimensions to fit the model input
    return image

if __name__ == '__main__':
    app.run(debug=True)
