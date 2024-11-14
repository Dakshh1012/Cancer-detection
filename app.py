from flask import Flask, jsonify, request, render_template, send_file
from keras.models import load_model
import numpy as np
from PIL import Image
import logging
import joblib
import pandas as pd

app = Flask(__name__)

model = load_model('model.keras')
forest_model = joblib.load('forest_model.joblib')

logging.basicConfig(level=logging.INFO)

@app.route("/")
def home():
    return render_template('index.html')

@app.route("/form_page")
def form():
    return render_template('form_page.html')

@app.route("/AIDoctorLogo.png")
def get_logo():
    return send_file('templates/AIDoctorLogo.png')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        app.logger.error('No file provided')
        return jsonify({'error': 'No file provided'}), 400

    image_file = request.files['file']
    app.logger.info('File received: %s', image_file.filename)
    
    try:
        image = preprocess_image(image_file)
        prediction = model.predict(image)
        label = np.argmax(prediction, axis=1)[0]
        prediction = ""
        if label == 2:
            prediction = "No Tumour"
        elif label == 3:
            prediction = "Pituitary"
        elif label == 1:
            prediction = "Meningioma"
        else:
            prediction = "Glioma"
        return jsonify({'label': prediction})
    except Exception as e:
        app.logger.error('Error processing file: %s', str(e))
        return jsonify({'error': str(e)}), 500

@app.route('/predict_forest', methods=['POST'])
def predict_forest():
    try:
        data = request.get_json()
        input_df = pd.DataFrame([data], columns=[
            "radius_mean", "texture_mean", "perimeter_mean", "area_mean", "smoothness_mean", 
            "compactness_mean", "concavity_mean", "concave points_mean", "symmetry_mean", 
            "fractal_dimension_mean", "radius_se", "texture_se", "perimeter_se", "area_se", 
            "smoothness_se", "compactness_se", "concavity_se", "concave points_se", 
            "symmetry_se", "fractal_dimension_se", "radius_worst", "texture_worst", 
            "perimeter_worst", "area_worst", "smoothness_worst", "compactness_worst", 
            "concavity_worst", "concave points_worst", "symmetry_worst", 
            "fractal_dimension_worst"
        ])
        input_df['id'] = 842302
        prediction = forest_model.predict(input_df)
        return jsonify({'diagnosis': "Malignant" if int(prediction[0]) == 1 else "Benign"})
    except Exception as e:
        app.logger.error('Error during prediction: %s', str(e))
        return jsonify({'error': str(e)}), 500

def preprocess_image(image_file):
    image = Image.open(image_file)
    image = image.resize((150, 150))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

if __name__ == '__main__':
    app.run(debug=True)
