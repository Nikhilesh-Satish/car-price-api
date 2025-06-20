from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np

app = Flask(__name__)
CORS(app)

# Load your model
model = joblib.load("model.joblib")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json

        # Convert incoming JSON values to 2D array
        features = np.array([list(data.values())]).astype(float)

        prediction = model.predict(features)
        return jsonify({'predicted_price': float(prediction[0])})
    
    except Exception as e:
        return jsonify({'error': str(e)})
