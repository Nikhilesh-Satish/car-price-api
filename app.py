from flask import Flask, request, jsonify
import numpy as np
import joblib

app = Flask(__name__)
model = joblib.load("model.joblib")

@app.route('/')
def home():
    return "Car Price Prediction API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    input_data = [list(data.values())]
    prediction = model.predict(input_data)
    
    # Convert np.float32 → Python float
    output = float(prediction[0])
    return jsonify({'predicted_price': output})
