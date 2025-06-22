from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import joblib
import json

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all domains

# Load model, scaler, encoder, and feature names
model = joblib.load('model.joblib')
scaler = joblib.load('scaler.joblib')
encoder = joblib.load('encoder.joblib')

with open('columns.json') as f:
    feature_names = json.load(f)

# Categorical columns used during training
categorical_cols = ['brand', 'fuel_type', 'transmission_type', 'seller_type']

@app.route('/')
def home():
    return "✅ Car Price Prediction API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # 1. Get JSON input
        input_data = request.json

        # 2. Convert to DataFrame
        df = pd.DataFrame([input_data])

        # 3. Encode categorical columns
        df[categorical_cols] = encoder.transform(df[categorical_cols])

        # 4. Reorder columns to match training data
        df = df[feature_names]

        # 5. Scale features
        X_scaled = scaler.transform(df)

        # 6. Predict
        predicted_price = model.predict(X_scaled)[0]

        # 7. Return result
        return jsonify({"predicted_price": round(predicted_price, 2)})

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
