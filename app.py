from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import joblib

app = Flask(__name__)
CORS(app)

# Load model, scaler, and encoder
model = joblib.load("model.joblib")
scaler = joblib.load("scaler.joblib")
encoder = joblib.load("encoder.joblib")

# Categorical columns encoded during training
categorical_cols = ['brand', 'fuel_type', 'transmission_type', 'seller_type']

@app.route('/')
def home():
    return "✅ Car Price Prediction API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # 1. Get input JSON
        input_data = request.get_json()
        df = pd.DataFrame([input_data])

        # 2. Encode categorical columns
        df[categorical_cols] = encoder.transform(df[categorical_cols])

        # 3. Scale all features
        X_scaled = scaler.transform(df)

        # 4. Predict
        prediction = model.predict(X_scaled)[0]
        return jsonify({"predicted_price": round(prediction, 2)})

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
