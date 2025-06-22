from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import joblib

app = Flask(__name__)
CORS(app)

# Load model components
model = joblib.load("model.joblib")
scaler = joblib.load("scaler.joblib")
encoder = joblib.load("encoder.joblib")
model_features = joblib.load("model_features.joblib")

# Columns that were encoded during training
categorical_cols = ['brand', 'fuel_type', 'transmission_type', 'seller_type']

@app.route("/")
def home():
    return "✅ Car Price Prediction API is running!"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Step 1: Load input JSON
        input_data = request.get_json()
        df = pd.DataFrame([input_data])

        # Step 2: Encode categorical features
        df[categorical_cols] = encoder.transform(df[categorical_cols])

        # Step 3: Ensure column order matches training
        df = df[model_features]

        # Step 4: Scale features
        X_scaled = scaler.transform(df)

        # Step 5: Predict
        prediction = model.predict(X_scaled)[0]
        return jsonify({"predicted_price": round(prediction, 2)})

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True)
