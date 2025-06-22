from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import joblib

app = Flask(__name__)
CORS(app)

# Load trained model components
model = joblib.load("model.joblib")
scaler = joblib.load("scaler.joblib")
encoder = joblib.load("encoder.joblib")
model_features = joblib.load("model_features.joblib")  # ['vehicle_age', 'km_driven', ..., 'seller_type']

# Define categorical columns (must match those used in training)
categorical_cols = ['brand', 'fuel_type', 'transmission_type', 'seller_type']

@app.route("/")
def home():
    return "✅ Car Price Prediction API is running!"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Step 1: Load input
        input_data = request.get_json()
        df = pd.DataFrame([input_data])

        # Step 2: Reorder columns to match expected order
        df = df[model_features]  # Important to maintain same order

        # Step 3: Encode categorical columns using same encoder
        df[categorical_cols] = encoder.transform(df[categorical_cols])

        # Step 4: Scale all features
        X_scaled = scaler.transform(df)

        # Step 5: Predict
        prediction = model.predict(X_scaled)[0]
        return jsonify({"predicted_price": round(prediction, 2)})

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True)
