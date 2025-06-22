from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import joblib

# Initialize app
app = Flask(__name__)
CORS(app)

# Load model and scaler
model = joblib.load("model.joblib")
scaler = joblib.load("scaler.joblib")

# Model was trained on these features — update this list based on training
model_features = [
    'vehicle_age', 'km_driven', 'mileage', 'engine', 'max_power', 'seats', 'seller_type',
    # One-hot encoded brand (drop_first=True)
    'brand_BMW', 'brand_Ford', 'brand_Honda', 'brand_Hyundai', 'brand_Maruti', 'brand_Tata', 'brand_Toyota',
    # One-hot encoded fuel type (drop_first=True)
    'fuel_type_Diesel', 'fuel_type_Electric', 'fuel_type_LPG', 'fuel_type_Petrol',
    # One-hot encoded transmission (drop_first=True)
    'transmission_type_Manual'
]

# Define which features to one-hot encode
categorical_cols = ['brand', 'fuel_type', 'transmission_type']

@app.route('/')
def home():
    return "✅ Car Price Prediction API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # 1. Receive JSON input
        input_data = request.get_json()

        # 2. Convert input to DataFrame
        df = pd.DataFrame([input_data])

        # 3. One-hot encode categorical columns (drop_first=True to match training)
        df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

        # 4. Ensure all expected features are present
        for col in model_features:
            if col not in df_encoded.columns:
                df_encoded[col] = 0  # Add missing column with 0

        # 5. Ensure correct column order
        df_encoded = df_encoded[model_features]

        # 6. Scale input features
        X_scaled = scaler.transform(df_encoded)

        # 7. Predict
        prediction = model.predict(X_scaled)[0]

        # 8. Return result
        return jsonify({"predicted_price": round(prediction, 2)})

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
