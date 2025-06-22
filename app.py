from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import joblib

app = Flask(__name__)
CORS(app)

# Load trained model and scaler
model = joblib.load("model.joblib")
scaler = joblib.load("scaler.joblib")

# Manually defined feature list from training (must match one-hot structure)
model_features = [
    'vehicle_age', 'km_driven', 'mileage', 'engine', 'max_power', 'seats',
    'brand_BMW', 'brand_Bentley', 'brand_Datsun', 'brand_Force', 'brand_Ford',
    'brand_Honda', 'brand_Hyundai', 'brand_ISUZU', 'brand_Isuzu', 'brand_Jaguar',
    'brand_Jeep', 'brand_Kia', 'brand_Land_Rover', 'brand_Lexus', 'brand_MG',
    'brand_Mahindra', 'brand_Maruti', 'brand_Maserati', 'brand_Mercedes_AMG',
    'brand_Mercedes_Benz', 'brand_Mini', 'brand_Nissan', 'brand_Porsche',
    'brand_Renault', 'brand_Skoda', 'brand_Tata', 'brand_Toyota',
    'brand_Volkswagen', 'brand_Volvo',
    'fuel_type_Diesel', 'fuel_type_Electric', 'fuel_type_LPG', 'fuel_type_Petrol',
    'transmission_type_Manual', 'seller_type_Dealer', 'seller_type_Individual'
]

@app.route('/')
def home():
    return "✅ Car Price Prediction API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_data = request.get_json()
        df = pd.DataFrame([input_data])

        # Sanitize special characters for dummy variable compatibility
        df.columns = df.columns.str.replace("-", "_").str.replace(" ", "_")

        # One-hot encode categorical features
        categorical_cols = ['brand', 'fuel_type', 'transmission_type', 'seller_type']
        df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

        # Ensure all required model features exist
        for col in model_features:
            if col not in df.columns:
                df[col] = 0

        # Reorder to match model
        df = df[model_features]

        # Scale numerical inputs
        X_scaled = scaler.transform(df)

        # Predict and return result
        prediction = model.predict(X_scaled)[0]
        return jsonify({"predicted_price": round(prediction, 2)})

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
