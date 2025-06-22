from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import joblib

app = Flask(__name__)
CORS(app)

# Load trained model and scaler
model = joblib.load("model.joblib")
scaler = joblib.load("scaler.joblib")

# Hardcoded feature list used during training (replace with your actual columns)
model_features = [
    'vehicle_age', 'km_driven', 'mileage', 'engine', 'max_power', 'seats', 'seller_type',
    # Example one-hot columns based on drop_first=True
    'brand_BMW', 'brand_Ford', 'brand_Honda', 'brand_Hyundai', 'brand_Maruti', 'brand_Tata', 'brand_Toyota',
    'fuel_type_Diesel', 'fuel_type_Electric', 'fuel_type_LPG', 'fuel_type_Petrol',
    'transmission_type_Manual'
    # Add all actual one-hot column names that the model was trained on
]

@app.route('/')
def home():
    return "✅ Car Price Prediction API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # 1. Receive input
        input_data = request.get_json()

        # 2. Convert to DataFrame
        df = pd.DataFrame([input_data])

        # 3. One-hot encode necessary categorical features
        categorical_cols = ['brand', 'fuel_type', 'transmission_type']
        df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

        # 4. Add missing one-hot columns
        for col in model_features:
            if col not in df.columns:
                df[col] = 0

        # 5. Reorder columns
        df = df[model_features]

        # 6. Scale numeric values
        X_scaled = scaler.transform(df)

        # 7. Predict
        prediction = model.predict(X_scaled)[0]
        return jsonify({"predicted_price": round(prediction, 2)})

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
