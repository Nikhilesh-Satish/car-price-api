from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import joblib

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all domains

# Load trained model and scaler
model = joblib.load('model.joblib')
scaler = joblib.load('scaler.joblib')

@app.route('/')
def home():
    return "✅ Car Price Prediction API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # 1. Get input JSON from frontend
        input_data = request.json

        # 2. Convert to DataFrame
        df = pd.DataFrame([input_data])

        # 3. Scale numeric features (all columns assumed numerical or already one-hot)
        X_scaled = scaler.transform(df)

        # 4. Predict using the trained model
        predicted_price = model.predict(X_scaled)[0]

        # 5. Return prediction
        return jsonify({"predicted_price": round(predicted_price, 2)})

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
