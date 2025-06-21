from flask import Flask, request, jsonify
from flask_cors import CORS  # ✅ Import CORS
import joblib
import numpy as np

app = Flask(__name__)
CORS(app)  # ✅ Allow all origins for now (or pass origins=["http://localhost:5173"])

# Load model
model = joblib.load("model.joblib")

@app.route('/', methods=['GET'])
def home():
    return "Car Price Prediction API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        input_data = [list(data.values())]
        prediction = model.predict(input_data)
        output = float(prediction[0])
        return jsonify({'predicted_price': output})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
