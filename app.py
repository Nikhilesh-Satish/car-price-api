from flask import Flask, request, jsonify
import joblib
import numpy as np

# Create Flask app
app = Flask(__name__)

# Load the model
model = joblib.load("model.joblib")

# Root route (optional for testing)
@app.route('/', methods=['GET'])
def home():
    return "Car Price Prediction API is running!"

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON input
        data = request.get_json(force=True)

        # Convert input values to 2D array
        input_data = [list(data.values())]

        # Make prediction
        prediction = model.predict(input_data)

        # Convert NumPy float to native Python float
        output = float(prediction[0])

        return jsonify({'predicted_price': output})

    except Exception as e:
        return jsonify({'error': str(e)}), 400

# Run app locally (ignored in production)
if __name__ == '__main__':
    app.run(debug=True)
