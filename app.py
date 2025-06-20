from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np

app = Flask(__name__)
CORS(app)

# Load the trained model
model = joblib.load('model.joblib')

# Define the expected feature columns
feature_names = [
    'vehicle_age', 'km_driven', 'mileage', 'engine', 'max_power', 'seats',
    'brand_BMW', 'brand_Datsun', 'brand_Force', 'brand_Ford', 'brand_Honda', 'brand_Hyundai',
    'brand_ISUZU', 'brand_Isuzu', 'brand_Jaguar', 'brand_Jeep', 'brand_Kia', 'brand_Land Rover',
    'brand_Lexus', 'brand_MG', 'brand_Mahindra', 'brand_Maruti', 'brand_Mercedes-Benz', 'brand_Mini',
    'brand_Nissan', 'brand_Porsche', 'brand_Renault', 'brand_Skoda', 'brand_Tata', 'brand_Toyota',
    'brand_Volkswagen', 'brand_Volvo', 'fuel_type_Diesel', 'fuel_type_Electric', 'fuel_type_LPG',
    'fuel_type_Petrol', 'transmission_type_Manual'
]

@app.route('/')
def index():
    return "Car Price Predictor API is running"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json  # Expecting a JSON object with all features
        input_data = []

        for feature in feature_names:
            value = data.get(feature, 0)
            input_data.append(float(value))

        input_array = np.array(input_data).reshape(1, -1)
        prediction = model.predict(input_array)

        return jsonify({'predicted_price': round(prediction[0], 2)})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
