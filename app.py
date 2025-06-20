import pickle
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Load the trained model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# Define the expected feature order
feature_order = [
    'vehicle_age', 'km_driven', 'mileage', 'engine', 'max_power', 'seats',
    'brand_BMW', 'brand_Datsun', 'brand_Force', 'brand_Ford', 'brand_Honda',
    'brand_Hyundai', 'brand_ISUZU', 'brand_Isuzu', 'brand_Jaguar', 'brand_Jeep',
    'brand_Kia', 'brand_Land Rover', 'brand_Lexus', 'brand_MG', 'brand_Mahindra',
    'brand_Maruti', 'brand_Mercedes-Benz', 'brand_Mini', 'brand_Nissan',
    'brand_Porsche', 'brand_Renault', 'brand_Skoda', 'brand_Tata', 'brand_Toyota',
    'brand_Volkswagen', 'brand_Volvo', 'fuel_type_Diesel', 'fuel_type_Electric',
    'fuel_type_LPG', 'fuel_type_Petrol', 'transmission_type_Manual'
]

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get JSON data
        data = request.get_json()

        # Extract feature values in correct order
        input_features = [data.get(feat, 0) for feat in feature_order]

        # Convert to numpy array
        input_array = np.array([input_features])

        # Make prediction
        prediction = model.predict(input_array)[0]

        return jsonify({
            "status": "success",
            "predicted_price": round(float(prediction), 2)
        })

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

if __name__ == "__main__":
    app.run(debug=True)
