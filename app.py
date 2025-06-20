import pandas as pd

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json

    # Define the exact column names used during training
    columns = [
        "vehicle_age", "km_driven", "mileage", "engine", "max_power", "seats",
        "brand_BMW", "brand_Datsun", "brand_Force", "brand_Ford", "brand_Honda", "brand_Hyundai",
        "brand_ISUZU", "brand_Isuzu", "brand_Jaguar", "brand_Jeep", "brand_Kia", "brand_Land Rover",
        "brand_Lexus", "brand_MG", "brand_Mahindra", "brand_Maruti", "brand_Mercedes-Benz", "brand_Mini",
        "brand_Nissan", "brand_Porsche", "brand_Renault", "brand_Skoda", "brand_Tata", "brand_Toyota",
        "brand_Volkswagen", "brand_Volvo", "fuel_type_Diesel", "fuel_type_Electric", "fuel_type_LPG",
        "fuel_type_Petrol", "transmission_type_Manual"
    ]

    # Convert incoming data to a DataFrame
    input_df = pd.DataFrame([data], columns=columns)

    # Predict
    prediction = model.predict(input_df)

    return jsonify({"predicted_price": float(prediction[0])})
