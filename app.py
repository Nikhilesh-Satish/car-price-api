from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib

# ✅ Create Flask app instance
app = Flask(__name__)
CORS(app)

# ✅ Example route
@app.route('/')
def home():
    return "API is live"

# ✅ Predict route
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    # Example model prediction (make sure model.pkl exists and is loaded correctly)
    # model = joblib.load("model.pkl")
    # prediction = model.predict([[data['some_feature'], ...]])
    return jsonify({"predicted_price": 123456.78})

# ✅ Required only when running locally
# if __name__ == "__main__":
#     app.run(debug=True)
