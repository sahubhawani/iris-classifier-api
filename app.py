from flask import Flask, request, jsonify
import joblib
import numpy as np

model = joblib.load("iris_classifier_model.pkl")

app = Flask(__name__)

@app.route("/", methods = ["GET"])
def home():
    return "The API is working"

@app.route("/predict", methods = ["POST"])
def predict():
    # extract the data from the request
    data = request.get_json(force=True)
    features = data["features"]
    features = np.array(features).reshape(1,-1)

    prediction = model.predict(features)[0]
    classes = ["Setosa", "Versicolor", "Verginica"]
    pred_class = classes[prediction]
    
    result = {"Predicted Class": pred_class}


    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True)