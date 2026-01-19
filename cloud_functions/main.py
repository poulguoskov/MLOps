"""Cloud Function for iris classification."""

import pickle

import functions_framework
from google.cloud import storage

# Global model variable (loaded once per instance)
model = None


def load_model():
    """Load model from GCS bucket."""
    global model
    if model is None:
        client = storage.Client()
        bucket = client.bucket("poul-mlops_models")
        blob = bucket.blob("model.pkl")
        blob.download_to_filename("/tmp/model.pkl")
        with open("/tmp/model.pkl", "rb") as f:
            model = pickle.load(f)
    return model


@functions_framework.http
def classify_iris(request):
    """HTTP Cloud Function for iris classification."""
    request_json = request.get_json(silent=True)

    if not request_json or "data" not in request_json:
        return {"error": "Please provide 'data' with 4 features"}, 400

    data = request_json["data"]
    if len(data) != 4:
        return {"error": "Expected 4 features"}, 400

    model = load_model()
    prediction = model.predict([data])

    class_names = ["setosa", "versicolor", "virginica"]

    return {
        "input": data,
        "prediction": int(prediction[0]),
        "class_name": class_names[prediction[0]],
    }
