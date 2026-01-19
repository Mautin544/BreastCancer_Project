"""
Breast Cancer Prediction Web App
================================
Production-ready Flask application that serves a trained
Artificial Neural Network (ANN) for breast cancer classification.

Author: Noah Babadina
Model: Feed-Forward Neural Network (Keras)
Deployment Target: Render
"""

# ---------------------------------------------------------
# Imports
# ---------------------------------------------------------
from flask import Flask, render_template, request, jsonify
import numpy as np
import logging

from tensorflow.keras.models import load_model
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------
# App Configuration
# ---------------------------------------------------------
app = Flask(__name__)

# Logging configuration (Render-compatible)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)

MODEL_PATH = "model/model_cancer_predictor.h5"
THRESHOLD = 0.5

# ---------------------------------------------------------
# Load Model & Preprocessing Objects
# ---------------------------------------------------------
logger.info("Loading ANN model...")
model = load_model(MODEL_PATH)
logger.info("Model loaded successfully")

# Load dataset to recreate scaler exactly as during training
data = load_breast_cancer(as_frame=True)
X = data.data
FEATURE_NAMES = list(X.columns)

scaler = StandardScaler()
scaler.fit(X)

# ---------------------------------------------------------
# Routes
# ---------------------------------------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    """
    Main application page.
    Allows users to input tumor features and receive predictions.
    """
    result = None
    probability = None
    prediction_class = None

    if request.method == "POST":
        try:
            # Extract feature values in correct order
            input_features = [
                float(request.form[feature])
                for feature in FEATURE_NAMES
            ]

            input_array = np.array(input_features).reshape(1, -1)
            input_scaled = scaler.transform(input_array)

            # Model prediction
            prob = float(model.predict(input_scaled, verbose=0)[0][0])
            probability = round(prob, 4)

            if prob >= THRESHOLD:
                result = "Benign Tumor"
                prediction_class = "benign"
            else:
                result = "Malignant Tumor"
                prediction_class = "malignant"

            logger.info(
                f"Prediction made | Result: {result} | Probability: {probability}"
            )

        except Exception as e:
            logger.error(f"Prediction error: {e}")
            result = "Invalid input values. Please check your entries."
            prediction_class = "error"

    return render_template(
        "index.html",
        features=FEATURE_NAMES,
        prediction=result,
        prediction_class=prediction_class,
        probability=probability
    )


@app.route("/health", methods=["GET"])
def health_check():
    """
    Health check endpoint for monitoring and Render verification.
    """
    return jsonify(
        status="ok",
        model_loaded=True
    ), 200


# ---------------------------------------------------------
# Application Entry Point
# ---------------------------------------------------------
if __name__ == "__main__":
    app.run(debug=False)
