# ---------------------------------------------------------
# Imports
# ---------------------------------------------------------
from flask import Flask, render_template, request, jsonify
import numpy as np
import logging
import os  # Added for path handling

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

# --- FIX: ROBUST PATH HANDLING ---
# This gets the absolute path of the directory where app.py is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# This joins that path with the 'model' folder and filename
MODEL_PATH = os.path.join(BASE_DIR, "model", "model_cancer_predictor.h5")
# ---------------------------------

THRESHOLD = 0.5

# ---------------------------------------------------------
# Load Model & Preprocessing Objects
# ---------------------------------------------------------
logger.info(f"Attempting to load ANN model from: {MODEL_PATH}")

if not os.path.exists(MODEL_PATH):
    logger.error(f"CRITICAL ERROR: Model file not found at {MODEL_PATH}")
else:
    try:
        model = load_model(MODEL_PATH)
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Error loading model: {e}")

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
    result = None
    probability = None
    prediction_class = None

    if request.method == "POST":
        try:
            input_features = [
                float(request.form[feature])
                for feature in FEATURE_NAMES
            ]

            input_array = np.array(input_features).reshape(1, -1)
            input_scaled = scaler.transform(input_array)

            prob = float(model.predict(input_scaled, verbose=0)[0][0])
            probability = round(prob, 4)

            # Note: Checking threshold logic (0.5 typically means >= 0.5 is Benign in your code)
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
    return jsonify(
        status="ok",
        model_loaded=True
    ), 200

# ---------------------------------------------------------
# Application Entry Point
# ---------------------------------------------------------
if __name__ == "__main__":
    # Added host and port for easier local testing
    app.run(debug=True, host="127.0.0.1", port=5000)